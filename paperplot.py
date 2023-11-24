####
# 1. training curves with comparison to state-of-the-art
####

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import glob
import numpy as np
import os
from sklearn.manifold import TSNE
import pickle
from utils import *
from test_one_staged_lqr_pixel import *
from scipy import signal
import control

import torchvision
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO


def build_agent(exp_config_path):
    # load yaml configuration
    with open(exp_config_path+'.yaml') as file:
        exp_config = yaml.safe_load(file)
    utils.set_seed_everywhere(exp_config['seed'])

    # set environment
    env = dmc2gym.make(
        domain_name=exp_config['domain_name'],
        task_name=exp_config['task_name'],
        seed=exp_config['seed'],
        visualize_reward=False,
        from_pixels=(exp_config['env']['encoder_type'] == 'pixel'),
        height=exp_config['env']['pre_transform_image_size'],
        width=exp_config['env']['pre_transform_image_size'],
        frame_skip=exp_config['env']['action_repeat'])
    env.seed(exp_config['seed'])

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # stack several consecutive frames together
    if exp_config['env']['encoder_type'] == 'pixel':
        env = utils.FrameStack(env, k=exp_config['env']['frame_stack'])
    if exp_config['env']['encoder_type'] == 'pixel':
        obs_shape = (3*exp_config['env']['frame_stack'], exp_config['env']['image_size'], exp_config['env']['image_size'])
        pre_aug_obs_shape = (3*exp_config['env']['frame_stack'],exp_config['env']['pre_transform_image_size'],exp_config['env']['pre_transform_image_size'])
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape
    action_shape = env.action_space.shape

    # make agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        config=exp_config,
        device=device)

    return agent

def smooth(scalars, weight): # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plot_train_curve(src_dir, step_episode_ratio, optimal_cost, config={"show_error_percent":True, "show_smoothed":True, "num_lines":"all", "smooth_ratio":0.6}):
    all_data = pd.DataFrame()
    for dir in src_dir:
        csv_files = glob.glob(dir + '/*.csv')
        assert len(csv_files) == 1
        data = pd.read_csv(csv_files[0])
        if config['num_lines'] != "all":
            data = data[:config['num_lines']]
        # process plain cost value
        value = data['Value'] if np.all(data['Value'] < 0)  else -data['Value']
        smoothed_value = smooth(scalars=value, weight=config["smooth_ratio"])
        data['Smoothed_Value'] = smoothed_value

        # add error percentage
        err_percent = [100*(optimal_cost-x)/optimal_cost for x in value]
        err_percent = [0 if err<0 else err for err in err_percent]
        smoothed_err_percent = smooth(scalars=err_percent, weight=config["smooth_ratio"])
        data['Error_Percent'] = err_percent
        data['Smoothed_Error_Percent'] = smoothed_err_percent

        data['Num_Episode'] = data['Step']/step_episode_ratio
        all_data = pd.concat([all_data, data], axis=0)

    plt.figure(figsize=(9, 7)) 

    if config['show_error_percent']:
        if config['show_smoothed']:
            ax = sns.lineplot(data=all_data, x='Num_Episode', y='Smoothed_Error_Percent', color='#0000FF')
        else:
            ax = sns.lineplot(data=all_data, x='Num_Episode', y='Error_Percent', color='#0000FF')
    else:
        if config['show_smoothed']:
            ax = sns.lineplot(data=all_data, x='Num_Episode', y='Smoothed_Value', color='#0000FF')
        else:
            ax = sns.lineplot(data=all_data, x='Num_Episode', y='Value', color='#0000FF')
    ax.tick_params(axis="y", direction='in', labelsize=25)
    ax.tick_params(axis="x", direction='in', labelsize=25)
    ax.set_xlabel('Number of Episodes', fontsize=25)

    ylabel = 'Error (%) to Reference Cost' if config['show_error_percent'] else 'Evaluation Cost'
    ax.set_ylabel(ylabel, fontsize=25)
    ax.grid(True, alpha=0.4)
    # Create the line plot
    plt.savefig('tests/test_embed_lqr_rl/paperplots/{}.png'.format(config['fn']), bbox_inches='tight')

def plot_latent_tSNE(config):
    full_filepath = os.path.join(config['src_dir'], config['filepath'])
    with open(full_filepath, 'rb') as f:
        data = pickle.load(f)
    
    print("number of transitions: {}".format(len(data)))
    # Latent state predictions
    X = []
    y = []
    for i, it in enumerate(data):
        if i % 1 == 0:
            X.append(it['g_next'])
            X.append(it['g_pred'])
            y.append('z_next')
            y.append('z_pred')
    
    latent_state = np.concatenate(X, axis=0)  # Your high-dimensional latent state predictions
    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=40, random_state=config['seed'])  # Set the desired number of dimensions
    latent_tsne = tsne.fit_transform(latent_state)
    tsne_result_df = pd.DataFrame({'tsne_1': latent_tsne[:,0], 'tsne_2': latent_tsne[:,1], 'label': y})
    plt.figure(figsize=(10, 8)) 
    fig, ax = plt.subplots(1)

    blue = "#0000FF"  # RGB value for navy blue
    red = "#FF0000"  # Hexadecimal value for maroon
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', style='label', data=tsne_result_df, ax=ax, s=30, palette=[blue, red],markers=["s", "^"])
    ax.tick_params(axis="y", direction='in', labelsize=20)
    ax.tick_params(axis="x", direction='in', labelsize=20)
    ax.set_xlabel('t-SNE Dim 1', fontsize=20)
    ax.set_ylabel('t-SNE Dim 2', fontsize=20)
    ax.legend(fontsize=20) 
    plt.savefig('./paperplots/{}.png'.format(config['fn']), bbox_inches='tight')

def plot_pole_zero(config):
    src_dir = config['src_dir']
    step = config['step']
    fn = config['fn']
    agent = build_agent(config['exp_config'])
    agent.load(src_dir, step)
    A = agent.actor.trunk._g_affine.detach().cpu().numpy()
    B = agent.actor.trunk._u_affine.detach().cpu().numpy()
    print(A.shape[0])
    print(B.shape)

    # Get the poles and zeros of the latent system
    sys = control.StateSpace(A, B, np.eye(50), np.zeros((50,1)), dt=0.1)
    poles = sys.poles()
    print("poles:", poles)
    zeros = sys.zeros()
    print("zeros:", zeros)
    

    # Plot pole-zero for true 4D cart-pole system 
    m = 0.1
    g = 9.81
    M = 1.0
    l = 1.0
    # Define the state-space representation of the cart-pole system
    cp_A = [[0, 1, 0, 0],
        [0, 0, -m*g/M, 0],
        [0, 0, 0, 1],
        [0, 0, (M+m)*g/(M*l), 0]]
    cp_B = [[0],
        [1/M],
        [0],
        [-1/(M*l)]]
    cp_C = [[1, 0, 0, 0],
        [0, 0, 1, 0]]
    cp_D = [[0],
        [0]]

    # Create the state-space model
    cp_sys = control.StateSpace(cp_A, cp_B, cp_C, cp_D)

    # Compute the zeros and poles
    true_zeros = control.zero(cp_sys)
    true_poles = control.pole(cp_sys)
    # Print the results
    print("Zeros:", true_zeros)
    print("Poles:", true_poles)

    # Plot the pole-zero plot
    fig, ax = plt.subplots(1)
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='#FF0000', label='Latent System Poles',s=60)
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='#FF0000', label='Latent System Zeros',s=60)

    plt.scatter(np.real(true_poles), np.imag(true_poles), marker='x', color='#0000FF', label='True System Poles', s=40)
    plt.scatter(np.real(true_zeros), np.imag(true_zeros), marker='o', color='#0000FF', label='True System Zeros', s=40)


    plt.xlabel('Real', fontsize=20)
    plt.ylabel('Imaginary', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

    # Specify the labels and corresponding handles for the legend
    legend_labels = ['Latent System Poles', 'True System Poles']
    legend_handles = [plt.scatter([], [], label='Latent System Poles', marker='x', color='#FF0000', s=60),
                      plt.scatter([], [], label='True System Poles', marker='x', color='#0000FF', s=40)]
    plt.legend(fontsize=16, labels=legend_labels, handles=legend_handles)

    plt.tight_layout()
    plt.savefig('./paperplots/{}_latent_and_true_pole_zero.png'.format(config['fn']))

    

    # # Plot the pole-zero plot
    # fig, ax = plt.subplots(1)
    # plt.scatter(np.real(poles), np.imag(poles), marker='x', color='#FF0000', label='Poles', s=50)
    # plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='#0000FF', label='Zeros', s=50)
    # plt.xlabel('Real',fontsize=20)
    # plt.ylabel('Imaginary', fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # # Set tick direction
    # plt.tick_params(axis='x', direction='in')
    # plt.tick_params(axis='y', direction='in')
    # plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    # plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    # plt.savefig('./paperplots/{}_true_pole_zero.png'.format(config['fn']), bbox_inches='tight')


def plot_ctrb_obsv_matrix(config):
    src_dir = config['src_dir']
    step = config['step']
    fn = config['fn']
    agent = build_agent(config['exp_config'])
    agent.load(src_dir, step)
    A = agent.actor.trunk._g_affine.detach().cpu().numpy()
    B = agent.actor.trunk._u_affine.detach().cpu().numpy()
    print(A.shape[0])
    print(B.shape)
    C = np.eye(50)
    D = np.zeros((50,1))

    sys = control.StateSpace(A, B, C, D, dt=0.1)

    # Compute the controllabilitymatrix
    ctrb = control.ctrb(sys.A, sys.B)
    print("Controllability Matrix: {}".format(ctrb))
    print("rank of controllability matrix: {}".format(np.linalg.matrix_rank(ctrb)))

    # Compute the observability matrix
    obsv = control.obsv(sys.A, sys.C)
    print("Observability Matrix: {}".format(obsv))
    print("rank of Observability matrix: {}".format(np.linalg.matrix_rank(obsv)))

def plot_QR(config):
    src_dir = config['src_dir']
    step = config['step']
    fn = config['fn']
    agent = build_agent(config['exp_config'])
    agent.load(src_dir, step)
    q_diag_log = agent.actor.trunk._q_diag_log.detach().cpu().numpy()
    Q = np.diag(np.exp(q_diag_log))
    print(Q.shape)
    Q_min = np.min(Q)
    print(np.min(Q))
    Q_max = np.max(Q)
    print(np.max(Q))

    print(np.diagonal(Q))


    # Plot the heatmap
    Q[Q==0] = np.nan

    if config['use_log']:
        Q = np.log10(Q)
    heatmap = plt.imshow(Q, cmap="Purples")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20) 
    plt.tick_params(axis='x', direction='in')
    plt.tick_params(axis='y', direction='in')

    # Add colorbar
    from matplotlib.ticker import FuncFormatter
    cbar = plt.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=20)
    def custom_formatter(x, pos):
        # Format the tick value as desired
        return f'{(10**x)}'

    # Set the tick label formatter
    if config['use_log']:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    # Show the plot
    plt.savefig('./paperplots/{}.png'.format(config['fn']))



def plot_state_curves(config):
    fullpath = os.path.join(config['src_dir'], 'eval_transitions', "{}.pkl".format(config['step']))
    with open(fullpath, 'rb') as f:
        trans = pickle.load(f)
    obs = trans[0]['obs']
    data = {'Steps':[], 'cart_pose':[], 'pole_angle':[], 'cart_vel':[], 'pole_angvel':[]}
    for i, it in enumerate(obs):
        data['Steps'].append(i)
        data['cart_pose'].append(it[0])
        data['pole_angle'].append(np.arccos(it[1]))
        data['cart_vel'].append(it[3])
        data['pole_angvel'].append(it[4])
        if i > 999:
            break
    
    plt.figure(figsize=(9, 6)) 
    df = pd.DataFrame(data)
    
    

    ax = sns.lineplot(data=df[['cart_pose', 'pole_angle', 'cart_vel', 'pole_angvel']], palette=["#FF0000", '#0000FF', "#33A02C", "#FFC700"], linewidth=2)
    ax.tick_params(axis="y", direction='in', labelsize=20)
    ax.tick_params(axis="x", direction='in', labelsize=20)
    ax.set_xlabel('Number of Steps', fontsize=20)
    ax.set_ylabel("Cart-pole Physical States", fontsize=20)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=20)  # Adjust the font size as needed
    plt.savefig('tests/test_embed_lqr_rl/paperplots/paperplots/{}.png'.format(config['fn']))
    


if __name__ == "__main__":
    ##### 1. for showing cartpole low and hi dim control perf  #####
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-05-14-im84-b128-s2023-fc-1684103194',
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-05-15-im84-b128-s2023-fc-1684117863'
    # plot_config = {'src_dir':["/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-06-05-im84-b128-s2021-fc-1686009235",
    #                           "/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-06-05-im84-b128-s2022-fc-1686000487",
    #                           "/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-06-05-im84-b128-s2024-fc-1685949032"
    #                           ],
    #                         'step_episode_ratio': 250,
    #                         'optimal_cost':-848,
    #                         'show_error_percent':True,
    #                         "show_smoothed":True,
    #                         'fn':"result_01_lim_7",
    #                         'meta':"lo-dim",
    #                         'num_lines':7,
    #                         'smooth_ratio':0.3}
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cheetah/cheetah-run-05-27-im84-b128-s2023-fc-1685153234',
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cheetah/cheetah-run-05-27-im84-b512-s2023-fc-1685164913'
    # plot_config = {'src_dir':["/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cheetah/cheetah-run-05-27-im84-b128-s2023-fc-1685153234",
    #                           "/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cheetah/cheetah-run-05-27-im84-b512-s2023-fc-1685164913",
    #                           "/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cheetah/cheetah-run-06-05-im84-b512-s2024-fc-1685948930",
    #                           "/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cheetah/cheetah-run-06-05-im84-b512-s2025-fc-1685995729"],
    #                         'step_episode_ratio': 250,
    #                         'optimal_cost':-518,
    #                         'show_error_percent':True,
    #                         "show_smoothed":True,
    #                         'fn':"result_cheetah",
    #                         'meta':"hi-dim",
    #                         'num_lines':'all',
    #                         'smooth_ratio':0.6}
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-04-20-im84-b128-s2023-pixel-1682025432',
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-04-21-im84-b128-s2023-pixel-1682118147',
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-04-22-im84-b128-s2023-pixel-1682202227',
    # '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-04-23-im84-b128-s2023-pixel-1682285066'
    plot_config = {'src_dir':['/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-04-20-im84-b128-s2023-pixel-1682025432',
                               '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-05-24-im84-b128-s2023-pixel-1684907295',
                               '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-06-05-im84-b128-s2024-pixel-1685993119'],
                             'step_episode_ratio': 125,
                             'optimal_cost':-848,
                             'show_error_percent':True,
                             "show_smoothed":True,
                             'fn':"result_temp_delete_anytime",
                             'meta':"hi-dim",
                             'num_lines':"all",
                             'smooth_ratio':0.6}

    plot_train_curve(src_dir=plot_config['src_dir'], 
                      step_episode_ratio=plot_config['step_episode_ratio'], 
                      optimal_cost=plot_config['optimal_cost'],
                      config={"show_error_percent":plot_config['show_error_percent'], 
                              "show_smoothed":plot_config['show_smoothed'], 
                              'fn':plot_config['fn'],
                              'num_lines': plot_config['num_lines'],
                              'smooth_ratio':plot_config['smooth_ratio']})  # for cart-pole, we use optimal cost of -848 from CURL paper

    ##### 2. for showing latent prediction of Koopman Operator  #####
    # plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-05-15-im84-b128-s2023-fc-1684117863/action_repeat=1',
    #                'filepath': 'lat/200000.pkl',
    #                'seed':2026,
    #                'fn': "result_03_2026",
    #                'meta': "latent prediction"}
    
    # plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-06-05-im84-b128-s2024-fc-1685995477',
    #                'filepath': 'lat/100000.pkl',
    #                'seed':2026,
    #                'fn': "result_03_2026_ar4",
    #                'meta': "latent prediction"}  # this is for testing if action_repeat=1 or 4 affects the prediction, no it doesn't
    # plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cheetah/cheetah-run-05-27-im84-b512-s2023-fc-1685164913',
    #                'filepath': 'lat/3950000.pkl',
    #                'seed':2026,
    #                'fn': "result_03_cheetah_2026_3950000",
    #                'meta': "latent prediction cheetah"}
    # plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-06-05-im84-b128-s2024-pixel-1685993119',
    #                'filepath': 'lat/100000.pkl',
    #                'seed':2013,
    #                'fn': "result_03_cartpole2_pix_2013_100000",
    #                'meta': "latent prediction cartpole pixel"}

    #plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cheetah/cheetah-run-06-05-im84-b512-s2025-fc-1685995729',
    #               'filepath': 'lat/950000.pkl',
    #               'seed':2030,
    #               'fn': "result_03_cheetah_2030_950000",
    #               'meta': "latent prediction cheetah"}
    #plot_latent_tSNE(plot_config)



    ##### 3. for showing the analysis of stability, controllability and observability of learned latent Koopman Operator #####
    # plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-05-14-im84-b128-s2023-fc-1684103194/model',  
    #                'step': 150000,
    #                'exp_config': '/kpmlilat/tests/test_embed_lqr_rl/config/cartpole-swingup-embedlqr-state',
    #                'fn': 'result_04'}
    # plot_pole_zero(plot_config)
    # plot_ctrb_obsv_matrix(plot_config)


    ##### 4. showing Q, R and their sparsity, as well as relations to important states in original system
    # plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-05-14-im84-b128-s2023-fc-1684103194/model',  
    #                'step': 150000,
    #                'exp_config': '/kpmlilat/tests/test_embed_lqr_rl/config/cartpole-swingup-embedlqr-state',
    #                'fn': 'result_05_Purples',
    #                'use_log':False}
    
    # plot_config = {'src_dir': '/kpmlilat/tests/test_embed_lqr_rl/tmp_paper/cartpole/cartpole-swingup-06-05-im84-b128-s2024-pixel-1685993119/model',  
    #                'step': 300000,
    #                'exp_config': '/kpmlilat/tests/test_embed_lqr_rl/config/cartpole-swingup-embedlqr',
    #                'fn': 'result_05_cartpole_pix_Purples',
    #                'use_log':True}
    # plot_QR(plot_config)


    ##### 5. visualize mapping
    # plot_Q_mapping()


    ##### 6. plot state curves, specially for 4D cart-pole
    # plot_config = {"src_dir": '/kpmlilat/tests/test_embed_lqr_rl/tmp/cartpole/cartpole-swingup-05-20-im84-b128-s2023-fc-1684625834',
    #                'step': 450000,
    #                'fn': 'result_state_curve'}
    # plot_state_curves(plot_config)


    pass