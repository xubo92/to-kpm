"""
# This file mainly tests the goal-driven LQR control on Koopman-based latent state and dynamics
# Author: XL
# Date: 2023.2.6
# Instructions: 
#   = Envs: InvertedPendulum, Ant, HalfCheetah: lo-dim state -> hi-dim Koopman state. Pixel-Wise control scenarios: hi-dim -> lo-dim
#   = Modes: 1. Freeze A, B, encoder, only update Q via RL. Motivation: even A, B and encoder are well-trained, misinitialization of Q is not good.
#               This is useful for lo-to-hi dim situation where model is easy to learn but Q is not easy to tune.
#            2. Update A, B, encoder and Q all together. Motivation: Encoder and model are intrinsically difficult to train. This is useful for hi-to-lo 
#               dim situation where pixel-based dynamics is not easy to learn, but people don't have access to physical states.
#   = Advantages (analysis): think and discuss
#   
"""
import sys
import torch
import numpy as np
import gym

sys.path.append("/root")
from koopman_policy.utils import KoopmanFCNNLift

def test_example(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = gym.make(config.env)
    env.seed(config.seed)
    env.observation_space.seed(config.seed)
    env.action_space.seed(config.seed)

    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]
    z_dim = config.z_dim

    # Collect pre-training (supervised) data
    n_trajs = 1000
    n_steps = 200

    trajs = np.random.rand(n_trajs, n_steps, x_dim) * 2 -1 
    u     = np.random.rand(n_trajs, n_steps, u_dim) * 2 -1

    for i in range(n_trajs):
        trajs[i, 0, :] = env.reset()
        for j in range(n_steps-1):
            u[i, j] = env.action_space.sample()
            x_new, r, done, info = env.step(u[i, j])
            trajs[i, j+1, :] = x_new
    
    # Statistics of data
    mean = np.mean(trajs, axis=(0, 1))
    std = np.std(trajs, axis=(0, 1))

    max = np.float32(np.max(trajs, axis=(0, 1)))
    min = np.float32(np.min(trajs, axis=(0, 1)))
    print('max, min:', max, min)
    print('mean, std;', mean, std)


    # Set up encoder
    phi_fcnn_basis = KoopmanFCNNLift(in_dim=x_dim, out_dim=n_k, hidden_dim=[4, 4])
    ctrl = kpm.KoopmanLQR(k=n_k, x_dim=x_dim, u_dim=u_dim, x_goal=torch.zeros(5).float(), T=5, phi=phi_fcnn_basis, u_affine=None)
    
    # ctrl.cuda()
    ctrl.fit_koopman(torch.from_numpy(trajs).float(), torch.from_numpy(u).float(), 
        train_phi=True, 
        train_phi_inv=False,
        train_metric=False,
        ls_factor=1,
        recurr = 1,
        n_itrs=10, 
        lr=2.5e-3, 
        verbose=True)

def test_pendulum():
    torch.manual_seed(0)
    np.random.seed(0)

    env = gym.make('InvertedPendulumBulletEnv-v0')

    env.seed(0)
    env.observation_space.seed(0)
    env.action_space.seed(0)

    #collect data
    n_trajs = 1000
    n_steps = 200

    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]
    z_dim = config.z_dim

    trajs = np.random.rand(n_trajs, n_steps, x_dim) * 2 -1 
    u = np.random.rand(n_trajs, n_steps, u_dim) * 2 -1 
    for i in range(n_trajs):
        trajs[i, 0, :] = env.reset()
        for j in range(n_steps-1):
            u[i, j] = env.action_space.sample()
            x_new, r, done, info = env.step(u[i, j])
            trajs[i, j+1, :] = x_new
    
    #statistics of data
    mean = np.mean(trajs, axis=(0, 1))
    std = np.std(trajs, axis=(0, 1))
    #whiten data
    # trajs_normed = (np.array(trajs) - mean[None, None, :])/std[None, None, :]
    max = np.float32(np.max(trajs, axis=(0, 1)))
    min = np.float32(np.min(trajs, axis=(0, 1)))
    print('max, min:', max, min)
    print('mean, std;', mean, std)

    phi_fixed_basis = KoopmanThinPlateSplineBasis(in_dim=x_dim, n_basis=n_k-x_dim, center_dist_box_scale=max-min)
    phi_fcnn_basis = KoopmanFCNNLift(in_dim=x_dim, out_dim=n_k, hidden_dim=[4, 4])
    ctrl = kpm.KoopmanLQR(k=n_k, x_dim=x_dim, u_dim=u_dim, x_goal=torch.zeros(5).float(), T=5, phi=phi_fcnn_basis, u_affine=None)
    
    # ctrl.cuda()
    ctrl.fit_koopman(torch.from_numpy(trajs).float(), torch.from_numpy(u).float(), 
        train_phi=True, 
        train_phi_inv=False,
        train_metric=False,
        ls_factor=1,
        recurr = 1,
        n_itrs=10, 
        lr=2.5e-3, 
        verbose=True)
    # ctrl.cpu()

    #for prediction
    # test_step = 300

    # x_0 = env.reset()
    # u = np.array([(-1) ** (np.rint(np.arange(test_step) / 30))]).T[np.newaxis, :, :]
    
    # test_traj = [x_0]
    # pred_traj = [ctrl._phi(torch.from_numpy(x_0).float().unsqueeze(0)).detach().numpy()]    #unsqueeze for the batch dimension
    # for t in range(test_step-1):
    #     x_new, r, done, info = env.step(u[:, t, :])
        
    #     x_pred = ctrl.predict_koopman(torch.from_numpy(pred_traj[-1]).float(), torch.from_numpy(u[:, t, :]).float())
    #     test_traj.append(x_new)
    #     pred_traj.append(x_pred.detach().numpy())

    # test_traj = ctrl._phi(torch.from_numpy(np.array(test_traj)[None, ...]).float()).detach().numpy()
    # pred_traj = np.swapaxes(np.array(pred_traj), 0, 1)
    # pred_traj[:, :, :5]=pred_traj[:, :, :5]*std+mean

    # fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(121)
    # ax.plot(np.arange(test_traj.shape[1]), u[0, :, 0])
    # ax = fig.add_subplot(122)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 0] , '.b', markersize=1)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 1] , '.g', markersize=1)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 2] , '.r', markersize=1)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 4] , '.y', markersize=1)
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 0], 'b-')
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 1], 'g-')
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 2], 'r-')
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 4], 'y-')
    # plt.show()



    #for test
    x_0 = env.reset()
    test_traj = [x_0]
    tol_r = 0

    Q = np.ones(n_k) * 1e-3
    #focusing on theta dimension 
    # Q[0] = 0.01
    Q[1] = 0.5
    Q[2] = 1.
    Q[3] = 1.
    # Q[4] = 1e-6

    R = np.ones(u_dim)*1e-4
    #state format is np.array([x, vx, np.cos(self.theta), np.sin(self.theta), theta_dot])
    ctrl._x_goal = nn.Parameter(torch.from_numpy(np.concatenate((x_0[:2], np.array([1, 0, 0])))).float().unsqueeze(0))
    ctrl._q_diag_log = nn.Parameter(torch.from_numpy(Q).float().log())
    ctrl._r_diag_log = nn.Parameter(torch.from_numpy(R).float().log())
    u_lst = []

    frames = []
    for t in range(300):
        u_klqr = ctrl(torch.from_numpy(test_traj[-1]).float().unsqueeze(0)).detach().numpy()[0]
        u_lst.append(u_klqr)
        x_new, r, done, info = env.step(u_klqr) 
        fr = env.render("rgb_array")  # XL
        frames.append(fr)
        test_traj.append(x_new)
        tol_r += r
        # if done:
        #     print('Terminated at Step {0}'.format(t))
        #     break
    
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile("test_koopman_lqr3.mp4")

    test_traj = np.array(test_traj)
    u_lst = np.array(u_lst)

    print(tol_r)    
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    ax.plot(np.arange(test_traj.shape[0]-1), u_lst[:, 0])
    ax = fig.add_subplot(122)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 0], 'b', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 1], 'g', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 2], 'r', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 3], 'c', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 4], 'y', markersize=1)
    plt.show()

    return

if __name__ == "__main__":
    # test_solve_lqr()
    # test_mpc()    
    #test_cost_to_go()
    test_pendulum()
