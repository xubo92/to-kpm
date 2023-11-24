#plot imgs for rebuttal 
import numpy as np
import json
import matplotlib.pyplot as plt
import ast
import pandas as pd
import glob
import os, sys
import seaborn as sns

#1. yours vs AE
#2. yours vs planet vs dreamer

def smooth(scalars, weight): # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def compare_mbrl(src_dir, save_name):
    all_data = pd.DataFrame()
    json_files = glob.glob(src_dir + '/*.json')
    for json_f in json_files:
        data = pd.read_json(json_f)
        data["y"] = -data["y"]
        data["sm_y"] = smooth(scalars=data["y"], weight=0.6)
        all_data = pd.concat([all_data, data], axis=0)

    plt.figure(figsize=(9, 7))
    ax = sns.lineplot(data=all_data, x='x', y='sm_y', hue='method', hue_order=["ours", "dreamer", "planet"])
    ax.tick_params(axis="y", direction='in', labelsize=25)
    ax.tick_params(axis="x", direction='in', labelsize=25)
    ax.set_xlabel('Number of Episodes', fontsize=25)
    ylabel = 'Evaluation Cost'
    ax.set_ylabel(ylabel, fontsize=25)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=20) 
    ax.set_xlim(0, 1000)
    # Create the line plot
    plt.savefig(src_dir + '/{}.png'.format(save_name), bbox_inches='tight')


def compare_ae(src_dir, save_name, max_num_ep):
    all_data = pd.DataFrame()
    json_files = glob.glob(src_dir + '/*.json')
    for json_f in json_files:
        data = pd.read_json(json_f)
        data["y"] = -data["y"]
        data["sm_y"] = smooth(scalars=data["y"], weight=0.6)
        all_data = pd.concat([all_data, data], axis=0)

    plt.figure(figsize=(9, 7))
    ax = sns.lineplot(data=all_data, x='x', y='sm_y', hue='method', hue_order=["ours", "AE_recon", "AE_recon_pred"])
    ax.tick_params(axis="y", direction='in', labelsize=25)
    ax.tick_params(axis="x", direction='in', labelsize=25)
    ax.set_xlabel('Number of Episodes', fontsize=25)
    ylabel = 'Evaluation Cost'
    ax.set_ylabel(ylabel, fontsize=25)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=20) 
    ax.set_xlim(0, max_num_ep)
    # Create the line plot
    plt.savefig(src_dir + '/{}.png'.format(save_name), bbox_inches='tight')


def ablation_params():
    pass



if __name__ == "__main__":
    src_dir = "/kpmlilat/tests/test_embed_lqr_rl/rebuttalplots/comp_mbrl/cartpole_pixel"
    compare_mbrl(src_dir, save_name="comp_mbrl_sm06_cartpole_pixel")
    src_dir = "/kpmlilat/tests/test_embed_lqr_rl/rebuttalplots/comp_mbrl/cartpole_fc"
    compare_mbrl(src_dir, save_name="comp_mbrl_sm06_cartpole_fc")
    src_dir = "/kpmlilat/tests/test_embed_lqr_rl/rebuttalplots/comp_mbrl/cheetah"
    compare_mbrl(src_dir, save_name="comp_mbrl_sm06_cheetah")

    # src_dir = "/kpmlilat/tests/test_embed_lqr_rl/rebuttalplots/comp_ae/cartpole_fc"
    # compare_ae(src_dir, save_name="comp_ae_sm06_cartpole_fc_new", max_num_ep=1000)
    # src_dir = "/kpmlilat/tests/test_embed_lqr_rl/rebuttalplots/comp_ae/cartpole_pixel"
    # compare_ae(src_dir, save_name="comp_ae_sm06_cartpole_pixel_new", max_num_ep=1000)
    # src_dir = "/kpmlilat/tests/test_embed_lqr_rl/rebuttalplots/comp_ae/cheetah"
    # compare_ae(src_dir, save_name="comp_ae_sm06_cheetah_new", max_num_ep=4000)
