"""
In this file, we hope to merge the implementation of CURL and EmbedLQR to achieve a new working scenario: EmbedLQR over Pixel-Based Control
# Author: XL
# Date: 2023.2.9
# Location: SFU MarsLab
"""
import os
import argparse
import yaml
import utils
import time
import torch
import dmc2gym
import json

from utils import *
import pybullet as p
import pybulletgym.envs
from gym.wrappers import NormalizeObservation





CONFIG_PATH = '/kpmlilat/tests/test_embed_lqr_rl/config'


def parse_args():
    parser = argparse.ArgumentParser(description='args', add_help=False)
    parser.add_argument('--config', type=str,
                        default='cartpole-swingup-embedlqr', help='Name of the config file')
    # parser.add_argument('--config', type=str,
    #                     default='cartpole-swingup', help='Name of the config file')
    args, unknown = parser.parse_known_args()
    return args




def main():
    args = parse_args()
    #load yaml configuration
    with open(os.path.join(CONFIG_PATH, args.config+'.yaml')) as file:
        config = yaml.safe_load(file)

    utils.set_seed_everywhere(config['seed'])

    # set environment
    if config.get('domain_name') is not None and config.get('task_name') is not None:
        env = dmc2gym.make(
            domain_name=config['domain_name'],
            task_name=config['task_name'],
            seed=config['seed'],
            visualize_reward=False,
            from_pixels=(config['env']['encoder_type'] == 'pixel'),
            height=config['env']['pre_transform_image_size'],
            width=config['env']['pre_transform_image_size'],
            frame_skip=config['env']['action_repeat'])
    elif config.get('env_name') is not None:
        env = NormalizeObservation(gym.make(config['env_name']))
        env._max_episode_steps = 1000
    env.seed(config['seed'])

    # stack several consecutive frames together
    if config['env']['encoder_type'] == 'pixel':
        env = utils.FrameStack(env, k=config['env']['frame_stack'])
    

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = config['domain_name'] + '-' + config['task_name'] if config.get('domain_name') is not None and config.get('task_name') is not None else config['env_name']
    exp_name = env_name + '-' + ts + '-im' + str(config['env']['image_size']) +'-b'  \
    + str(config['train']['batch_size']) + '-s' + str(config['seed'])  + '-' + config['env']['encoder_type'] + '-' + str(time.time()).split(".")[0]
    config['work_dir'] = config['work_dir'] + '/'  + exp_name

    utils.make_dir(config['work_dir'])
    video_dir = utils.make_dir(os.path.join(config['work_dir'], 'video'))
    model_dir = utils.make_dir(os.path.join(config['work_dir'], 'model'))
    buffer_dir = utils.make_dir(os.path.join(config['work_dir'], 'buffer'))

    video = VideoRecorder(video_dir if config['save_video'] else None)
    print("video is initialized ...")

    with open(os.path.join(config['work_dir'], 'args.json'), 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # replay buffer
    if config['env']['encoder_type'] == 'pixel':
        obs_shape = (3*config['env']['frame_stack'], config['env']['image_size'], config['env']['image_size'])
        pre_aug_obs_shape = (3*config['env']['frame_stack'],config['env']['pre_transform_image_size'],config['env']['pre_transform_image_size'])
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    action_shape = env.action_space.shape


    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=config['env']['replay_buffer_capacity'],
        batch_size=config['train']['batch_size'],
        device=device,
        image_size=config['env']['image_size'],
        from_pixel=(config['env']['encoder_type'] == 'pixel')
    )


    # make agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        config=config,
        device=device)
    
    
    # Logger
    L = Logger(config['work_dir'], use_tb=config['save_tb'])

    # training process
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(config['train']['num_train_steps']):
        # evaluate agent periodically
        if step % config['eval']['eval_freq'] == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, config['eval']['num_eval_episodes'], L, step, config)
            if config['save_model']:
                agent.save_curl(model_dir, step)
                agent.save(model_dir, step)
            if config['save_buffer']:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % config['log_interval'] == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % config['log_interval'] == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % config['log_interval'] == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < config['train']['init_steps']:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= config['train']['init_steps']:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

