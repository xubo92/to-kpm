import torch
import numpy as np 
import random 
from collections import deque
import gym

import imageio
import os

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import shutil
import torchvision
from termcolor import colored

import time
from skimage.util.shape import view_as_windows
from torch.utils.data import Dataset, DataLoader

import pickle
import ast

# Codebase is modifield based on https://github.com/MishaLaskin/curl
# https://github.com/navigator8972/koopman_policy


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
        
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop(imgs, output_size):
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_size = img_size - output_size
    imgs = np.transpose(imgs, (0,2,3,1))
    w1 = crop_size // 2
    h1 = crop_size // 2

    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

# XL: handle vectoried states, not pixels.
def random_noise(states, noise_level=0.1):
    n = states.shape[0]
    state_dim = states.shape[-1]
    abs_states = np.abs(states)
    noise = np.random.uniform(low=-noise_level*abs_states, high=noise_level*np.abs(states), size=(n, state_dim))
    noise_states = states + noise
    # print("adding noise to vectorized states ...")
    return noise_states


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None, from_pixel=True):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        # XL: use contrastive loss for vectorized states, not pixels
        self.from_pixel = from_pixel

        

    def add(self, obs, action, reward, next_obs, done):
       
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    # XL: only used by SAC_AE module
    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def sample_proprio(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = center_crop(obses, self.image_size) if self.from_pixel else obses
        next_obses = center_crop(next_obses, self.image_size) if self.from_pixel else next_obses

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()
    
        # XL: handle pixel and vector states
        obses = random_crop(obses, self.image_size) if self.from_pixel else random_noise(obses)  
        next_obses = random_crop(next_obses, self.image_size) if self.from_pixel else random_noise(next_obses)
        pos = random_crop(pos, self.image_size) if self.from_pixel else random_noise(pos)
    
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 
    
class ReplayBufferT3(ReplayBuffer):
    def __init__(self,obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None, from_pixel=True):
        super(ReplayBufferT3,self).__init__(obs_shape, action_shape, capacity, batch_size, device,image_size=image_size,transform=transform, from_pixel=from_pixel)

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.nexxt_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.nexxxt_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)

        
class ReplayBufferT5(ReplayBuffer):
    def __init__(self):  
        pass

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)



class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            try:
                frame = env.render(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id
                )
            except:
                frame = env.render(
                    mode='rgb_array',
                )
    
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)



def make_agent(obs_shape, action_shape, config, device):
    if config['agent']['name'] == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            config=config,
            hidden_dim=config['agent']['hidden_dim'],
            discount=config['agent']['discount'],
            init_temperature=config['agent']['init_temperature'],
            alpha_lr=config['agent']['alpha_lr'],
            alpha_beta=config['agent']['alpha_beta'],
            actor_lr=config['agent']['actor_lr'],
            actor_beta=config['agent']['actor_beta'],
            actor_log_std_min=config['agent']['actor_log_std_min'],
            actor_log_std_max=config['agent']['actor_log_std_max'],
            actor_update_freq=config['agent']['actor_update_freq'],
            critic_lr=config['agent']['critic_lr'],
            critic_beta=config['agent']['critic_beta'],
            critic_tau=config['agent']['critic_tau'],
            critic_target_update_freq=config['agent']['critic_target_update_freq'],
            encoder_type=config['env']['encoder_type'],
            encoder_feature_dim=config['agent']['encoder_feature_dim'],
            encoder_lr=config['agent']['encoder_lr'],
            encoder_tau=config['agent']['encoder_tau'],
            num_layers=config['agent']['num_layers'],
            num_filters=config['agent']['num_filters'],
            log_interval=config['log_interval'],
            detach_encoder=config['agent']['detach_encoder'],
            curl_latent_dim=config['agent']['curl_latent_dim']

        )
    elif config['agent']['name'] == 'curl_sac_koopmanlqr':
        return CurlSacKoopmanAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            config=config,
            hidden_dim=config['agent']['hidden_dim'],
            discount=config['agent']['discount'],
            init_temperature=config['agent']['init_temperature'],
            alpha_lr=config['agent']['alpha_lr'],
            alpha_beta=config['agent']['alpha_beta'],
            actor_lr=config['agent']['actor_lr'],
            actor_beta=config['agent']['actor_beta'],
            actor_log_std_min=config['agent']['actor_log_std_min'],
            actor_log_std_max=config['agent']['actor_log_std_max'],
            actor_update_freq=config['agent']['actor_update_freq'],
            critic_lr=config['agent']['critic_lr'],
            critic_beta=config['agent']['critic_beta'],
            critic_tau=config['agent']['critic_tau'],
            critic_target_update_freq=config['agent']['critic_target_update_freq'],
            encoder_type=config['env']['encoder_type'],
            encoder_feature_dim=config['agent']['encoder_feature_dim'],
            encoder_lr=config['agent']['encoder_lr'],
            encoder_tau=config['agent']['encoder_tau'],
            num_layers=config['agent']['num_layers'],
            num_filters=config['agent']['num_filters'],
            log_interval=config['log_interval'],
            detach_encoder=config['agent']['detach_encoder'],
            curl_latent_dim=config['agent']['curl_latent_dim']
        )
    else:
        assert 'agent is not supported: %s' % config['agent']['name']

def evaluate(env, agent, video, num_episodes, L, step, config, save_transitions=False):
    all_ep_rewards = []
    all_ep_costs   = []

    all_fit_losses = []
    all_transitions = [{"obs":[], "act":[], "rew":[], "done":[]}]*num_episodes
    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            episode_cost = 0  # XL: record the cost
            episode_fit_loss = 0 # XL: record the fit loss
            episode_lat_trans = []
            episode_timestep = 0
            if save_transitions:
                all_transitions[i]["obs"].append(obs)
                all_transitions[i]["done"].append(done)
            while not done:
                # center crop image
                if config['env']['encoder_type'] == 'pixel':
                    obs = utils.center_crop_image(obs,config['env']['image_size'])
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)

                # XL: latent state for curr obs
                with utils.eval_mode(agent):
                    with torch.no_grad():
                        th_obs = torch.FloatTensor(obs).to(config['device'])
                        th_obs = th_obs.unsqueeze(0)
                        th_act = torch.FloatTensor(action).to(config['device'])
                        g = agent.actor.encoder(th_obs)
                        g_pred = agent.actor.trunk._predict_koopman(g, th_act)

                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

                cost = -reward  # XL: record the cost
                episode_cost += cost
                episode_timestep += 1


                if save_transitions:
                    all_transitions[i]["obs"].append(obs)
                    all_transitions[i]["act"].append(action)
                    all_transitions[i]["rew"].append(reward)
                    all_transitions[i]["done"].append(done)
                # XL: latent state for next obs
                with utils.eval_mode(agent):
                    with torch.no_grad():
                        # center crop image
                        if config['env']['encoder_type'] == 'pixel':
                            obs = utils.center_crop_image(obs,config['env']['image_size'])
                        th_obs = torch.FloatTensor(obs).to(config['device'])
                        th_obs = th_obs.unsqueeze(0)
                        g_next = agent.actor.encoder(th_obs)
                        loss_fn = nn.MSELoss()
                        fit_loss = loss_fn(g_pred, g_next)

                episode_fit_loss += (fit_loss.detach().cpu().numpy())
                episode_lat_trans.append({'g':g.detach().cpu().numpy(), 
                                          'g_next':g_next.detach().cpu().numpy(), 
                                          'g_pred':g_pred.detach().cpu().numpy()})

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
            
            # XL: record the cost
            L.log('eval/' + prefix + 'episode_cost', episode_cost, step)
            all_ep_costs.append(episode_cost) 

            # XL: record mean fitting loss
            L.log('eval/' + prefix + 'episode_fit_loss', episode_fit_loss, step)
            all_fit_losses.append(episode_fit_loss/episode_timestep)

            # XL: record latent transition info
            path = '/'.join(video.dir_name.split('/')[0:-1] + ['lat'] + ['{}.pkl'.format(step)])
            folder_path = os.path.dirname(path)
            os.makedirs(folder_path, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(episode_lat_trans, f)

        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        # XL: record the cost
        mean_ep_cost = np.mean(all_ep_costs)
        best_ep_cost = np.min(all_ep_costs)
        L.log('eval/' + prefix + 'mean_episode_cost', mean_ep_cost, step)
        L.log('eval/' + prefix + 'best_episode_cost', best_ep_cost, step)

        # XL: record the fit loss
        mean_ep_fit_loss = np.mean(all_fit_losses)
        best_ep_fit_loss = np.min(all_fit_losses)
        L.log('eval/' + prefix + 'mean_episode_fit_loss', mean_ep_fit_loss, step)
        L.log('eval/' + prefix + 'best_episode_fit_loss', best_ep_fit_loss, step)

        if save_transitions:
            eval_path = '/'.join(video.dir_name.split('/')[0:-1] + ['eval_transitions'] + ['{}.pkl'.format(step)])
            eval_folder_path = os.path.dirname(eval_path)
            os.makedirs(eval_folder_path, exist_ok=True)
            with open(eval_path, 'wb') as f:
                pickle.dump(all_transitions, f)
                
    run_eval_loop(sample_stochastically=False)
    L.dump(step)


FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), ('step', 'S', 'int'),
            ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'),
            ('batch_reward', 'BR', 'float'), ('actor_loss', 'A_LOSS', 'float'),
            ('critic_loss', 'CR_LOSS', 'float'), ('curl_loss', 'CU_LOSS', 'float'), ('kpm_fitting_loss', 'KPMFIT_LOSS', 'float')
        ],
        'eval': [('step', 'S', 'int'), ('episode_reward', 'ER', 'float'), ('episode_cost', 'EC', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()

class Logger(object):

    def __init__(self, log_dir, use_tb=True, config='rl'):
        self._log_dir = log_dir
        if use_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[config]['eval']
        )

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_image(key, image, step)

    def log_video(self, key, frames, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')

import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,output_logits=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass

# XL: add a Fully Connected Encoder for vectorized states
class FCEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__() 
        
        assert len(obs_shape) == 1  # only a vector state
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_filters = num_filters
        self.num_layers = num_layers

        self.fcs = nn.ModuleList(
            [nn.Linear(obs_shape[0], num_filters)]
        )
        for i in range(num_layers - 1):
            self.fcs.append(nn.Linear(num_filters, num_filters))

        self.fc = nn.Linear(num_filters, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def forward_fc(self, obs):
        self.outputs['obs'] = obs

        fc = torch.relu(self.fcs[0](obs))
        self.outputs['fc1'] = fc

        for i in range(1, self.num_layers):
            fc = torch.relu(self.fcs[i](fc))
            self.outputs['fc%s' % (i + 1)] = fc

        h = fc.view(fc.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_fc(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_weights_from(self, source):
        """Tie fc layers"""
        # only tie fc layers
        for i in range(self.num_layers):
            tie_weights(src=source.fcs[i], trg=self.fcs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/fc%s' % (i + 1), self.fcs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)



_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder, 'fc': FCEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
# from encoder import make_encoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class CurlSacAgent(object):
    """CURL representation learning with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        config,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128
    ):
        self.config=config
        self.device = device
        self.hidden_dim = hidden_dim
        self.discount = discount
        self.init_temperature=init_temperature
        self.alpha_lr=alpha_lr
        self.alpha_beta=alpha_beta
        self.actor_log_std_min=actor_log_std_min
        self.actor_log_std_max=actor_log_std_max
        self.actor_update_freq = actor_update_freq
        self.critic_lr=critic_lr
        self.critic_beta=critic_beta
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.encoder_type = encoder_type
        self.encoder_featured_dim = encoder_feature_dim
        self.encoder_lr=encoder_lr
        self.encoder_tau = encoder_tau
        self.num_layers=num_layers
        self.num_filters=num_filters
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.detach_encoder = detach_encoder
        self.curl_latent_dim = curl_latent_dim
        
        
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type in ['pixel','fc']:
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                        self.curl_latent_dim, self.critic,self.critic_target, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type in ['pixel','fc']:
            self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)  # TODO: debug this actor, the policy_action has 1 more dimension
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):
        
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
        
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', loss, step)



    def update(self, replay_buffer, L, step):
        if self.encoder_type in ['pixel','fc']:
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if step % self.cpc_update_freq == 0 and self.encoder_type in ['pixel','fc']:
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
 


class KoopmanActor(nn.Module):
    """Koopman LQR actor."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,
        config
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.action_shape = action_shape
        self.config = config
        # XL: set up goal reference for different situations
        goal_meta = self.config['koopman']['koopman_goal_image_path']
        if isinstance(goal_meta, str) and goal_meta.endswith(".pkl"):
            with open(self.config['koopman']['koopman_goal_image_path'], "rb") as f:
                self.goal_obs = torch.from_numpy(pickle.load(f)).unsqueeze(0).to(torch.device(self.config['device']))
        elif isinstance(goal_meta, list):
            self.goal_obs = torch.from_numpy(np.array(self.config['koopman']['koopman_goal_image_path'], dtype=np.float32)).unsqueeze(0).to(torch.device(self.config['device']))
        else:
            self.goal_obs = None

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_init = torch.nn.Parameter(torch.Tensor([1.0]).log()) # XL: initialize a log_std

        # XL: Koopman control module as trunk
        self.trunk = KoopmanLQR(k=encoder_feature_dim, 
                                T=5,
                                g_dim=encoder_feature_dim,
                                u_dim=action_shape[0],
                                g_goal=None,
                                u_affine=None)

        self.outputs = dict()
        self.apply(weight_init)

        

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        # XL: encode the goal images to be used in self.trunk
        if self.goal_obs is None:
            self.trunk._g_goal = torch.zeros((1, obs.shape[1])).squeeze(0).to(torch.device(self.config['device']))
        else:
            goal_obs = self.encoder(self.goal_obs, detach=detach_encoder)
            self.trunk._g_goal = goal_obs.squeeze(0)

        # XL: do not chunk to 2 parts as LQR directly gives mu; use constant log_std
        broadcast_shape = list(obs.shape[:-1]) + [self.action_shape[0]]
        mu, log_std = self.trunk(obs).chunk(1, dim=-1)[0], \
                    self.log_std_init + torch.zeros(*broadcast_shape).to(torch.device(self.config['device']))

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        # L.log_param('train_actor/fc1', self.trunk[0], step)
        # L.log_param('train_actor/fc2', self.trunk[2], step)
        # L.log_param('train_actor/fc3', self.trunk[4], step)

    

class KoopmanCritic(Critic):
    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters):
        super(KoopmanCritic, self).__init__(obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters)



class KoopmanLQR(nn.Module):
    def __init__(self, k, T, g_dim, u_dim, g_goal=None, u_affine=None):
        """
        k:          rank of approximated koopman operator
        T:          length of horizon
        g_dim:      dimension of latent state
        u_dim:      dimension of control input
        g_goal:     None by default. If not, override the x_goal so it is not necessarily corresponding to a concrete goal state
                    might be useful for non regularization tasks.  
        u_affine:   should be a linear transform for an augmented observation phi(x, u) = phi(x) + nn.Linear(u)
        """
        super().__init__()
        self._k = k
        self._T = T
        self._g_dim = g_dim
        self._u_dim = u_dim
        self._g_goal = g_goal
        self._u_affine = u_affine
        
        # prepare linear system params
        self._g_affine = nn.Parameter(torch.empty((k, k)))
        
        if self._u_affine is None:
            self._u_affine = nn.Parameter(torch.empty((k, u_dim)))
        else:
            self._u_affine = nn.Parameter(self._u_affine)
        
        # try to avoid degenerated case, can it be fixed with initialization?
        torch.nn.init.normal_(self._g_affine, mean=0, std=1)
        torch.nn.init.normal_(self._u_affine, mean=0, std=1)

        # parameters of quadratic functions
        self._q_diag_log = nn.Parameter(torch.zeros(self._k))  # to use: Q = diag(_q_diag_log.exp())
        self._r_diag_log = nn.Parameter(torch.zeros(self._u_dim)) # gain of control penalty, in theory need to be parameterized...
        self._r_diag_log.requires_grad = False

        # zero tensor constant for k and v in the case of fixed origin
        # these will be automatically moved to gpu so no need to create and check in the forward process
        self.register_buffer('_zero_tensor_constant_k', torch.zeros((1, self._u_dim)))
        self.register_buffer('_zero_tensor_constant_v', torch.zeros((1, self._k)))

        # we may need to create a few cache for K, k, V and v because they are not dependent on x
        # unless we make g_goal depend on it. This allows to avoid repeatively calculate riccati recursion in eval mode
        self._riccati_solution_cache = None
        return

    def forward(self, g0):
        '''
        perform mpc with current parameters given the initial x0
        '''
        K, k, V, v = self._retrieve_riccati_solution()
        u = -self._batch_mv(K[0], g0) + k[0]  # apply the first control as mpc
        return u
    
    @staticmethod
    def _batch_mv(bmat, bvec):
        """
        Performs a batched matrix-vector product, with compatible but different batch shapes.

        This function takes as input `bmat`, containing :math:`n \times n` matrices, and
        `bvec`, containing length :math:`n` vectors.

        Both `bmat` and `bvec` may have any number of leading dimensions, which correspond
        to a batch shape. They are not necessarily assumed to have the same batch shape,
        just ones which can be broadcasted.
        """
        return torch.matmul(bmat, bvec.unsqueeze(-1)).squeeze(-1)
    
    def _retrieve_riccati_solution(self):
        if self.training or self._riccati_solution_cache is None:
            Q = torch.diag(self._q_diag_log.exp()).unsqueeze(0)
            R = torch.diag(self._r_diag_log.exp()).unsqueeze(0)

            # use g_goal
            if self._g_goal is not None:
                goals = torch.repeat_interleave(self._g_goal.unsqueeze(0).unsqueeze(0), repeats=self._T+1, dim=1)
            else:
                goals = None

            # solve the lqr problem via a differentiable process.
            K, k, V, v = self._solve_lqr(self._g_affine.unsqueeze(0), self._u_affine.unsqueeze(0), Q, R, goals)
            self._riccati_solution_cache = (
                [tmp.detach().clone() for tmp in K], 
                [tmp.detach().clone() for tmp in k], 
                [tmp.detach().clone() for tmp in V], 
                [tmp.detach().clone() for tmp in v])
                 
        else:
            K, k, V, v = self._riccati_solution_cache
        return K, k, V, v
    

    def _solve_lqr(self, A, B, Q, R, goals):
        # a differentiable process of solving LQR, 
        # time-invariant A, B, Q, R (with leading batch dimensions), but goals can be a batch of trajectories (batch_size, T+1, k)
        #       min \Sigma^{T} (x_t - goal[t])^T Q (x_t - goal[t]) + u_t^T R u_t
        # s.t.  x_{t+1} = A x_t + B u_t
        # return feedback gain and feedforward terms such that u = -K x + k

        T = self._T
        K = [None] * T
        k = [None] * T
        V = [None] * (T+1)
        v = [None] * (T+1)

        A_trans = A.transpose(-2,-1)
        B_trans = B.transpose(-2,-1)

        V[-1] = Q  # initialization for backpropagation
        if goals is not None:
            v[-1] = self._batch_mv(Q, goals[:, -1, :])
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of AX = B to avoid direct inverse, note it also returns LU
                # for new torch.linalg.solve, no LU is returned
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V[i+1]), B) + R, B_trans)
                K[i] = torch.matmul(torch.matmul(V_uu_inv_B_trans, V[i+1]), A)
                k[i] = self._batch_mv(V_uu_inv_B_trans, v[i+1])

                # riccati difference equation, A-BK
                A_BK = A - torch.matmul(B, K[i])
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q
                v[i] = self._batch_mv(A_BK.transpose(-2, -1), v[i+1]) + self._batch_mv(Q, goals[:, i, :])
        else:
            # None goals means a fixed regulation point at origin. ignore k and v for efficiency
            for i in reversed(range(T)):
                # using torch.solve(B, A) to obtain the solution of A X = B to avoid direct inverse, note it also returns LU
                V_uu_inv_B_trans = torch.linalg.solve(torch.matmul(torch.matmul(B_trans, V[i+1]), B) + R, B_trans)
                K[i] = torch.matmul(torch.matmul(V_uu_inv_B_trans, V[i+1]), A)
                
                A_BK = A - torch.matmul(B, K[i]) #riccati difference equation: A-BK
                V[i] = torch.matmul(torch.matmul(A_trans, V[i+1]), A_BK) + Q
            k[:] = self._zero_tensor_constant_k
            v[:] = self._zero_tensor_constant_v       

        # we might need to cat or 
        #  to return them as tensors but for mpc maybe only the first time step is useful...
        # note K is for negative feedback, namely u = -Kx+k
        return K, k, V, v

    def _predict_koopman(self, G, U):
        '''
        predict dynamics with current koopman parameters
        note both input and return are embeddings of the predicted state, we can recover that by using invertible net, e.g. normalizing-flow models
        but that would require a same dimensionality
        '''
        return torch.matmul(G, self._g_affine.transpose(0, 1))+torch.matmul(U, self._u_affine.transpose(0, 1))
    
   
    
class CurlSacKoopmanAgent(CurlSacAgent):
    def __init__(self, obs_shape, 
                 action_shape, 
                 device,
                 config,
                 hidden_dim=256,
                 discount=0.99,
                 init_temperature=0.01,
                 alpha_lr=1e-3,
                 alpha_beta=0.9,
                 actor_lr=1e-3,
                 actor_beta=0.9,
                 actor_log_std_min=-10,
                 actor_log_std_max=2,
                 actor_update_freq=2,
                 critic_lr=1e-3,critic_beta=0.9,
                 critic_tau=0.005,
                 critic_target_update_freq=2,
                 encoder_type='pixel',
                 encoder_feature_dim=50,
                 encoder_lr=1e-3,
                 encoder_tau=0.005,
                 num_layers=4,
                 num_filters=32,
                 cpc_update_freq=1,
                 log_interval=100,
                 detach_encoder=False,
                 curl_latent_dim=128):
        super(CurlSacKoopmanAgent, self).__init__(obs_shape, action_shape, device, config,
                                                  hidden_dim=hidden_dim,
                                                  discount=discount,
                                                  init_temperature=init_temperature,
                                                  alpha_lr=alpha_lr,
                                                  alpha_beta=alpha_beta,
                                                  actor_lr=actor_lr,
                                                  actor_beta=actor_beta,
                                                  actor_log_std_min=actor_log_std_min,
                                                  actor_log_std_max=actor_log_std_max,
                                                  actor_update_freq=actor_update_freq,
                                                  critic_lr=critic_lr,
                                                  critic_beta=critic_beta,
                                                  critic_tau=critic_tau,
                                                  critic_target_update_freq=critic_target_update_freq,
                                                  encoder_type=encoder_type,
                                                  encoder_feature_dim=encoder_feature_dim,
                                                  encoder_lr=encoder_lr,
                                                  encoder_tau=encoder_tau,
                                                  num_layers=num_layers,
                                                  num_filters=num_filters,
                                                  cpc_update_freq=cpc_update_freq,
                                                  log_interval=log_interval,
                                                  detach_encoder=detach_encoder,
                                                  curl_latent_dim=curl_latent_dim)

        print("encoder type:{}".format(encoder_type))

        self.koopman_update_freq = config['koopman']['koopman_update_freq']
        self.koopman_fit_coeff   = config['koopman']['koopman_fit_coeff']

        self.actor = KoopmanActor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, config
        ).to(device)

        self.critic = KoopmanCritic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = KoopmanCritic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # XL: additional optimizers
        self.koopman_optimizers = torch.optim.Adam(
            self.actor.trunk.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        if self.encoder_type in ['pixel','fc']:
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                        self.curl_latent_dim, self.critic,self.critic_target, output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()
    
    # TODO: check loss terms in koopmanlqr_sac_garage.py
    def update_actor_and_alpha(self, obs, next_obs, action, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # XL: [not useful] maybe add more loss terms for embedlqr
        # koopman_fit_loss = self.koopman_fit_loss(obs, next_obs, action, self.config['koopman']['least_square_fit_coeff'])
        # actor_loss += self.config['koopman']['koopman_fit_coeff'] * koopman_fit_loss
        # print("actor loss: {} || koopman fit loss: {}".format(actor_loss, koopman_fit_loss))

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step() 


    def update_kpm(self, obs, next_obs, action, L, step, use_ls=False):
        # XL: we only use fit_loss for now, 
        # we do not use recon_loss as (1) curl is kind of recon, (2) we don't have a decoder spec.
        # we do not use reg_loss as it is not very important.
        g = self.actor.encoder(obs)
        g_next = self.actor.encoder(next_obs)
        g_pred = self.actor.trunk._predict_koopman(g, action)
        loss_fn = nn.MSELoss()
        fit_loss = loss_fn(g_pred, g_next)   
        
        if step % self.log_interval == 0: 
            L.log('train/kpm_fitting_loss', fit_loss, step)
        
        # XL: [not useful] update critic's encoder (should we update actor's encoder? maybe not because actor's encoder is tied to critic's)
        # self.encoder_optimizers.zero_grad()
        # fit_loss.backward()
        # self.encoder_optimizer.zero_grad()

        # update self.actor.trunk's parameters (A, B, Q, R)
        self.koopman_optimizers.zero_grad()
        fit_loss.backward()
        self.koopman_optimizers.step()


    def update(self, replay_buffer, L, step):
        if self.encoder_type in ['pixel','fc']:
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, next_obs, action, L, step)  # XL: fit the form of new update_actor_and_alpha()

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if step % self.cpc_update_freq == 0 and self.encoder_type in ['pixel','fc']:
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos,cpc_kwargs, L, step)
        
        
        if step % self.koopman_update_freq == 0 and self.encoder_type in ['pixel','fc'] and self.koopman_fit_coeff > 0:
            self.update_kpm(obs, next_obs, action, L, step)

        
# ==================== From here we add AE (decoder) related code ==================================== #

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

class FCDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.feature_dim = feature_dim

        self.fc = nn.Linear(self.feature_dim, num_filters)

        self.fcs = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.fcs.append(
                nn.Linear(num_filters, num_filters)
            )
        self.fcs.append(nn.Linear(num_filters, obs_shape[0]))

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        for i in range(0, self.num_layers-1):
            h = torch.relu(self.fcs[i](h))
            self.outputs['fc%s'%(i+1)] = h
        
        obs = self.fcs[-1](h)
        self.outputs['obs'] = obs
        return obs
    
    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/fc%s' % (i + 1), self.fcs[i], step)
        L.log_param('train_decoder/fc', self.fc, step)

class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


_AVAILABLE_DECODERS = {'pixel': PixelDecoder, 'fc': FCDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )  


class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        config,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32
    ):
        self.config = config
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda

        self.encoder_type = encoder_type
        self.encoder_feature_dim = encoder_feature_dim
        self.encoder_lr = encoder_lr
        self.decoder_type = decoder_type
        self.decoder_lr = decoder_lr

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, target_obs, L, step):
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
    

class KoopmanSacAeAgent(SacAeAgent):
    def __init__(self, obs_shape,
        action_shape,
        device,
        config,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32):
        super(KoopmanSacAeAgent, self).__init__(obs_shape, action_shape, device, config,
                                                hidden_dim=hidden_dim,
                                                discount=discount,
                                                init_temperature=init_temperature,
                                                alpha_lr=alpha_lr,
                                                alpha_beta=alpha_beta,
                                                actor_lr=actor_lr,
                                                actor_beta=actor_beta,
                                                actor_log_std_min=actor_log_std_min,
                                                actor_log_std_max=actor_log_std_max,
                                                actor_update_freq=actor_update_freq,
                                                critic_lr=critic_lr,
                                                critic_beta=critic_beta,
                                                critic_tau=critic_tau,
                                                critic_target_update_freq=critic_target_update_freq,
                                                encoder_type=encoder_type,
                                                encoder_feature_dim=encoder_feature_dim,
                                                encoder_lr=encoder_lr,
                                                encoder_tau=encoder_tau,
                                                decoder_type=decoder_type,
                                                decoder_lr=decoder_lr,
                                                decoder_update_freq=decoder_update_freq,
                                                decoder_latent_lambda=decoder_latent_lambda,
                                                decoder_weight_lambda=decoder_weight_lambda,
                                                num_layers=num_layers,
                                                num_filters=num_filters)

        print("encoder type:{}".format(encoder_type))

        self.koopman_update_freq = config['koopman']['koopman_update_freq']
        self.koopman_fit_coeff   = config['koopman']['koopman_fit_coeff']

        self.actor = KoopmanActor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, config).to(device)
        
        self.critic = KoopmanCritic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters).to(device)
        
        self.critic_target = KoopmanCritic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters).to(device)
        
    
        # XL: additional optimizers
        self.koopman_optimizers = torch.optim.Adam(
            self.actor.trunk.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()


    def update_kpm(self, obs, next_obs, action, L, step, use_ls=False):
        g = self.actor.encoder(obs)
        g_next = self.actor.encoder(next_obs)
        g_pred = self.actor.trunk._predict_koopman(g, action)
        loss_fn = nn.MSELoss()
        fit_loss = loss_fn(g_pred, g_next)   
        
        if step % 100 == 0: 
            L.log('train/kpm_fitting_loss', fit_loss, step)

        # update self.actor.trunk's parameters (A, B, Q, R)
        self.koopman_optimizers.zero_grad()
        fit_loss.backward()
        self.koopman_optimizers.step()

    # XL: a new update decoder function with mlp prediction loss
    def update_decoder_tot(self, obs, target_obs, nx_obs, target_nx_obs, action, L, step):
        h = self.critic.encoder(obs)
        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        if target_nx_obs.dim() == 4:
            target_nx_obs = utils.preprocess_obs(target_nx_obs)
        nx_h = self.actor.trunk._predict_koopman(h, action)
        nx_rec_obs = self.decoder(nx_h)
        pred_loss = F.mse_loss(target_nx_obs, nx_rec_obs)

        loss = rec_loss + self.decoder_latent_lambda * latent_loss + pred_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            if self.config['AE_tot']:
                # print("we are using AE tot")
                self.update_decoder_tot(obs, obs, next_obs, next_obs, action, L, step)
            else:
                self.update_decoder(obs, obs, L, step)

        if step % self.koopman_update_freq == 0 and self.encoder_type in ['pixel','fc'] and self.koopman_fit_coeff > 0:
            self.update_kpm(obs, next_obs, action, L, step)

        
