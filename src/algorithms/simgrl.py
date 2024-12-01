import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC
import augmentations
import os

from .rl_utils import (
    compute_attribution,
    compute_attribution_mask,
    make_attribution_pred_grid,
    make_obs_grid,
    make_obs_grad_grid,
)
import matplotlib.pyplot as plt


class SimGRL(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        
        self.algorithm = args.algorithm
        
        self.init_steps = args.init_steps
        self.work_dir = os.path.join(
            args.log_dir,
            args.domain_name + "_" + args.task_name,
            args.algorithm,
            str(args.seed),
            "save_mask",
        )
        
    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        B, C, H, W = obs.shape
        obs_clean =obs.clone()
        next_obs_clean = next_obs.clone()
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            
            if self.algorithm in ['simgrl', 'simgrl-S']:
                obs = augmentations.random_overlay_shift(obs)
            else:
                obs = augmentations.random_overlay(obs)
            
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
   
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        current_Q1_clean, current_Q2_clean = self.critic(obs_clean, action)
        critic_loss_clean = F.mse_loss(current_Q1_clean, target_Q) + F.mse_loss(current_Q2_clean, target_Q)
        critic_loss = critic_loss + critic_loss_clean
        critic_loss = 0.5*critic_loss

        if L is not None:
            L.log('train_critic/loss', critic_loss, step)
            
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if step%10000==0:
            if not os.path.isdir(f'{self.work_dir}'):
                os.mkdir(f'{self.work_dir}')
            if not os.path.isdir(f'{self.work_dir}{"/"}{step}'):
                os.mkdir(f'{self.work_dir}{"/"}{step}')
            with torch.no_grad():
                obs_grad = compute_attribution(self.critic, obs_clean, action.detach())
                mask = compute_attribution_mask(obs_grad, quantile=0.95).float()
                obs_mask = obs_clean*mask
                for i in range(2):
                    for j in range(3):
                        plt.imsave(f'{self.work_dir}{"/"}{step}{"/"}{"obs_clean_"}{i}{"_"}{j}.jpg',obs_clean[i][3*j:3*(j+1)].permute(1,2,0).cpu().data.numpy()/255)
                        plt.imsave(f'{self.work_dir}{"/"}{step}{"/"}{"obs_masked_"}{i}{"_"}{j}.jpg',obs_mask[i][3*j:3*(j+1)].permute(1,2,0).cpu().data.numpy()/255)
                        plt.imsave(f'{self.work_dir}{"/"}{step}{"/"}{"mask_"}{i}{"_"}{j}.jpg',mask[i][3*j:3*(j+1)].permute(1,2,0).cpu().data.numpy())
                        plt.imsave(f'{self.work_dir}{"/"}{step}{"/"}{"obs_aug_"}{i}{"_"}{j}.jpg',obs[i][3*j:3*(j+1)].permute(1,2,0).cpu().data.numpy()/255)
        
        
    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()
        
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        
        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)
                        
        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
        
        
