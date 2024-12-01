import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder

from algorithms.rl_utils import (
    compute_TID_score
)
import logging
import sys
import cv2
import imageio

def evaluate(env, agent, save_video, num_episodes, L, step, test_env=False, mode=None):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        torch_obs = []
        torch_action = []
        vid_type = 2
        #video.init(enabled=(i==vid_type))
        if save_video:
            if i==vid_type:
                vid_frames = []
                _obs = agent._obs_to_input(obs)
                vid_frame = np.uint8(_obs[0,:3].permute(1,2,0).cpu().data.numpy())
                vid_frame = cv2.resize(vid_frame, (384, 384))
                vid_frames.append(vid_frame)
        
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            #video.record(env, mode)
            episode_reward += reward
            
            if save_video:
                if i == vid_type:
                    _obs = agent._obs_to_input(obs)
                    vid_frame = np.uint8(_obs[0,:3].permute(1,2,0).cpu().data.numpy())
                    vid_frame = cv2.resize(vid_frame, (384, 384))
                    vid_frames.append(vid_frame)
            episode_step += 1
        if save_video:
            if i==vid_type:
                video_dir = os.path.join(L._log_dir,'video')
                if not os.path.isdir(video_dir):
                    os.mkdir(video_dir)
                train_or_test = "train" if test_env==False else mode
                vid_name = str(step) + '_' + train_or_test + '.mp4'
                imageio.mimsave(os.path.join(video_dir, vid_name), vid_frames, fps=25)
            
        if L is not None:
            _test_env = '_test' if test_env else ''
            #video.save(f'{step}{_test_env}.mp4')
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
            
        episode_rewards.append(episode_reward)
        
    return np.mean(episode_rewards)

def eval_tid(agent, step, work_dir_tid, tid_samples, tid_logger, rho_value="auto", test=0):
    if not os.path.isdir(f'{work_dir_tid}{"/"}{step}'):
        os.mkdir(f'{work_dir_tid}{"/"}{step}')
    if test==1:
        tid_path = f'{work_dir_tid}{"/"}{step}{"/"}{"02.test"}{"/"}'
        tid_path_rho = f'{work_dir_tid}{"/"}{step}{"/"}{"02.test"}{"/"}{"rho_"}{rho_value}{"/"}'
    else:
        tid_path = f'{work_dir_tid}{"/"}{step}{"/"}{"01.train"}{"/"}'
        tid_path_rho = f'{work_dir_tid}{"/"}{step}{"/"}{"01.train"}{"/"}{"rho_"}{rho_value}{"/"}'
    if not os.path.isdir(tid_path):
        os.mkdir(tid_path)
    if not os.path.isdir(tid_path_rho):
        os.mkdir(tid_path_rho)
    obs = tid_samples[0]
    gt_mask = tid_samples[1]
    with torch.no_grad():
        tid_score, tid_var = compute_TID_score(obs, agent.actor, agent.critic, tid_path_rho, gt_mask = gt_mask, rho_value=rho_value)
    if test==1:
        tid_logger.info(("Step", step, "TID Score_test", tid_score, "TID Variance_test", tid_var, "rho", rho_value))
    else:
        tid_logger.info(("Step", step, "TID Score", tid_score, "TID Variance", tid_var, "rho", rho_value))

def main(args):
	# Set seed
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,100)
    utils.set_seed_everywhere(args.seed)
    
	# Initialize environments
    gym.logger.set_level(40)
    env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='train'
	)
    
    test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode,
		intensity=args.distracting_cs_intensity
	) if args.eval_mode is not None else None
    

	# Create working directory
    work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
    print('Working directory:', work_dir)
    assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    #video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, 'info.log'))
    
    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
    cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
    print('Observations:', env.observation_space.shape)
    print('Cropped observations:', cropped_obs_shape)
    agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
    # Set TID
    work_dir_tid = os.path.join(
            args.log_dir,
            args.domain_name + "_" + args.task_name,
            args.algorithm,
            str(args.seed),
            "TID",
        )
    tid_data_dir = '../TID_Image/'+args.domain_name+'.npy'
    tid_dict = np.load(tid_data_dir, allow_pickle=True).item()
    tid_keys = list(tid_dict.keys())
    tid_keys.sort()
    tid_train_mask = tid_dict[tid_keys[0]]
    tid_train_obs = tid_dict[tid_keys[1]]
    tid_test_mask = tid_dict[tid_keys[2]]
    tid_test_obs = tid_dict[tid_keys[3]]
    
    tid_train_mask_torch = agent._obs_to_input(tid_train_mask)[0].permute(0,3,1,2)
    tid_train_obs_torch = agent._obs_to_input(tid_train_obs)[0].permute(0,3,1,2)
    tid_test_mask_torch = agent._obs_to_input(tid_test_mask)[0].permute(0,3,1,2)
    tid_test_obs_torch = agent._obs_to_input(tid_test_obs)[0].permute(0,3,1,2)
    tid_samples_train = [tid_train_obs_torch, tid_train_mask_torch]
    tid_samples_test = [tid_test_obs_torch, tid_test_mask_torch]
    if not os.path.isdir(f'{work_dir_tid}'):
        os.mkdir(f'{work_dir_tid}')
    tid_log_dir = f'{work_dir_tid}{"/"}{"tid.log"}'
    tid_logger = logging.getLogger()
    tid_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(tid_log_dir)
    tid_logger.addHandler(file_handler)
    #
    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    for step in range(start_step, args.train_steps+1):
        if done:
            if step > start_step:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)
                
            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print('Evaluating:', work_dir)
                L.log('eval/episode', episode, step)
                
                evaluate(env, agent, args.save_video, args.eval_episodes, L, step)
                if test_env is not None:
                    evaluate(test_env, agent, args.save_video, args.eval_episodes, L, step, test_env=True, mode=args.eval_mode)
                
            L.dump(step)

			# Save agent periodically
            if step > start_step and step % args.save_freq == 0:
                torch.save(agent, os.path.join(model_dir, f'{step}.pt'))
            
            L.log('train/episode_reward', episode_reward, step)
            
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            
            L.log('train/episode', episode, step)
            
            if step % 10000==0:
                eval_tid(agent, step, work_dir_tid, tid_samples_train, tid_logger, rho_value="auto")
                eval_tid(agent, step, work_dir_tid, tid_samples_test, tid_logger, rho_value="auto", test=1)
                
                eval_tid(agent, step, work_dir_tid, tid_samples_train, tid_logger, rho_value=0.9)
                eval_tid(agent, step, work_dir_tid, tid_samples_test, tid_logger, rho_value=0.9, test=1)
                
                eval_tid(agent, step, work_dir_tid, tid_samples_train, tid_logger, rho_value=0.95)
                eval_tid(agent, step, work_dir_tid, tid_samples_test, tid_logger, rho_value=0.95, test=1)

		# Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)
                
        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
                
        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs
        
        episode_step += 1
    
    print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)
