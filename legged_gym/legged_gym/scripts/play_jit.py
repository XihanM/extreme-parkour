
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
from torch import nn
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

from rsl_rl.modules import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87

def load_model_path(device):

    path = '../../traced'
    base_model = '336-11-29500-base_jit.pt'
    vision_model = '336-11-29500-vision_weight.pt'

    base_model_path = os.path.join(path, base_model)
    vision_model_path = os.path.join(path, vision_model)

    base_model = torch.jit.load(base_model_path, map_location=device)

    vision_model = torch.load(vision_model_path, map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])

    return base_model, depth_encoder

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    # env_cfg.env.num_envs = 5 if not args.save else 64 # 16
    env_cfg.env.num_envs = 10
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0., # 0
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, # 0.2 
                                    "demo": 0.2
                                    }
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    base_model, depth_encoder = load_model_path(device=env.device)
    base_model.eval()
    depth_encoder.eval()

    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(env.device)[:, -1]

    obs_jit = torch.zeros((env.num_envs, 114), device=env.device)

    for i in range(10*int(env.max_episode_length)):
        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_jit[:, :env.cfg.env.n_proprio])

                depth_latent = depth_latent_and_yaw[:, :-2]
                depth_latent_len = depth_latent.shape[1]
                # n_depth_latent
                obs_jit[:, env.cfg.env.n_proprio:env.cfg.env.n_proprio+depth_latent_len] = depth_latent
                
                yaw = 1.5 * depth_latent_and_yaw[:, -2:]
            obs_jit[:, 6:8] = yaw

        use_estimate_lin_vel = True
        if use_estimate_lin_vel:
            est_lin_vel = estimator(obs[:, :env.cfg.env.n_proprio])
            est_lin_vel_len = est_lin_vel.shape[1]
            obs_jit[:, env.cfg.env.n_proprio+depth_latent_len:env.cfg.env.n_proprio+depth_latent_len+est_lin_vel_len] = est_lin_vel
        
        activation = nn.ELU()
        hist_proprio = obs[:, -env.cfg.env.history_len*env.cfg.env.n_proprio:].view(-1, env.cfg.env.history_len, env.cfg.env.n_proprio)
        history_latent = hist_encoder(activation, hist_proprio)
        
        history_latent_len = history_latent.shape[1]
        obs_jit[:, -history_latent_len:] = history_latent

        actions = actor(obs_jit.detach())

        obs, _, rews, dones, infos = env.step(actions.detach())
        obs_jit[:, :env.cfg.env.n_proprio] = obs[:, :env.cfg.env.n_proprio]

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
