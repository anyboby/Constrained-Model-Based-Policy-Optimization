from gym.envs.registration import register
import gym

import os
import sys
dirpath = os.path.dirname(os.path.dirname(__file__))
sys.path.append(dirpath)

env_specs = gym.envs.registry.env_specs

if 'HumanoidSafe-v2' not in env_specs:
    register(
        id='HumanoidSafe-v2',
        entry_point='mujoco_safety_gym.envs:HumanoidEnv',
        max_episode_steps=1000,
    )
if 'AntSafe-v2' not in env_specs:
    register(
        id='AntSafe-v2',
        entry_point='mujoco_safety_gym.envs:AntEnv',
        max_episode_steps=1000,
    )
if 'AntSafeVisualize-v2' not in env_specs:
    register(
        id='AntSafeVisualize-v2',
        entry_point='mujoco_safety_gym.envs:AntEnvVisualize',
        max_episode_steps=1000,
    )
if 'HopperSafe-v2' not in env_specs:    
    register(
        id='HopperSafe-v2',
        entry_point='mujoco_safety_gym.envs:HopperEnv',
        max_episode_steps=1000,
    )
if 'HalfCheetahSafe-v2' not in env_specs:
    register(
        id='HalfCheetahSafe-v2',
        entry_point='mujoco_safety_gym.envs:HalfCheetahEnv',
        max_episode_steps=1000,
    )
if 'FetchPushSafety-v0' not in env_specs:
    register(
        id='FetchPushSafety-v0', 
        entry_point='mujoco_safety_gym.envs:FetchPushEnv', 
        max_episode_steps=1000,
    )
if 'FetchReachSafety-v0' not in env_specs:
    register(
        id='FetchReachSafety-v0', 
        entry_point='mujoco_safety_gym.envs:FetchReachEnv', 
        max_episode_steps=1000,
    )
if 'FetchSlideSafety-v0' not in env_specs:
    register(
        id='FetchSlideSafety-v0', 
        entry_point='mujoco_safety_gym.envs:FetchSlideEnv', 
        max_episode_steps=1000,
    )