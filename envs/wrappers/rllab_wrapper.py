"""Implements a GymWrapper that converts rllab envs into GymEnv."""

import numpy as np
import gym
from gym import spaces
from gym.spaces import Box, Discrete
from collections import defaultdict

import json
import os
import sys

#### Warning: unclean, but rllab would otherwise require 
    # modifications to work directly
    # We're adding rllab to path, so imports within rllab work, as
    # well as <isintance()> queries etc.
from pathlib import Path
import envs.rllab.rllab as rllab_dir        ## add rllab to sys path for internal imports
path = str(Path(os.path.dirname(rllab_dir.__file__)).parent.absolute())
sys.path.append(path)

from rllab.spaces.box import Box as rllBox
from rllab.spaces.discrete import Discrete as rllDiscrete
from rllab.spaces.product import Product as rllProduct
from sandbox.cpo.envs.mujoco.gather.point_gather_env import PointGatherEnv
from sandbox.cpo.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from sandbox.cpo.envs.mujoco_safe.ant_env_safe import SafeAntEnv
from sandbox.cpo.envs.mujoco_safe.point_env_safe import SafePointEnv
from sandbox.cpo.envs.mujoco_safe.half_cheetah_env_safe import SafeHalfCheetahEnv
from sandbox.cpo.envs.mujoco_safe.simple_humanoid_env_safe import SafeSimpleHumanoidEnv

RLLAB_ENVIRONMENTS = {
    'PointGather':['v0'],
    'AntGather':['v0'],
    'AntCircle':['v0'],
    'PointCircle':['v0'],
    'HalfCheetahCircle':['v0'],
    'SimplehumanoidCircle':['v0'],
}

OVERWRITE = {
    'SimplehumanoidCircle-v0':{
        'observation_space': lambda x: Box(low=np.ones(shape=36)*x.low[0], high=np.ones(shape=36)*x.high[0]),
        'next_observation': lambda x: np.delete(x, np.s_[-69:-3]),
    }
}

RLLAB_ENTRIES = {
    'PointGather-v0':PointGatherEnv,
    'AntGather-v0':AntGatherEnv,
    'AntCircle-v0':SafeAntEnv,
    'PointCircle-v0':SafePointEnv,
    'HalfCheetahCircle-v0':SafeHalfCheetahEnv,
    'SimplehumanoidCircle-v0':SafeSimpleHumanoidEnv,
}

RLLAB_KWARGS = {
    'PointGather-v0':{
        'apple_reward':10,
        'bomb_cost':1,
        'n_apples':2,
        'activity_range':6,
        },
    'AntGather-v0':{
        'apple_reward':10,
        'bomb_cost':1,
        'n_apples':2,
        'activity_range':6,
        },
    'AntCircle-v0':{                       
        'xlim':3,
        'circle_mode':True,
        'target_dist':10,
        'abs_lim':True,
        },
    'PointCircle-v0':{},
    'HalfCheetahCircle-v0':{},
    'SimplehumanoidCircle-v0':{
        'xlim':2.5,
        'circle_mode':True,
        'target_dist':10,
        'abs_lim':True,
    },
}

#### Cost params according to cpo paper
RLLAB_COST_PARAMS = {
    'AntCircle-v0': {
        'cidx':-3,
        'xlim': 3,
        'target_dist': 10 },
    'SimplehumanoidCircle-v0':{
        'cidx':-3,
        'xlim': 2.5,
        'target_dist': 10 },
    }

def eval_cost_gather(obs, info, env_id):
    return info['bombs']

def eval_cost_mjc(obs, info, env_id):
    return float(np.abs(obs[RLLAB_COST_PARAMS[env_id]['cidx']]) >= RLLAB_COST_PARAMS[env_id]['xlim'])

RLLAB_COSTF = {
    'PointGather-v0':eval_cost_gather,
    'AntGather-v0':eval_cost_gather,
    'AntCircle-v0':eval_cost_mjc,
    'PointCircle-v0':eval_cost_mjc,
    'HalfCheetahCircle-v0':eval_cost_mjc,
    'SimplehumanoidCircle-v0':eval_cost_mjc,
}

class RllabEnv(gym.Env):
    """Adapter that implements gym environments for rllab envs."""

    def __init__(self,
                 task,
                 *args,
                 **kwargs):
        assert not args, (
            "Gym environments don't support args. Use kwargs instead.")
        assert task is not None, "Please provide a task!"
        super(RllabEnv, self).__init__(**kwargs)
        self.env_id = env_id = task
        env = RLLAB_ENTRIES[env_id](**RLLAB_KWARGS[env_id])
        self._env = env

    @property
    def observation_space(self):
        observation_space = self._env.observation_space
        obs_space_gym = self._convert_to_gym_space(observation_space)
        if self.env_id in OVERWRITE:
            obs_space_gym = OVERWRITE[self.env_id]['observation_space'](obs_space_gym)        
        return obs_space_gym

    @property
    def action_space(self, *args, **kwargs):
        action_space = self._env.action_space
        action_space_gym = self._convert_to_gym_space(action_space)
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Action space ({}) is not flat, make sure to check the"
                " implemenation.".format(action_space))
        if self.env_id in OVERWRITE:
            action_space_gym = OVERWRITE[self.env_id].get('action_space', action_space_gym)
        return action_space_gym

    def step(self, action, *args, **kwargs):
        o, r, done, info = self._env.step(action, *args, **kwargs)
        info['cost'] = RLLAB_COSTF[self.env_id](o, info, self.env_id)
        
        if self.env_id in OVERWRITE:
            o = OVERWRITE[self.env_id]['next_observation'](o)

        return o, r, done, info

    def reset(self, *args, **kwargs):
        o = self._env.reset(*args, **kwargs)
        if self.env_id in OVERWRITE:
            o = OVERWRITE[self.env_id]['next_observation'](o)
        return o

    def render(self, *args, **kwargs):
        mode = 'human' if 'human' in args else None
        mode = 'rgb_array' if 'rgb_array' in args else mode

        return self._env.render(mode=mode, **kwargs)

    def close(self, *args, **kwargs):
        return self._env.close(*args, **kwargs)

    def get_sim_state(self, *args, **kwargs):       #### @anyboby
        assert hasattr(self._env, 'get_sim_state')
        return self._env.get_sim_state(*args, **kwargs)

    def seed(self, *args, **kwargs):
        return self._env.seed(*args, **kwargs)
    
    @property
    def unwrapped(self):
        return self._env

    def _convert_to_gym_space(self, space):
        if isinstance(space, rllBox):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, rllDiscrete):
            return Discrete(n=space.n)
        else:
            raise NotImplementedError