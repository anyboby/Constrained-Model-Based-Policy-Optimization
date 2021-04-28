import numpy as np
import tensorflow as tf
import pdb

from models.pens.pe_factory import build_PE, format_samples_for_dyn, format_samples_for_cost
from models.pens.utils import average_dkl, median_dkl
from models.statics import (REWS_BY_TASK, COST_BY_TASK, TERMS_BY_TASK)

from itertools import count
import warnings
import time

EPS = 1e-8

class FakeEnv:

    def __init__(self, 
                    true_environment,
                    task,
                    model,
                    predicts_delta,
                    predicts_rew,
                    predicts_cost,
                    ):
        
        self.env = true_environment
        self.obs_dim = np.prod(self.observation_space.shape)
        self.act_dim = np.prod(self.action_space.shape)
        self._task = task
        
        self._model = model
        self._uses_ensemble = self._model.is_ensemble
        self._is_probabilistic = self._model.is_probabilistic
        
        self._predicts_delta = predicts_delta
        self._predicts_rew = predicts_rew
        self._predicts_cost = predicts_cost

        #### create fake env from model
        self.input_dim = self._model.in_dim
        self.output_dim = self._model.out_dim
        
    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
    
    def step(self, obs, act, deterministic=True):
        assert len(obs.shape) == len(act.shape)
        assert obs.shape[-1]==self.obs_dim and act.shape[-1]==self.act_dim

        obs_depth = len(obs.shape)
        if obs_depth == 1:
            obs = obs[None]
            act = act[None]
            return_single=True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)

        if obs_depth==3:
            inputs, shuffle_indxs = self.forward_shuffle(inputs)

        if self._uses_ensemble:
            pred = self._model.predict_ensemble(inputs)       #### dyn_vars gives ep. vars for 
                                                                                                    ## deterministic ensembles and al. var for probabilistic
        else:
            pred = self._model.predict(inputs)
        
        if self._is_probabilistic:
            pred_mean, pred_var = pred
        else:
            pred_mean, pred_var = pred, np.zeros_like(pred)

        if obs_depth==3:
            pred_mean, pred_var = self.inverse_shuffle(pred_mean, shuffle_indxs), self.inverse_shuffle(pred_var, shuffle_indxs)
        
        #### probabilistic transitions if var is predicted
        pred_std = np.sqrt(pred_var)
        if not deterministic:
            next_obs = pred_mean[...,:self.obs_dim] + pred_std[...,:self.obs_dim]
        else:
            next_obs = pred_mean[...,:self.obs_dim]

        #### extract uncertainty measures
        if self._uses_ensemble:
            ens_ep_var = np.var(next_obs, axis=0)
            ens_dkl_path = np.mean(average_dkl(next_obs, pred_std[...,:self.obs_dim]), axis=-1) ##@anyboby gives ugly numbers if var=0
            ens_dkl_mean = np.mean(ens_dkl_path)
        else:
            ens_ep_var = 0
            ens_dkl_path = np.zeros(shape=obs.shape[1])
            ens_dkl_mean = 0

        #### choose one model from ensemble randomly, if ensemble and not 3d inputs
        if self._uses_ensemble and obs_depth<3:
            _, batch_size, _ = next_obs.shape
            model_inds = self.random_inds(batch_size) ## only elites
            batch_inds = np.arange(0, batch_size)
            next_obs = next_obs[model_inds, batch_inds]
        else:
            next_obs = next_obs
        ##########################
        
        #### add to obs if delta predictions
        if self._predicts_delta:
            next_obs += obs

        #### extract rew, cost, or call fallback functions for terms, rews and costs
        if TERMS_BY_TASK.get(self._task, None):
            terms = TERMS_BY_TASK[self._task](obs, act, next_obs)
        else: 
            terms = TERMS_BY_TASK['default'](obs, act, next_obs)

        if self._predicts_cost:
            c = pred_mean[...,-1:]
            c = c[model_inds, batch_inds]
            pred_mean = pred_mean[...,:-1]
        elif COST_BY_TASK.get(self._task, None):
            c = COST_BY_TASK[self._task](obs, act, next_obs)
        else: 
            c = np.zeros_like(terms)

        if self._predicts_rew:
            r = pred_mean[...,-1:]
            r = r[model_inds, batch_inds]
            pred_mean = pred_mean[...,:-1]
        elif REWS_BY_TASK.get(self._task, None):
            r = REWS_BY_TASK[self._task](obs, act, next_obs)

        assert r is not None, \
            "Please provide either static functions or predictions for rewards, costs and terms"

        if return_single:
            next_obs = next_obs[0]
            r = r[0]
            c = c[0]
            terms = terms[0]

        info = {
                'ensemble_dkl_mean' : ens_dkl_mean,
                'ensemble_dkl_path' : ens_dkl_path,
                'ensemble_ep_var' : ens_ep_var,
                'rew':r,
                'cost':c,
                }

        return next_obs, r, terms, info

    def random_inds(self, size):
        if self._model.is_ensemble:
            return np.random.choice(self._model.elite_inds, (size))
        else:
            return np.random.choice([0], (size))
        
    def forward_shuffle(self, ndarray):
        """
        shuffles ndarray forward along axis 0 with random elite indices, 
        Returns shuffled copy of ndarray and indices with which was shuffled
        """
        idxs = np.random.permutation(ndarray.shape[0])
        shuffled = ndarray[idxs]
        return shuffled, idxs

    def inverse_shuffle(self, ndarray, idxs):
        """
        inverses a shuffle of ndarray forward along axis 0, given the used indices. 
        Returns unshuffled copy of ndarray
        """
        unshuffled = ndarray[idxs]
        return unshuffled

    def close(self):
        pass
