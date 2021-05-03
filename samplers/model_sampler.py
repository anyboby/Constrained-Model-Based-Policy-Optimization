from collections import defaultdict
from collections import deque, OrderedDict
from itertools import islice

import numpy as np
import random

from samplers.cpo_sampler import CpoSampler
from utilities.logx import EpochLogger
from utilities.mpi_tools import mpi_sum
from utilities.utils import EPS

class ModelSampler(CpoSampler):
    def __init__(self,
                 max_path_length,
                 batch_size=1000,
                 store_last_n_paths = 10,
                 rollout_mode = False,
                 logger = None):
        self._max_path_length = max_path_length
        self._path_length = np.zeros(batch_size)

        self.rollout_mode = rollout_mode

        if logger:
            self.logger = logger
        else: 
            self.logger = EpochLogger()

        self._store_last_n_paths = store_last_n_paths
        self._last_n_paths = deque(maxlen=store_last_n_paths)

        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._current_observation = None
        self._last_action = None

        self._total_samples = 0
        self._n_episodes = 0
        self._total_Vs = 0
        self._total_CVs = 0
        self._total_rew = 0
        self._total_rew_var = 0
        self._total_cost = 0
        self._total_cost_var = 0
        self._total_dyn_ep_var = 0
        self._path_dyn_var = 0
        self._total_dkl = 0
        self._max_dkl = 0
        self._dyn_dkl_path = 0
        self._total_mean_var = 0

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_debug_buf(self, pool):
        self.pool_debug = pool

    def set_policy(self, policy):
        self.policy = policy

    def set_logger(self, logger):
        """
        provide a logger (Sampler creates it's own logger by default, 
        but you might want to share a logger between algo, samplers, etc.)
        
        automatically shares logger with agent
        Args: 
            logger : instance of EpochLogger
        """ 
        self.logger = logger        

    def terminate(self):
        self.env.close()

    def get_diagnostics(self):
        diagnostics = OrderedDict({'pool-size': self.pool.size})
        mean_rollout_length = self._total_samples / (self.batch_size+EPS)

        ensemble_rew_var_perstep = self._total_rew_var/(self._total_samples+EPS)
        ensemble_cost_var_perstep = self._total_cost_var/(self._total_samples+EPS)
        ensemble_dyn_var_perstep = self._total_dyn_ep_var/(self._total_samples+EPS)

        if len(self._path_cost.shape)>1:
            cost_sum = np.sum(np.mean(self._path_cost, axis=0))
        else:
            cost_sum = np.sum(self._path_cost)
            
        if len(self._path_return.shape)>1:
            ret_sum = np.sum(np.mean(self._path_return, axis=0))
        else:
            ret_sum = np.sum(self._path_return)

        ensemble_cost_rate = cost_sum/(self._total_samples+EPS)
        ensemble_rew_rate = ret_sum/(self._total_samples+EPS)

        vals_mean = self._total_Vs / (self._total_samples+EPS)

        cval_mean = self._total_CVs / (self._total_samples+EPS)

        dyn_Dkl = self._total_dkl / (self._total_samples+EPS)
        mean_var = self._total_mean_var/ (self._total_samples+EPS)
        diagnostics.update({
            'msampler/samples_added': self._total_samples,
            'msampler/rollout_H_max': self._n_episodes,
            'msampler/rollout_H_mean': mean_rollout_length,
            'msampler/rew_var_perstep': ensemble_rew_var_perstep,
            'msampler/cost_var_perstep' : ensemble_cost_var_perstep,
            'msampler/dyn_var_perstep' : ensemble_dyn_var_perstep,
            'msampler/cost_rate' : ensemble_cost_rate,
            'msampler/rew_rate' : ensemble_rew_rate,
            'msampler/v_mean':vals_mean,
            'msampler/cv_mean':cval_mean,
            'msampler/ens_DKL': dyn_Dkl,
            'msampler/ens_mean_var': mean_var,
            'msampler/max_path_return': self._max_path_return,
            'msampler/max_dkl': self._max_dkl,
        })

        return diagnostics

    def __getstate__(self):
        state = {
            key: value for key, value in self.__dict__.items()
            if key not in ('env', 'policy', 'pool')
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self.env = None
        self.policy = None
        self.pool = None

    def clear_last_n_paths(self):
        self._last_n_paths.clear()

    def compute_dynamics_dkl(self, obs_batch, depth=1):
        for _ in range(depth):
            get_action_outs = self.policy.get_action_outs(obs_batch)
            a = get_action_outs['pi']
            next_obs, _, terminal, info = self.env.step(obs_batch, a)
            dyn_dkl_mean = info.get('ensemble_dkl_mean', 0)
            
            n_paths = next_obs.shape[0]
            self._total_dkl += dyn_dkl_mean*n_paths
            self._total_samples += n_paths

            obs_batch = next_obs[np.squeeze(~terminal)]

        dkl_mean = self.dyn_dkl
        dkl_path_mean = dkl_mean*depth

        return dkl_path_mean
        
    def set_rollout_dkl(self, dkl):
        self.dkl_lim = dkl

    def set_max_path_length(self, path_length):
        self._max_path_length = path_length

    def get_last_n_paths(self, n=None):
        if n is None:
            n = self._store_last_n_paths

        last_n_paths = tuple(islice(self._last_n_paths, None, n))

        return last_n_paths

    @property
    def dyn_dkl(self):
        return self._total_dkl / (self._total_samples+EPS)

    def batch_ready(self):
        return self.pool.size >= self.pool.max_size

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              cost,
                              terminal,
                              next_observation,
                              info):

        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': reward,
            'cost'   : cost,
            'terminals': terminal,
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def reset(self, observations):
        self.batch_size = observations.shape[0]

        self._starting_uncertainty = np.zeros(self.batch_size)
        self._current_observation = observations

        self.policy.reset() #does nohing for cpo policy atm
        self.pool.reset(self.batch_size)

        self._path_length = np.zeros(self.batch_size)
        self._path_return = np.zeros(shape=(self.batch_size))
        self._path_cost = np.zeros(shape=(self.batch_size))

        self._path_return_var = np.zeros(self.batch_size)
        self._path_cost_var = np.zeros(self.batch_size)
        self._path_dyn_var = np.zeros(self.batch_size)
        self._dyn_dkl_path = np.zeros(self.batch_size)

        self._total_samples = 0
        self._n_episodes = 0
        self._total_Vs = 0
        self._total_CVs = 0
        self._total_cost = 0
        self._total_cost_var = 0
        self._total_rew = 0
        self._total_rew_var = 0
        self._total_dyn_ep_var = 0
        self._total_dkl = 0
        self._max_dkl = 0
        self._total_mean_var = 0
        self._max_path_return = 0

    def sample(self, max_samples=None):
        assert self.pool.has_room           #pool full! empty before sampling.
        assert self._current_observation is not None # reset before sampling !
        assert self.pool.alive_paths.any()  # reset before sampling !

        self._n_episodes += 1
        alive_paths = self.pool.alive_paths
        current_obs = self._current_observation

        # Get outputs from policy
        get_action_outs = self.policy.get_action_outs(current_obs)
        
        a = get_action_outs['pi']
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        v_t = get_action_outs['v']
        vc_t = get_action_outs['vc']
        #####

        ## ____________________________________________ ##
        ##                      Step                    ##
        ## ____________________________________________ ##

        next_obs, reward, terminal, info = self.env.step(current_obs, a)

        reward = np.squeeze(reward, axis=-1)
        c = np.squeeze(info.get('cost', np.zeros(reward.shape)))
        terminal = np.squeeze(terminal, axis=-1)
        dyn_dkl_mean = info.get('ensemble_dkl_mean', 0)
        dyn_dkl_path = info.get('ensemble_dkl_path', 0)
        dyn_ep_var = info.get('ensemble_ep_var', np.zeros(shape=reward.shape[1:]))
        ens_mean_var = info.get('ensemble_mean_var', 0)

        if self._n_episodes == 1:
            self._starting_uncertainty = np.mean(dyn_ep_var, axis=-1)
            self._starting_uncertainty_dkl = dyn_dkl_path
        ## ____________________________________________ ##
        ##    Check Uncertainty f. each Trajectory      ##
        ## ____________________________________________ ##

        ### check if too uncertain before storing info of the taken step 
        ### (so we don't take a "bad step" by appending values of next state)
        if self.rollout_mode=='uncertainty':
            next_dkl = self._dyn_dkl_path[self.pool.alive_paths]+dyn_dkl_path
            too_uncertain_paths = next_dkl>=self.dkl_lim
        else:
            too_uncertain_paths = np.zeros(shape=self.pool.alive_paths.sum(), dtype=np.bool)
        
        ### early terminate paths if max_samples is given
        if max_samples:
            n = self._total_samples + alive_paths.sum() - too_uncertain_paths.sum()
            n = max(n-max_samples, 0)
            early_term = np.zeros_like(too_uncertain_paths[~too_uncertain_paths], dtype=np.bool)
            early_term[:n] = True
            too_uncertain_paths[~too_uncertain_paths] = early_term

        ### finish too uncertain paths before storing info of the taken step into buffer
        # remaining_paths refers to the paths we have finished and has the same shape 
        # as our terminal mask (too_uncertain_mask)
        # alive_paths refers to all original paths and therefore has shape batch_size
        remaining_paths = self._finish_paths(too_uncertain_paths, append_vals=True, append_cvals=True)
        alive_paths = self.pool.alive_paths
        if not alive_paths.any():
            info['alive_ratio'] = 0
            return next_obs, reward, terminal, info

        ## ____________________________________________ ##
        ##    Store Info of the remaining paths         ##
        ## ____________________________________________ ##
        current_obs     = current_obs[remaining_paths]
        a               = a[remaining_paths]
        next_obs        = next_obs[remaining_paths]
        reward          = reward[remaining_paths]
        v_t             = v_t[remaining_paths]
        c               = c[remaining_paths]
        vc_t            = vc_t[remaining_paths]
        terminal        = terminal[remaining_paths]
        dyn_dkl_path    = dyn_dkl_path[remaining_paths]
        logp_t          = logp_t[remaining_paths]
        pi_info_t       = {k:v[remaining_paths] for k,v in pi_info_t.items()}

        dyn_ep_var      = dyn_ep_var[remaining_paths]

        #### update some sampler infos
        self._total_samples += alive_paths.sum()

        self._total_cost += c.sum()
        self._total_rew += reward.sum()
        self._path_return[alive_paths] += reward
        self._path_cost[alive_paths] += c

        self._path_length[alive_paths] += 1
        self._path_dyn_var[alive_paths] += np.mean(dyn_ep_var, axis=-1)
        self._total_dyn_ep_var += dyn_ep_var.sum()

        self._total_Vs += v_t.sum()
        self._total_CVs += vc_t.sum()

        self._total_dkl += dyn_dkl_mean*alive_paths.sum()
        self._total_mean_var =+ ens_mean_var*alive_paths.sum()

        self._max_dkl = max(self._max_dkl, np.max(dyn_dkl_path))
        self._dyn_dkl_path[alive_paths] += dyn_dkl_path
        self._max_path_return = max(self._max_path_return, np.max(self._path_return))

        #### only store one trajectory in buffer 
        self.pool.store_multiple(current_obs,
                                        a,
                                        next_obs,
                                        reward,
                                        v_t,
                                        c,
                                        vc_t,
                                        np.mean(dyn_ep_var, axis=-1),
                                        logp_t,
                                        pi_info_t,
                                        terminal)

        #### terminate mature termination due to path length
        ## update obs before finishing paths (_finish_paths() uses current obs)
        self._current_observation = next_obs

        path_end_mask = (self._path_length >= self._max_path_length-1)[alive_paths]            
        remaining_paths = self._finish_paths(term_mask=path_end_mask, append_vals=True, append_cvals=True)
        if not remaining_paths.any():
            info['alive_ratio'] = 0
            return next_obs, reward, terminal, info

        ## update remaining paths and obs
        self._current_observation = self._current_observation[remaining_paths]

        prem_term_mask = terminal
        
        #### terminate real termination due to env end
        remaining_paths = self._finish_paths(term_mask=prem_term_mask, append_vals=False, append_cvals=True)
        if not remaining_paths.any():
            info['alive_ratio'] = 0
            return next_obs, reward, terminal, info

        ### update alive paths
        alive_paths = self.pool.alive_paths
        
        self._current_observation = self._current_observation[remaining_paths]

        alive_ratio = sum(alive_paths)/self.batch_size
        info['alive_ratio'] = alive_ratio

        return next_obs, reward, terminal, info

    def _finish_paths(self, term_mask, append_vals=False, append_cvals=False):
        """
        terminates paths that are indicated in term_mask. Append_vals should be set to 
        True/False to indicate, whether values of the current states of those paths should 
        be appended (Note: Premature termination due to environment term should not 
        include appended values, while Mature termination upon path length excertion should 
        include appended values)

        Warning! throws error if trying to terminate an already terminated path. 

        Args:
            term_mask: Mask with the shape of the currently alive paths that indicates which 
                paths should be termianted
            append_vals: True/False whether values of the current state should be appended
        
        Returns: 
            remaining_mask: A Mask that indicates the remaining alive paths. Has the same shape 
                as the arg term_mask
        """
        if not term_mask.any():
            return np.logical_not(term_mask)

        # We do not count env time out (mature termination) as true terminal state, append values
        if append_vals:
            last_val = self.policy.get_v(self._current_observation[term_mask])
        else:
            # init final values
            last_val = np.zeros(shape=(term_mask.sum()))

        if append_cvals:
            last_cval = self.policy.get_vc(self._current_observation[term_mask])
        else:
            # init final values
            last_cval = np.zeros(shape=(term_mask.sum()))

        self.pool.finish_path_multiple(term_mask, last_val, last_cval)

        remaining_path_mask = np.logical_not(term_mask)

        return remaining_path_mask
        
    def finish_all_paths(self):

        alive_paths=self.pool.alive_paths ##any paths that are still alive did not terminate by env
        # init final values and quantify according to termination type
        # Note: we do not count env time out as true terminal state
        if not alive_paths.any(): return self.get_diagnostics()

        if alive_paths.any():
            term_mask = np.ones(shape=alive_paths.sum(), dtype=np.bool)
            if self.policy.agent.reward_penalized:
                last_val = self.policy.get_v(self._current_observation)
            else:
                last_val = self.policy.get_v(self._current_observation)
                last_cval = self.policy.get_vc(self._current_observation)

            self.pool.finish_path_multiple(term_mask, last_val, last_cval)
            
        alive_paths = self.pool.alive_paths
        assert alive_paths.sum()==0   ## something went wrong with finishing all paths
        
        return self.get_diagnostics()