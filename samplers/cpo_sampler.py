from collections import defaultdict
from collections import deque, OrderedDict
from itertools import islice

import numpy as np

from utilities.logx import EpochLogger
from utilities.mpi_tools import mpi_sum

class CpoSampler():
    def __init__(self,
                 max_path_length,
                 render_mode = None,
                 logger = None):        
        """
        Sampler for direct interactions with real environment. 
        Stores samples for CPO in provided buffer.

        Args:
            max_path_length(`int`): maximum path length, trajectory terminates
            render_mode(`str`): renders env if provided
            logger(`Logger`): logger, if not provided new one is created
        """
        self._max_path_length = max_path_length
        self._path_length = 0
        self._path_return = 0
        self._path_cost = 0
        self.cum_cost = 0
        if logger:
            self.logger = logger
        else: 
            self.logger = EpochLogger()
        self._render_mode = render_mode
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._last_action = None

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

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
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
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

    @property
    def max_path_length(self):
        return self._max_path_length

    def batch_ready(self):
        """
        Checks if buffer is filled with samples
        """
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
            'rewards': [reward],
            'cost'   : [cost],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self, timestep):
        if self._current_observation is None:
            ### Reset environment
            self._current_observation, reward, terminal, c = np.squeeze(self.env.reset()), 0, False, 0
            self._last_action = np.zeros(shape=self.env.action_space.shape)

        ### Get outputs from policy
        get_action_outs = self.policy.get_action_outs(self._current_observation)

        a = get_action_outs['pi']
        v_t = get_action_outs['v']
        vc_t = get_action_outs['vc']  # Agent may not use cost value func
        logp_t = get_action_outs['logp_pi']
        pi_info_t = get_action_outs['pi_info']

        ### step in env
        next_observation, reward, terminal, info = self.env.step(a)
        if self._render_mode:
            self.env.render(mode=self._render_mode)

        next_observation = np.squeeze(next_observation)
        reward = np.squeeze(reward)
        terminal = np.squeeze(terminal)        
        c = info.get('cost', 0)

        ### store measurements in buffer and log
        self.pool.store(self._current_observation, a, next_observation, reward, v_t, c, vc_t, logp_t, pi_info_t, terminal, timestep)
        self.logger.store(VVals=v_t, CostVVals=vc_t)
        
        self.cum_cost += c
        self._path_length += 1
        self._path_return += reward
        self._path_cost += c
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=a,
            reward=reward,
            cost=c,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        #### update current obs before finishing
        self._current_observation = next_observation
        self._last_action = a

        #### terminate paths 
        if terminal or self._path_length >= self._max_path_length:
            # If trajectory didn't reach terminal state, bootstrap value target(s)
            
            if terminal and not(self._path_length >= self._max_path_length):
                ## Note: we do not count env time-out as true terminal state,
                ## But costs are calculated for the maximum episode length,
                ## even for timed-out paths
                self.finish_all_paths(append_val=False, append_cval=True)
            else:
                self.finish_all_paths(append_val=True, append_cval=True)
            
        return next_observation, reward, terminal, info


    def finish_all_paths(self, append_val=False, append_cval=False, reset_path = True):
            if self._current_observation is None:   #return if already finished
                return
            ####--------------------####
            ####  finish pool traj  ####
            ####--------------------####

            ### If trajectory didn't reach terminal state, bootstrap value target(s)
            if not append_val:
                # Note: we do not count env time out as true terminal state
                last_val = np.zeros((1,))
            else:
                last_val = self.policy.get_v(self._current_observation)
            
            if not append_cval:
                last_cval = np.zeros((1,))
            else:
                last_cval = self.policy.get_vc(self._current_observation)
            
            self.pool.finish_path(last_val, last_cval)

            ####--------------------####
            ####  finish path       ####
            ####--------------------####

            if reset_path:
                self.logger.store(RetEp=self._path_return, EpLen=self._path_length, CostEp=self._path_cost, CostFullEp=self._path_cost/self._path_length * self._max_path_length)
                self.last_path = {
                    field_name: np.array(values)
                    for field_name, values in self._current_path.items()
                }

                self._max_path_return = max(self._max_path_return,
                                            self._path_return)
                self._last_path_return = self._path_return

                self.policy.reset() #does nohing for cpo policy atm
                self._current_observation = None
                self._last_action = np.zeros(shape=self.env.action_space.shape)
                self._path_length = 0
                self._path_return = 0
                self._path_cost = 0
                self._current_path = defaultdict(list)
                self._n_episodes += 1


    def log(self):
        """
        logs several stats over the timesteps since the last 
        flush (such as epCost, totalCost etc.)
        """
        logger = self.logger
        cumulative_cost = mpi_sum(self.cum_cost)    
        cost_rate = cumulative_cost / self._total_samples

        # Performance stats
        logger.log_tabular('RetEp', with_min_and_max=True)
        logger.log_tabular('CostEp', with_min_and_max=True)
        logger.log_tabular('CostFullEp', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('CostCumulative', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)

        # Value function values
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('CostVVals', with_min_and_max=True)

        # Time and steps elapsed
        # logger.log_tabular('TotalEnvInteracts', self._total_samples)
        