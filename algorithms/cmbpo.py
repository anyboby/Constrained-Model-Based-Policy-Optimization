
import os
import math
import pickle
from collections import OrderedDict
from collections import defaultdict
from numbers import Number
from itertools import count
import gtimer as gt
import pdb
import random
import sys
import warnings

import numpy as np
import tensorflow as tf

from algorithms.rl_algorithm import RLAlgorithm
from buffers.modelbuffer import ModelBuffer
from samplers.model_sampler import ModelSampler
from utilities.logx import EpochLogger
from utilities.utils import EPS

from models.fake_env import FakeEnv
from models.pens.pe_factory import (build_PE, 
                                    format_samples_for_cost,
                                    format_samples_for_dyn,)

from utilities.logging import Progress, update_dict


class CMBPO(RLAlgorithm):
    """Constrained Model-Based Policy Optimization (MBPO)
    """

    def __init__(
            self,
            env,
            policy,
            buffer,
            task='default',
            static_fns=None,
            n_env_interacts = 1e7,
            eval_every_n_steps=5e3,
            use_model = True,
            m_train_freq = 250,
            m_loss_type = 'MSPE',
            m_use_scaler_in = True,
            m_use_scaler_out = True,
            m_lr = 1e-3,
            m_networks = 7,
            m_elites = 5,
            m_hidden_dims=(200, 200, 200, 200),
            max_model_t=None,
            rollout_batch_size=10e3,
            sampling_alpha = 1,
            rollout_mode = 'uncertainty',
            rollout_schedule=[20,100,1,1],
            maxroll = 80,
            initial_real_samples_per_epoch = 5000,
            min_real_samples_per_epoch = 500,
            batch_size_policy = 25000,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            buffer (`CPOBuffer`): Replay pool to add gathered samples to.
        """

        super(CMBPO, self).__init__(**kwargs)

        self.obs_space = env.observation_space
        self.act_space = env.action_space
        self.obs_dim = np.prod(env.observation_space.shape)        
        self.act_dim = np.prod(env.action_space.shape)
        
        self.n_env_interacts = n_env_interacts
        self._task = task


        #### others
        self.eval_every_n_steps= eval_every_n_steps
        self._log_dir = os.getcwd()
        self._training_environment = env
        self._policy = policy
        self._initial_exploration_policy = policy   #overwriting initial _exploration policy, not implemented for cpo yet
        self.sampling_alpha = sampling_alpha

        #### set up buffer
        self._buffer = buffer
        pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in self._policy.pi_info_phs.items()}
        self._buffer.initialize(pi_info_shapes,
                                gamma = self._policy.gamma,
                                lam = self._policy.lam,
                                cost_gamma = self._policy.cost_gamma,
                                cost_lam = self._policy.cost_lam)

        #### create model fake environment
        self._use_model = use_model
        self._m_networks = m_networks
        self._m_elites = m_elites
        self._m_train_freq = m_train_freq
        self._m_loss_type = m_loss_type
        self._m_use_scaler_in = m_use_scaler_in
        self._m_use_scaler_out = m_use_scaler_out
        self._m_hiddel_dims = m_hidden_dims
        self._m_lr = m_lr
        self._rollout_batch_size = int(rollout_batch_size)
        self._rollout_schedule = rollout_schedule
        self._max_model_t = max_model_t


        #======================================================#
        #            Model Setup                               #
        #======================================================#        
        if use_model:
            self._model = build_PE(
                in_dim= self.obs_dim + self.act_dim,
                out_dim= self.obs_dim + 1,
                name='DynEns',
                loss=m_loss_type,
                hidden_dims=m_hidden_dims,
                lr=m_lr,
                num_networks=m_networks,
                num_elites=m_elites,
                use_scaler_in = m_use_scaler_in,
                use_scaler_out = m_use_scaler_out,
                decay=1e-6,
                max_logvar=.5,
                min_logvar=-10,
                session=self._session
            )
            self.fake_env = FakeEnv(true_environment=env,
                                    task=self._task,
                                    model=self._model,
                                    predicts_delta=True,
                                    predicts_rew=True,
                                    predicts_cost=False)

            #### model buffer
            self.rollout_mode = rollout_mode
            self.model_buf = ModelBuffer(batch_size=self._rollout_batch_size, 
                                            max_path_length=maxroll, 
                                            env = self.fake_env,
                                            )

            self.model_buf.initialize(pi_info_shapes,
                                        gamma = self._policy.gamma,
                                        lam = self._policy.lam,
                                        cost_gamma = self._policy.cost_gamma,
                                        cost_lam = self._policy.cost_lam,
                                        )

            #### model sampler
            self.model_sampler = ModelSampler(max_path_length=maxroll,
                                                batch_size=self._rollout_batch_size,
                                                store_last_n_paths=10,
                                                logger=None,
                                                rollout_mode = self.rollout_mode,
                                                )
        ##########################################

        ### batch sizes for model and policy
        self.init_real_samples = initial_real_samples_per_epoch
        self.min_real_samples = min_real_samples_per_epoch
        self.batch_size_policy = batch_size_policy

        #### provide policy and sampler with the same logger
        self.logger = EpochLogger()
        self._policy.set_logger(self.logger)    
        self.sampler.set_logger(self.logger)

    def _train(self):
        
        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """

        env_r = self._training_environment
        policy = self._policy
        pool = self._buffer

        if not self._training_started:
            #### perform some initial steps (gather samples) using initial policy
            ######  fills pool with _n_initial_exploration_steps samples
            self._initial_exploration_hook(
                env_r, self._policy, pool)
        
        #### set up sampler with train env and actual policy (may be different from initial exploration policy)
        ######## note: sampler is set up with the pool that may be already filled from initial exploration hook
        self.sampler.initialize(env_r, policy, pool)
        
        #### some model inits
        if self._use_model:
            self.model_sampler.initialize(self.fake_env, policy, self.model_buf)
            rollout_dkl_lim = self.model_sampler.compute_dynamics_dkl(obs_batch=self._buffer.rand_batch_from_archive(5000, fields=['observations'])['observations'], depth=self._rollout_schedule[2])
            self.model_sampler.set_rollout_dkl(rollout_dkl_lim)
            self.initial_model_dkl = self.model_sampler.dyn_dkl
            self.approx_model_batch = self.batch_size_policy-self.init_real_samples    ### some size to start off

        #### reset gtimer (for coverage of project development)
        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)
        
        self.policy_epoch = 0       ### count policy updates
        self.new_real_samples = 0
        self.last_eval_step = 0
        self.diag_counter = 0
        running_diag = {}

        #### not implemented, could train policy before hook
        self._training_before_hook()

        #### iterate over epochs, gt.timed_for to create loop with gt timestamps
        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            
            #### do something at beginning of epoch (in this case reset self._train_steps_this_epoch=0)
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            samples_added = 0
            #=====================================================================#
            #            Rollout model                                            #
            #=====================================================================#
            model_samples = None
            keep_rolling = True
            metrics = {}
            #### start model rollout
            if self._use_model: 
                #=====================================================================#
                #                           Model Rollouts                            #
                #=====================================================================#
                if self.rollout_mode == 'schedule':
                    self._set_rollout_length()

                while keep_rolling:
                    ep_b = self._buffer.epoch_batch(batch_size=self._rollout_batch_size, epochs=self._buffer.epochs_list, fields=['observations','pi_infos'])
                    kls = np.clip(self._policy.compute_DKL(ep_b['observations'], ep_b['mu'], ep_b['log_std']), a_min=0, a_max=None)
                    btz_dist = self._buffer.boltz_dist(kls, alpha=self.sampling_alpha)
                    btz_b = self._buffer.distributed_batch_from_archive(self._rollout_batch_size, btz_dist, fields=['observations','pi_infos'])
                    start_states, mus, logstds = btz_b['observations'], btz_b['mu'], btz_b['log_std']
                    btz_kl = np.clip(self._policy.compute_DKL(start_states, mus, logstds), a_min=0, a_max=None)

                    self.model_sampler.reset(start_states)

                    for i in count():
                        # print(f'Model Sampling step Nr. {i+1}')

                        _,_,_,info = self.model_sampler.sample(max_samples=int(self.approx_model_batch-samples_added))

                        if self.model_sampler._total_samples + samples_added >= .99*self.approx_model_batch:
                            keep_rolling = False
                            break
                        
                        if info['alive_ratio']<= 0.1: break

                    ### diagnostics for rollout ###
                    rollout_diagnostics = self.model_sampler.finish_all_paths()

                    ### get model_samples, get() invokes the inverse variance rollouts ###
                    model_samples_new, buffer_diagnostics_new = self.model_buf.get()
                    model_samples = [np.concatenate((o,n), axis=0) for o,n in zip(model_samples, model_samples_new)] if model_samples else model_samples_new

                    ### diagnostics
                    new_n_samples = len(model_samples_new[0])+EPS
                    diag_weight_old = samples_added/(new_n_samples+samples_added)
                    diag_weight_new = new_n_samples/(new_n_samples+samples_added)
                    metrics = update_dict(metrics, rollout_diagnostics, weight_a= diag_weight_old,weight_b=diag_weight_new)
                    metrics = update_dict(metrics, buffer_diagnostics_new,  weight_a= diag_weight_old,weight_b=diag_weight_new)
                    ### run diagnostics on model data
                    if buffer_diagnostics_new['poolm_batch_size']>0:
                        model_data_diag = self._policy.run_diagnostics(model_samples_new)
                        model_data_diag = {k+'_m':v for k,v in model_data_diag.items()}
                        metrics = update_dict(metrics, model_data_diag, weight_a= diag_weight_old,weight_b=diag_weight_new)
                    
                    samples_added += new_n_samples
                    metrics.update({'samples_added':samples_added})
                
                ## for debugging
                metrics.update({'cached_var':np.mean(self._model.scaler_out.cached_var)})
                metrics.update({'cached_mu':np.mean(self._model.scaler_out.cached_mu)})

                print(f'Rollouts finished')
                gt.stamp('epoch_rollout_model')

            #=====================================================================#
            #  Sample                                                             #
            #=====================================================================#
            if self._use_model:
                n_real_samples = self.model_sampler.dyn_dkl/self.initial_model_dkl * self.init_real_samples
                n_real_samples = max(n_real_samples, self.min_real_samples)
            else:
                n_real_samples = self.batch_size_policy

            metrics.update({'n_real_samples':n_real_samples})
            start_samples = self.sampler._total_samples                     
            ### sample ###
            for i in count():
                #### _timestep is within an epoch
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                #### not implemented atm
                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                ##### śampling from the real world ! #####
                _,_, _, _ = self._do_sampling(timestep=self.policy_epoch)
                gt.stamp('sample')

                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

                if self.ready_to_train or self._timestep>n_real_samples:
                    self.sampler.finish_all_paths(append_val=True, append_cval=True, reset_path=False)
                    self.new_real_samples += self._timestep
                    break

            #=====================================================================#
            #  Train model                                                        #
            #=====================================================================#
            if self.new_real_samples>self._m_train_freq and self._use_model:
                model_diag = self.train_model(min_epochs=1, max_epochs=10)
                self.new_real_samples = 0
                metrics.update(model_diag)

            #=====================================================================#
            #  Get Buffer Data                                                    #
            #=====================================================================#
            real_samples, buf_diag = self._buffer.get()

            ### run diagnostics on real data
            policy_diag = self._policy.run_diagnostics(real_samples)
            policy_diag = {k+'_r':v for k,v in policy_diag.items()}
            metrics.update(policy_diag)
            metrics.update(buf_diag)


            #=====================================================================#
            #  Update Policy                                                      #
            #=====================================================================#
            train_samples = [np.concatenate((r,m), axis=0) for r,m in zip(real_samples, model_samples)] \
                if model_samples else real_samples
            self._policy.update_real_c(real_samples)
            self._policy.update_policy(train_samples)
            self._policy.update_critic(train_samples, train_vc=(train_samples[-3]>0).any())    ### @anyboby: only train vc if there are any costs?
            
            if self._use_model:
                self.approx_model_batch = self.batch_size_policy-n_real_samples 

            self.policy_epoch += 1
            #### log policy diagnostics
            self._policy.log()

            gt.stamp('train')
            #=====================================================================#
            #  Log performance and stats                                          #
            #=====================================================================#

            self.sampler.log()
            # write results to file, ray prints for us, so no need to print from logger
            logger_diagnostics = self.logger.dump_tabular(output_dir=self._log_dir, print_out=False)
            #=====================================================================#

            gt.stamp('epoch_after_hook')

            new_diagnostics = {}

            time_diagnostics = gt.get_times().stamps.itrs  

            # add diagnostics from logger
            new_diagnostics.update(logger_diagnostics) 

            new_diagnostics.update(OrderedDict((
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'model/{key}', metrics[key])
                    for key in sorted(metrics.keys())
                ),
            )))

            #### updateing and averaging
            old_ts_diag = running_diag.get('timestep', 0)
            new_ts_diag = self._total_timestep-self.diag_counter-old_ts_diag
            w_olddiag = old_ts_diag/(new_ts_diag+old_ts_diag)
            w_newdiag = new_ts_diag/(new_ts_diag+old_ts_diag)
            running_diag = update_dict(running_diag, new_diagnostics, weight_a=w_olddiag, weight_b=w_newdiag)
            running_diag.update({'timestep':new_ts_diag + old_ts_diag})
            ####
            
            if new_ts_diag + old_ts_diag > self.eval_every_n_steps:
                running_diag.update({
                    'epoch':self._epoch,
                    'timesteps_total':self._total_timestep,
                    'train-steps':self._num_train_steps,
                })
                self.diag_counter = self._total_timestep
                diag = running_diag.copy() 
                running_diag = {}
                yield diag

            if self._total_timestep >= self.n_env_interacts:
                self.sampler.terminate()

                self._training_after_hook()

                print("###### DONE ######")
                yield {'done': True, **running_diag}

                break


    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
        if self._n_initial_exploration_steps < 1: return

        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self.sampler.initialize(env, initial_exploration_policy, pool)
        while True:
            self.sampler.sample(timestep=0)
            if self.sampler._total_samples >= self._n_initial_exploration_steps:
                self.sampler.finish_all_paths(append_val=True, append_cval=True, reset_path=False)
                pool.get()  # moves policy samples to archive
                break
        
        ### train model
        if self._use_model:
            self.train_model(min_epochs=150, max_epochs=500)

    def train_model(self, min_epochs=5, max_epochs=100, batch_size=2048):
        print(f'[ MBPO ] log_dir: {self._log_dir}')
        print(f'[ MBPO ] Training model at epoch {self._epoch} | freq {self._m_train_freq} | \
             timestep {self._timestep} (total: {self._total_timestep}) (total train: {self._num_train_steps})')

        model_samples = self._buffer.get_archive(['observations',
                                                'actions',
                                                'next_observations',
                                                'rewards',
                                                'costs',
                                                'terminals',
                                                'epochs',
                                                ])

        dyn_ins, dyn_outs = format_samples_for_dyn(model_samples)

        diag_dyn = self._model.train(
            dyn_ins,
            dyn_outs, 
            batch_size=batch_size, #512
            max_epochs=max_epochs, # max_epochs 
            min_epoch_before_break=min_epochs, # min_epochs, 
            holdout_ratio=0.2,
            max_t=self._max_model_t
            )

        c_ins, c_outs = format_samples_for_cost(model_samples,
            one_hot=False)

        # if self.fake_env.learn_cost:
        #     diag_c = self.fake_env.train_cost_model(
        #         model_samples, 
        #         batch_size= batch_size, #batch_size, #512, 
        #         min_epoch_before_break= min_epochs,#min_epochs,
        #         max_epochs=max_epochs, # max_epochs, 
        #         holdout_ratio=0.2, 
        #         max_t=self._max_model_t
        #         )
        #     diag_dyn.update(diag_c)
        return diag_dyn


    @property
    def _total_timestep(self):
        total_timestep = self.sampler._total_samples
        return total_timestep

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        self.model_sampler.set_max_path_length(self._rollout_length)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

   
    def _do_sampling(self, timestep):
        return self.sampler.sample(timestep = timestep)
    
    def get_diagnostics(self,
                        iteration,
                        obs_batch = None,
                        training_paths = None,
                        evaluation_paths = None):
        """Return diagnostic information as ordered dictionary.

        Records state value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        """

        # @anyboby 
        warnings.warn('diagnostics not implemented yet!')

        diagnostics = {}

        return diagnostics

    def save(self, savedir):
        if self._use_model:
            self._model.save(savedir, self._epoch)

    @property
    def tf_saveables(self):
        saveables = {
            self._policy.tf_saveables
        }
        return saveables
