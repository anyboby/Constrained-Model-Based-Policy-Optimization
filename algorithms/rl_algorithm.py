import abc
from collections import OrderedDict
from itertools import count
import gtimer as gt
import math
import os
import pdb

import tensorflow as tf
import numpy as np

from utilities.utils import save_video


class RLAlgorithm(tf.contrib.checkpoint.Checkpointable):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=int(10e7),
            n_initial_exploration_steps=0,
            initial_exploration_policy=None,
            epoch_length=1000,
            eval_n_episodes=10,
            eval_deterministic=True,
            eval_render_mode=None,
            video_save_frequency=0,
            session=None,
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_initial_exploration_steps: Number of steps in the beginning to
                take using actions drawn from a separate exploration policy.
            initial_exploration_policy: policy to follow during initial 
                exploration hook
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render_mode (`str`): Mode to render evaluation rollouts in.
                None to disable rendering.
        """
        self.sampler = sampler

        self._n_epochs = n_epochs
        self._epoch_length = epoch_length
        self._n_initial_exploration_steps = n_initial_exploration_steps
        self._initial_exploration_policy = initial_exploration_policy

        self._eval_n_episodes = eval_n_episodes
        self._eval_deterministic = eval_deterministic
        self._video_save_frequency = video_save_frequency

        if self._video_save_frequency > 0:
            assert eval_render_mode != 'human', (
                "RlAlgorithm cannot render and save videos at the same time")
            self._eval_render_mode = 'rgb_array'
        else:
            self._eval_render_mode = eval_render_mode

        self._session = session or tf.keras.backend.get_session()

        self._epoch = 0
        self._timestep = 0
        self._num_train_steps = 0

    def _initial_exploration_hook(self, env, initial_exploration_policy, pool):
        if self._n_initial_exploration_steps < 1: return

        if not initial_exploration_policy:
            raise ValueError(
                "Initial exploration policy must be provided when"
                " n_initial_exploration_steps > 0.")

        self.sampler.initialize(env, initial_exploration_policy, pool)
        while pool.size < self._n_initial_exploration_steps:
            self.sampler.sample()   

    def _training_before_hook(self):
        """Method called before the actual training loops."""
        pass

    def _training_after_hook(self):
        """Method called after the actual training loops."""
        pass

    def _timestep_before_hook(self, *args, **kwargs):
        """Hook called at the beginning of each timestep."""
        pass

    def _timestep_after_hook(self, *args, **kwargs):
        """Hook called at the end of each timestep."""
        pass

    def _epoch_before_hook(self):
        """Hook called at the beginning of each epoch."""
        self._train_steps_this_epoch = 0

    def _epoch_after_hook(self, *args, **kwargs):
        """Hook called at the end of each epoch."""
        pass

    def _training_batch(self, batch_size=None):
        return self.sampler.random_batch(batch_size)

    def _evaluation_batch(self, *args, **kwargs):
        return self._training_batch(*args, **kwargs)

    @property
    def _training_started(self):
        return self._total_timestep > 0

    @property
    def _total_timestep(self):
        total_timestep = self._epoch * self._epoch_length + self._timestep
        return total_timestep

    def _train(self):
        """Return a generator that performs RL training.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_diagnostics(self,):
        raise NotImplementedError

    @property
    def ready_to_train(self):
        return self.sampler.batch_ready()

    def _do_sampling(self, timestep):
        return self.sampler.sample()

    @property
    def tf_saveables(self):
        return {}

    def __getstate__(self):
        state = {
            '_epoch_length': self._epoch_length,
            '_epoch': (
                self._epoch + int(self._timestep >= self._epoch_length)),
            '_timestep': self._timestep % self._epoch_length,
        }

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
