import os
import copy
import glob
import pickle
import sys
import pdb

import tensorflow as tf
from ray import tune

from envs.utils import get_env_from_params
from algorithms.utils import get_algorithm_from_params
from policies.utils import get_policy_from_params
from buffers.utils import get_buffer_from_params
from samplers.utils import get_sampler_from_params

from utilities.utils import set_seed, initialize_tf_variables
from utilities.instrument import run_example_local, run_example_debug

class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        set_seed(variant['run_params']['seed'])

        self._variant = variant
        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(session)
        self._session = tf.keras.backend.get_session()

        self.train_generator = None
        self._built = False

    def _stop(self):
        tf.reset_default_graph()
        tf.keras.backend.clear_session()

    def _build(self):
        """
        called by tune to build algorithm 
        """
        variant = copy.deepcopy(self._variant)

        env_params = variant['environment_params']
        env = self.env = (
            get_env_from_params(env_params))

        buffer = self.buffer = (
            get_buffer_from_params(variant, env))
        sampler = self.sampler = get_sampler_from_params(variant)
        policy = self.policy = get_policy_from_params(
            variant, env, self._session)
        
        #### build algorithm 
        self.algorithm = get_algorithm_from_params(
            variant=self._variant,
            env=env,
            policy=policy,
            buffer=buffer,
            sampler=sampler,
            session=self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        # add graph since ray doesn't seem to automatically add that
        graph_writer = tf.summary.FileWriter(self.logdir, self._session.graph)
        graph_writer.flush()
        graph_writer.close()
        
        #### finalize graph
        tf.get_default_graph().finalize()
        self._built = True


    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_checkpoint(self):
        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        return tf_checkpoint

    def _save_replay_pool(self, checkpoint_dir):
        replay_pool_pickle_path = self._replay_pool_pickle_path(
            checkpoint_dir)
        self.buffer.save_latest_experience(replay_pool_pickle_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]
        for experience_path in experience_paths:
            self.buffer.load_experience(experience_path)

    def _save(self, checkpoint_dir):
        """Implements the saving logic.
        @anyboby: implementation very cmbpo specific saving methods, not optimal! 
            but general interfaces seem hard to implement due to all the different 
            frameworks (Keras, tf, pickling etc.)
        """

        ## only saves model atm
        self.policy_path = self.policy.save(checkpoint_dir)     ### @anyboby: this saves all tf objects
        self.algorithm.save(checkpoint_dir)

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._save_replay_pool(checkpoint_dir)

        return os.path.join(checkpoint_dir, '')
        
    def _restore(self, checkpoint_dir):
        raise NotImplementedError

def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    # __package__ should be `development.main`
    run_example_local(__package__, argv)
    #run_example_debug(__package__, argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])