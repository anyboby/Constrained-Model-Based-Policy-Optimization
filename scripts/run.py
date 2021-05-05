import os
import copy
import glob
import pickle
import sys
import pdb
import importlib
from dotmap import DotMap

import tensorflow as tf
import ray
from ray import tune
from ray.autoscaler.commands import exec_cluster

from envs.utils import get_env_from_params
from algorithms.utils import get_algorithm_from_params
from policies.utils import get_policy_from_params
from buffers.utils import get_buffer_from_params
from samplers.utils import get_sampler_from_params
from utilities.utils import set_seed, initialize_tf_variables
from utilities.instrument import create_trial_name_creator

class SimpleExperiment(tune.Trainable):
    def _setup(self, params):
        self._params = params
        
        #### set up tf session
        set_seed(params['run_params']['seed'])
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

        #### set up building blocks for algorithm
        params = copy.deepcopy(self._params)
        env_params = params['environment_params']
        env = self.env = (
            get_env_from_params(env_params))

        buffer = self.buffer = (
            get_buffer_from_params(params, env))

        sampler = self.sampler = get_sampler_from_params(params)

        policy = self.policy = get_policy_from_params(
            params, env, self._session)

        #### build algorithm 
        self.algorithm = get_algorithm_from_params(
            variant=self._params,
            env=env,
            policy=policy,
            buffer=buffer,
            sampler=sampler,
            session=self._session)

        #### finalize graph
        initialize_tf_variables(self._session, only_uninitialized=True)
        tf.get_default_graph().finalize()

        #### set train generator function
        self.train_generator = self.algorithm.train()
        self._built = True
        
    def _train(self):
        if not self._built:
            self._build()

        diagnostics = next(self.train_generator)
        return diagnostics

def main(argv=None):
    """
    run simple ray tune experiment.

    Please provide config file location, e.g.

    <python run.py configs.cmbpo_hcs>
    """
    assert argv[0] is not None, "Please provide config file location, e.g."

    #### create
    base_module = 'configs.baseconfig'
    base_module = importlib.import_module(base_module)

    #### tune configs
    trial_name_template = 'seed:{trial.config[run_params][seed]}'
    trial_name_creator = create_trial_name_creator(trial_name_template) ## generator for trial name (determines logdir)
    gpus=1      ## gpus to be used
    trial_gpus=1    ## gpus to be used in trial
    mode='local'    ## local or remote, currently only local supported

    config=str(argv[0])  ## config file location

    exp_config = DotMap(dict(
        gpus=gpus,
        trial_gpus=trial_gpus,
        mode=mode,
        config=config,
    ))

    ### build the experiment
    exp_config = base_module.get_variant_spec(exp_config)   ## merge base config and config file to final config
    exp_id = exp_config.get('exp_name')     ## name of the experiment
    exp_class = SimpleExperiment    ## tune trainable class that runs the experiments
    local_dir = os.path.join(exp_config.get('log_dir'), exp_config.get('task'))     ## directory for tf summaries, configs etc.

    ### define experiment
    experiment = {
        exp_id:{
        'run': exp_class,
        'config': exp_config,
        'local_dir': local_dir,
        'trial_name_creator': trial_name_creator,
        }
    }

    ### initialize ray und run experiments
    ray.init(
        num_gpus=gpus,
        local_mode=True,
        object_store_memory=100 * 1024 * 1024,  #@anyboby TODO: test the memory config 
        )

    tune.run_experiments(
        experiment,
        server_port=4321,
    )

if __name__ == '__main__':
    main(argv=sys.argv[1:])