from ray import tune
import numpy as np
import pdb

from utilities.utils import deep_update

M = 256 #256

NUM_COUPLING_LAYERS = 2

DEFAULT_MAX_PATH_LENGTH = 1000


CPO_POLICY_PARAMS_BASE = {
    'type': 'CPOPolicy',
    'kwargs': {
        'a_hidden_layer_sizes':   (M, M),
        'dyn_ensemble_size':    tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['algorithm_params']['kwargs']['m_networks']
            )),
        'constrain_cost':       True,
        'vf_lr':                3e-4,
        'vf_hidden_layer_sizes':(M,M),
        'vf_epochs':            8,                 
        'vf_batch_size':        2048,
        'vf_ensemble_size':     3,
        'vf_elites':            2,
        'vf_activation':        'swish',
        'vf_loss':              'MSE',          # choose from #'NLL' (inc. var); 'MSE' ; 'Huber'
        'vf_decay':             1e-6,
        'vf_clipping':          False,           # clip losses for a trust-region like update
        'vf_kl_cliprange':      0.0,
        'ent_reg':              0, # 5e-3
        'target_kl':            0.01,
        'cost_lim':             10,
        'cost_lam':             .5,
        'cost_gamma':           0.97,
        'lam':                  .95,
        'gamma':                0.99,
        'epoch_length': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['algorithm_params']['kwargs']['epoch_length'] 
            )),
        'max_path_length': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['sampler_params']['kwargs']['max_path_length']
            )),
        'log_dir': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['log_dir']
            )),
    }
}

POLICY_PARAMS_BASE = {
    'CPOPolicy' :   CPO_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'cpopolicy': POLICY_PARAMS_BASE['CPOPolicy']
})

ALGORITHM_PARAMS = {
    'CMBPO': {
        'type': 'CMBPO',
        'kwargs': {
            'task': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['environment_params']['task']
            )),
            'n_env_interacts': int(10e6),
            'epoch_length': 50000, #1000,    # samples per epoch, also determines train frequency 
            'eval_render_mode': 'human',    
            'eval_n_episodes': 1,
            'eval_every_n_steps': 5e3,
            'eval_deterministic': False,    # not implemented in cmbpo
            'n_initial_exploration_steps': int(10000), #5000

            #### it is crucial to choose a model that doesn't overfit when trained too often on seen data
            ## for model architecture finding:  1. play around with the start samples to find an architecture, that doesn't really overfit
                                            # 2. _epochs_since_update in the bnn can somewhat limit overfitting, but is only treating the symptom
                                            # 3. try finding a balance between the size of new samples per number of
                                            #  updates of the model network (with model_train_freq)

            'use_model': True, 
            'm_hidden_dims':(512,512), # hidden layer size of model bnn
            'm_loss_type': 'MSPE',
            'm_use_scaler_in': True,
            'm_use_scaler_out': True,
            'm_lr': 1e-3,
            'm_train_freq': 4000,        # model is only trained every (self._timestep % self._model_train_freq==0) steps (terminates when stops improving)
            'rollout_batch_size': 1.0e3,    # rollout_batch_size is the size of randomly chosen states to start from when rolling out model
            'm_networks': 7,              # size of model network ensemble
            'm_elites': 5,                # best networks to select from num_networks
            'max_model_t': None,            # a timeout for model training (e.g. for speeding up wallclock time)
            'sampling_alpha': 2,
            'rollout_mode' : 'uncertainty',           #### choose from 'schedule', or 'uncertainty'
            'rollout_schedule': [10, 500, 5, 30], #[15, 100, 1, 15],    # min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
                                                        # increases rollout length from min_length to max_length over 
                                                        # range of (min_epoch, max_epoch)
                                                        ### Only applies if rollout_mode=='schedule'
            'maxroll': 35,      ### only really relevant for iv gae
            'batch_size_policy': 50000,              ### how many samples 
            'initial_real_samples_per_epoch': 15000,
            'min_real_samples_per_epoch': 500,
        }
    },
}


BUFFER_PARAMS_PER_ALGO = {
    'CMBPO': {
        'type': 'CPOBuffer',
        'preprocess_type': 'default',
        'kwargs': {
            'size': tune.sample_from(lambda spec: (
               spec.get('config', spec)
               ['algorithm_params']['kwargs']['epoch_length'] 
            )),
            'archive_size': tune.sample_from(lambda spec: (
                {
                    'SimpleReplayPool': int(1e6),
                    'CPOBuffer':int(3e5),
                }.get(
                    spec.get('config', spec)
                    ['buffer_params']['type'],
                    int(1e6))
            )),
        }
    },
}

SAMPLER_PARAMS_PER_ALGO = {
    'default': {
        'type':'CPOSampler',
        'kwargs':{
            'max_path_length': DEFAULT_MAX_PATH_LENGTH,
            'render_mode': None,
        },
    },
    'CMBPO': {
        'type':'CPOSampler',
        'kwargs':{
            'max_path_length': DEFAULT_MAX_PATH_LENGTH,
            'render_mode': None,
        },
    }
}

RUN_PARAMS = {
    'seed': tune.sample_from(
        lambda spec: np.random.randint(0, 10000)),
    'checkpoint_at_end': True,
    'checkpoint_frequency': 50,
    'checkpoint_buffer': False,
}

def get_variant_spec(args, params):
    assert hasattr(params, 'universe') and \
        hasattr(params, 'task') and \
        hasattr(params, 'algorithm') and \
        hasattr(params, 'policy') 

    universe, task = params.universe, params.task
    algorithm, policy = params.algorithm_params.type, params.policy_params.type
    base_spec = {
        'log_dir': f'~/ray_{algorithm.lower()}',
        'exp_name': 'defaults',
        'environment_params': {
            'universe': universe,
            'task': task,
            'kwargs':{},
        },
        'policy_params': POLICY_PARAMS_BASE[policy],
        'algorithm_params': ALGORITHM_PARAMS[algorithm],
        'buffer_params': BUFFER_PARAMS_PER_ALGO[algorithm],
        'sampler_params': SAMPLER_PARAMS_PER_ALGO[algorithm],
        'run_params': RUN_PARAMS,
    }

    variant_spec = deep_update(
        base_spec,
        params
    )
    return variant_spec
