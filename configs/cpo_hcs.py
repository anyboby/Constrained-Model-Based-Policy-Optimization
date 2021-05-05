

params = {
    'universe': 'gym',
    'task': 'HalfCheetahSafe-v2',
    'environment_params': {
        'normalize_actions': True,
    },
    'algorithm_params': {
        'type': 'CMBPO',
        'kwargs':{
            'n_env_interacts': int(10e6),
            'epoch_length': 50000, 
            'eval_every_n_steps': 5e3,
            'n_initial_exploration_steps': int(0), 
            'use_model': False,
            'batch_size_policy': 35000,
        }
    },
    'policy_params':{
        'type':'cpopolicy',
        'kwargs':{
            'constrain_cost':       True,
            'a_hidden_layer_sizes': (128, 128),
            'vf_lr':                3e-4,
            'vf_hidden_layer_sizes':(128,128),
            'vf_epochs':            8,
            'vf_batch_size':        2048,
            'vf_ensemble_size':     3,
            'vf_elites':            2,
            'vf_activation':        'swish',
            'vf_loss':              'MSE', 
            'vf_decay':             1e-6,
            'vf_clipping':          False, 
            'vf_kl_cliprange':      0.0,
            'ent_reg':              0, # 5e-3
            'target_kl':            0.01,
            'lam':                  .95,
            'gamma':                0.99,
        }
    },
    'buffer_params': {},
    'sampler_params': {
        'kwargs':{
            'render_mode':None,
        }
    },
    'run_params': {},
}