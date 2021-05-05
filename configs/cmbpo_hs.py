

params = {
    'universe': 'gym',
    'task': 'HumanoidSafe-v2',
    'environment_params': {
        'normalize_actions': True,
    },
    'algorithm_params': {
        'type': 'CMBPO',
        'kwargs':{
            'n_env_interacts': int(10e6),
            'epoch_length': 50000, 
            'eval_every_n_steps': 5e3,
            'n_initial_exploration_steps': int(10000), 
            'use_model': True, 
            'm_hidden_dims':(512,512), 
            'm_loss_type': 'MSPE',
            'm_use_scaler_in': True,
            'm_use_scaler_out': True,
            'm_lr': 1e-3,
            'm_train_freq': 4000,        
            'rollout_batch_size': 1.0e3, 
            'm_networks': 7,             
            'm_elites': 5,               
            'max_model_t': None,         
            'sampling_alpha': 2,
            'rollout_mode' : 'uncertainty',      
            'rollout_schedule': [10, 500, 5, 30],
            'maxroll': 35,
            'batch_size_policy': 50000,
            'initial_real_samples_per_epoch': 15000,
            'min_real_samples_per_epoch': 500,
        }
    },
    'policy_params':{
        'type':'cpopolicy',
        'kwargs':{
            'constrain_cost':       False,
            'a_hidden_layer_sizes':   (128, 128),
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
            'render_mode':'human',
        }
    },
    'run_params': {},
}