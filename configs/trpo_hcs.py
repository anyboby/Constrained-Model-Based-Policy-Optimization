

params = {
    'universe': 'gym',
    'task': 'HalfCheetahSafe-v2',
    'algorithm_params': {
        'type': 'CMBPO',
        'kwargs':{
            'n_env_interacts': int(10e6),
            'epoch_length': 50000, #1000,    # samples per epoch, also determines train frequency 
            'eval_render_mode': 'human',    # 
            'eval_n_episodes': 1,
            'eval_every_n_steps': 5e3,
            'eval_deterministic': False,    # not implemented in cmbpo
            'n_initial_exploration_steps': int(0), #5000
            #### it is crucial to choose a model that doesn't overfit when trained too often on seen data
            ## for model architecture finding:  1. play around with the start samples to find an architecture, that doesn't really overfit
                                            # 2. _epochs_since_update in the bnn can somewhat limit overfitting, but is only treating the symptom
                                            # 3. try finding a balance between the size of new samples per number of
                                            #  updates of the model network (with model_train_freq)
            'use_model': False,
            'batch_size_policy': 5000,              ### how many samples 
        }
    },
    'policy_params':{
        'type':'cpopolicy',
        'kwargs':{
            'constrain_cost':       False,
            'a_hidden_layer_sizes': (128, 128),
            'vf_lr':                3e-4,
            'vf_hidden_layer_sizes':(128,128),
            'vf_epochs':            8,
            'vf_batch_size':        2048,
            'vf_ensemble_size':     3,
            'vf_elites':            2,
            'vf_activation':        'swish',
            'vf_loss':              'MSE',           # choose from #'NLL' (inc. var); 'MSE' ; 'Huber'
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