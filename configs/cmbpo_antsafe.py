

params = {
    'universe': 'gym',
    'task': 'AntSafe-v2',
    'algorithm_params': {
        'type': 'CMBPO',
        'kwargs':{
            'n_env_interacts': int(10e6),
            'epoch_length': 50000, #1000,    # samples per epoch, also determines train frequency 
            'eval_every_n_steps': 5e3,
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
            'initial_real_samples_per_epoch': 20000,
            'min_real_samples_per_epoch': 500,
        }
    },
    'policy_params':{
        'type':'cpopolicy',
        'kwargs':{
            'constrain_cost':       True,
            'a_hidden_layer_sizes':   (128, 128),
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