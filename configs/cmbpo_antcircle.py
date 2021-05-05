

params = {
    'universe': 'rllab',
    'task': 'AntCircle-v0',
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
            #### it is crucial to choose a model that doesn't overfit when trained too often on seen data
            ## for model architecture finding:  1. play around with the start samples to find an architecture, that doesn't really overfit
                                            # 2. m_train_freq in can somewhat limit overfitting, but is only treating the symptom
                                            # 3. try finding a balance between the size of new samples per number of
                                            #  updates of the model network (with m_train_freq)
            'use_model': True, 
            'm_hidden_dims':(512,512), # hidden layer size of model bnn
            'm_loss_type': 'MSPE',
            'm_use_scaler_in': True,
            'm_use_scaler_out': True,
            'm_lr': 1e-3,
            'm_train_freq': 4000,        # model is only trained every (self._timestep % self._model_train_freq==0) steps (terminates when stops improving)
            'rollout_batch_size': 1.0e3, # rollout_batch_size is the size of randomly chosen states to start from when rolling out model
            'm_networks': 7,             # size of model network ensemble
            'm_elites': 5,               # best networks to select from num_networks
            'max_model_t': None,         # a timeout for model training (e.g. for speeding up wallclock time)
            'sampling_alpha': 2,
            'rollout_mode' : 'uncertainty',           #### choose from 'schedule', or 'uncertainty'
            'rollout_schedule': [10, 500, 5, 30],       #[15, 100, 1, 15],    # min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
                                                        # increases rollout length from min_length to max_length over 
                                                        # range of (min_epoch, max_epoch)
                                                        ### Only applies if rollout_mode=='schedule'
            'maxroll': 35,              # maximum rollout horizon
            'batch_size_policy': 50000, # batch size before policy is updates
            'initial_real_samples_per_epoch': 20000,    # number of real samples contained in first batch
            'min_real_samples_per_epoch': 500, # absolute minimum of samples
        }
    },
    'policy_params':{
        'type':'cpopolicy',
        'kwargs':{
            'constrain_cost':       True,            # constrain_cost=False will perform TRPO updates
            'a_hidden_layer_sizes':   (128, 128),    # policy network hidden layers
            'vf_lr':                3e-4,            # learn rate for value learning
            'vf_hidden_layer_sizes':(128,128),       # nn hidden layers for vf
            'vf_epochs':            8,               # number of training epochs for values
            'vf_batch_size':        2048,            # minibatches for value training
            'vf_ensemble_size':     3,               # vf ensemble size
            'vf_elites':            2,               # vf elites 
            'vf_activation':        'swish',         # activation function
            'vf_loss':              'MSE',           # choose from 'NLL', 'MSPE' (inc. var); 'MSE' ; 'Huber'
            'vf_decay':             1e-6,            # decay for nn regularization
            'vf_clipping':          False,           # clip losses for a trust-region like vf update
            'vf_kl_cliprange':      0.0,                # only applicable if vf_clippping=True
            'ent_reg':              0, # 5e-3        # exploration bonus for maintaining pol. entropy
            'target_kl':            0.01,            # trust region diameter
            'cost_lim':             10,              
            'cost_lam':             .5,              # gae lambda
            'cost_gamma':           0.97,            # discounts
            'lam':                  .95,             # gae lambda
            'gamma':                0.99,            # discounts
        }
    },
    'buffer_params': {},
    'sampler_params': {
        'kwargs':{
            'render_mode':'human', #'human'
        }
    },
    'run_params': {},
}