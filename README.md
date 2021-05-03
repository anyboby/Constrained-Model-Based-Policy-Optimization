# Constrained Model-Based Policy Optimization

<p align="center">
	<!-- <img src="https://drive.google.com/uc?export=view&id=1DcXi5wY_anmtlNeIErl1ECgKGsGi4oR1" width="80%"> -->
	<img src="https://drive.google.com/uc?export=view&id=1DcXi5wY_anmtlNeIErl1ECgKGsGi4oR1" width="80%">
</p>

This repository contains code for Constrained Model-Based Policy Optimization (CMBPO), a model-based version of Constrained Policy Optimization (Achiam et al.). Installation, execution and code examples for the reproduction of the experiments described in [Safe Continuous Control with Constrained Model-Based Policy Optimization](https://arxiv.org/abs/2104.06922?context=cs) are provided below. 

# Prerequisites

1. The simulation experiments using [mujoco-Py](https://github.com/openai/mujoco-py) require a working install of [MuJoCo 2.0](https://www.roboti.us/license.html) and a valid license. 
2. We use conda environments for installs (tested on conda 4.6 - 4.10), please refer to [Anaconda](https://docs.anaconda.com/anaconda/install/) for instructions. 

# Installation

1. Clone this repository
```
git clone https://github.com/anyboby/Constrained-Model-Based-Policy-Optimization.git
```
2. Create a conda environment using the cmbpo yml-file
```sh
cd Constrained-Model-Based-Policy-Optimization/
conda env create -f cmbpo.yml
conda activate cmbpo
pip install -e .
```
This should create a conda environment labeled 'cmbpo' with the necessary packages and modules. The number of required modules is limited, so it is worth taking a look at the [cmbpo.yml](cmbpo.yml) and [requirements.txt](requirements.txt) files in case of troubles with the installs.

# Usage
To start an experiment with cmbpo, run 
```sh
cmbpo run_local configs.baseconfig --config=configs.cmbpo_hcs --gpus=1 --trial-gpus=1
```

`-- config` specifies the configuration file for experiment (here: CMBPO for HalfCheetahSafe)\
`-- gpus` specifies the number of gpus to use

A list of all available flags is provided in [baseconfig/utils](configs/baseconfig/utils.py). As of writing,only local running is supported. For further options, refer to the ray documentation.

The `cmbpo` command uses the [console scripts](scripts/console_scripts.py) as an entry point for running experiments. A simple workflow of running experiments with ray-tune is illustrated in [run.py](scripts/run.py).

## Algorithms
Constrained Model-Based Policy Optimization aims at combining Constrained Policy Optimization with model-based data augmentation and reconciling constraint satisfaction with the entailed model-errors. 

This repository can therefore also be used to run experiments with model-free versions of Constrained Policy Optimization and Trust-Region Policy Optimization by configuring the `use_model` and `constrain_cost` flags accordingly in the experiment configurations (see [CPO - HalfCheetahSafe](configs/cpo_hcs.py) and [TRPO - HalfCheetahSafe](configs/trpo_hcs.py)):
```py
'use_model': 		False,	# set to True for model-based
'constrain_cost':   False,  # set to True for cost-constrained optimziation
```

## Adding new environments and running custom experiments
Different environments can be tested by creating a config file in the [configs](configs/) directory. OpenAi gym environments can be loaded directly with the corresponding parameters, for example:
```py
'universe': 'gym',
'task':     'HalfCheetahSafe-v2',
```
Environments from other sources require an entry in the `ENVS_FUNCTIONS` dict in the [environment utils](envs/utils.py) that specifies how to create an instance of the environment. For example, the Gym environments are specified with the following entries: 
```py
def get_gym_env():
    import gym
    
    return gym.make

ENVS_FUNCTIONS = {
    'gym':get_gym_env()
}
```

## Model-Learning with custom environments
When using a model with custom environments, the model requires a few interfaces to function with the provided code. The [base model](models/base_model.py) should be inherited by a learned (or handcrafted) model and specify whether rewards, costs, and termination functions are predicted alongside the dynamics. 

By default our algorithm learns to predict rewards but assumes handcrafted cost- and termination-functions `c(s,a,s')` and `t(s,a,s')`. When adding a new environment, these functions should be defined (if not provided by the model) in the [statics](models/statics.py) file. For example, a default termination function that continues episodes for all states looks like this:
```py
def no_done(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

    done = np.zeros(shape=obs.shape[:-1], dtype=np.bool) #always false
    done = done[...,None]
    return done
```
The static functions should then be linked by the environments' task name, such that the [Fake Environment](models/fake_env.py) correctly discovers them:
```py
TERMS_BY_TASK = {
    'default':no_done,
    'HalfCheetah-v2':no_done,
}
```

## Hyperparameters
Hyperparameters for a new experiment can be defined in the [configs](configs/) folder. The general form of our config files follows the following structure:
```py
params = {
    'universe': 'gym',
    'task': 'HalfCheetahSafe-v2',
    'algorithm_params': {...},
    'policy_params':{...},
    'buffer_params': {...},
    'sampler_params': {...},
    'run_params': {...},
}
```
Parameters specified in a config file overwrite the [base config](configs/baseconfig/base.py) file. For new algorithms or a new suite of environments, it might be practical to directly change the base config. 

In addition to model-parameters and policy-parameters, the main parameters of concern in CMPBO define rollout- and sampling-behavior of the algorithm. 
```py
'n_initial_exploration_steps': int(10000), ### number of initial exploration steps for model-learning and 
                                            # determining uncertainty calibration measurements
'sampling_alpha': 2,                    ### temperature for boltzman-sampling
'rollout_mode' : 'uncertainty',         ### model rollouts terminate based on per-step uncertainty
'rollout_schedule': [10, 500, 5, 30],   ### if rollout_mode:'schedule' this schedule is defined as 
                                                # [min_epoch, max_epoch, min_horizon, max_horizon]
                                            ## if rollout_mode:'uncertainty', 'min_horizon' is used as 
                                            # the initial rollout horizon and adapted throughout 
                                            # training based on per-step uncertainty estimates 
                                            # (KL-Divergence).
'batch_size_policy': 50000,             ### batch size per policy update
'initial_real_samples_per_epoch': 1500, ### initial number of real samples per policy update, 
                                            # adapted   throughout training based on average uncertainty 
                                            # estimates (mean KL-Divergence).
'min_real_samples_per_epoch': 500,      ### absolute minimum number of real samples per policy update
```
## Logging
A range of measurements is logged automatically in tensorboard, the parameter configuration is saved as a JSON file. The location for summaries and checkpoints can be defined by specifying a `'log_dir'` in the configuration files. By default, this location will be set to `'~/ray_cmbpo/{env-taks}/defaults/{seed}'` and can be accessed with tensorboard by
```sh
tensorboard --logdir ~/ray_cmbpo/<env>/defaults/<seed_dir>
```

# Acknowledgments
Several sections of this repository contain code from other repositories, notably from [Tuomas Haarnoja](https://scholar.google.com/citations?user=VT7peyEAAAAJ&hl=en), [Kristian Hartikainen's](https://github.com/rail-berkeley/softlearning), [Michael Janner](https://github.com/JannerM/mbpo), [Kurtland Chua](https://github.com/kchua/handful-of-trials), and CPO by [Joshua Achiam and Alex Ray](https://github.com/openai/safety-starter-agents). 
