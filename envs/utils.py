import gym
import envs.mujoco_safety_gym

def get_gym_env():
    import gym
    import envs.mujoco_safety_gym
    
    return gym.make

def get_safety_gym():  
    import safety_gym

    return gym.make

ENVS_FUNCTIONS = {
    'gym':get_gym_env()
}

def get_environment(universe, task, environment_kwargs):
    env = ENVS_FUNCTIONS[universe](task, **environment_kwargs)
    return env

def get_env_from_params(env_params):
    universe = env_params['universe']
    task = env_params['task']
    environment_kwargs = env_params.get('kwargs', {}).copy()

    return get_environment(universe, task, environment_kwargs)
