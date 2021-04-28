from copy import deepcopy

def get_cpobuffer(env, *args, **kwargs):
    from buffers.cpobuffer import CPOBuffer

    buffer = CPOBuffer(
        *args,
        observation_space=env.observation_space,
        action_space=env.action_space,
        *args,
        **kwargs)

    return buffer

BUFFER_FUNCTIONS = {
    'CPOBuffer': get_cpobuffer,
}

def get_buffer_from_params(params, env, *args, **kwargs):
    buffer_params = params['buffer_params']
    buffer_type = buffer_params['type']
    buffer_kwargs = deepcopy(buffer_params['kwargs'])

    buffer = BUFFER_FUNCTIONS[buffer_type](
        env,
        *args,
        **buffer_kwargs,
        **kwargs)

    return buffer
