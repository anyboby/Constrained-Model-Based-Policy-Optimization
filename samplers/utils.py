from copy import deepcopy

import numpy as np

def get_cposampler(*args, **kwargs):
    from samplers.cpo_sampler import CpoSampler
    sampler = CpoSampler(
        *args,
        **kwargs)

    return sampler

SAMPLERS_FUNCTIONS = {
        'CPOSampler' : get_cposampler,
    }


def get_sampler_from_params(params, *args, **kwargs):

    sampler_params = params['sampler_params']
    sampler_type = sampler_params['type']

    sampler_args = deepcopy(sampler_params.get('args', ()))
    sampler_kwargs = deepcopy(sampler_params.get('kwargs', {}))

    sampler = SAMPLERS_FUNCTIONS[sampler_type](
        *sampler_args, *args, **sampler_kwargs, **kwargs)

    return sampler