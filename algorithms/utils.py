from copy import deepcopy
from dotmap import DotMap
from collections import OrderedDict      

def create_CMBPO_algorithm(variant, *args, **kwargs):
    from algorithms.cmbpo import CMBPO
    algorithm = CMBPO(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'CMBPO': create_CMBPO_algorithm,
}


def get_algorithm_from_params(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    # @anyboby, workaround for local_example_debug mode, for some reason gets DotMap isntead of 
    # OrderedDict as algorithm_kwargs, which doesn't seem to work for double asteriks !
    if isinstance(algorithm_kwargs, DotMap): 
        algorithm_kwargs = algorithm_kwargs.toDict()

    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
