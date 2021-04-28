from copy import deepcopy

def get_cpo_policy(env, session, *args, **kwargs):
    from policies.cpo_policy import CPOPolicy
    policy = CPOPolicy(
        obs_space=env.observation_space,
        act_space=env.action_space,
        session=session,
        *args,
        **kwargs)
    return policy

POLICY_FUNCTIONS = {
    'cpopolicy': get_cpo_policy,
}


def get_policy(policy_type, *args, **kwargs):
    return POLICY_FUNCTIONS[policy_type](*args, **kwargs)

def get_policy_from_params(params, env, *args, **kwargs):
    policy_params = params['policy_params']
    policy_type = policy_params['type']
    policy_kwargs = deepcopy(policy_params['kwargs'])

    policy = POLICY_FUNCTIONS[policy_type](
        env,
        *args,
        **policy_kwargs,
        **kwargs)

    return policy
