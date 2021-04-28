import numpy as np

def no_done(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

    done = np.zeros(shape=obs.shape[:-1], dtype=np.bool) #np.array([False]).repeat(len(obs))
    done = done[...,None]
    return done
    
def hcs_cost_f(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

    xdist = next_obs[...,-1]*10
    obj_cost = np.array((np.abs(xdist)<2.0), dtype=np.float32)[..., None]
    return obj_cost

def antsafe_term_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

    z = next_obs[..., 0]
    body_quat = next_obs[...,1:5]
    z_rot = 1-2*(body_quat[...,1]**2+body_quat[...,2]**2)

    notdone = np.isfinite(next_obs).all(axis=-1) \
        * (z >= 0.2) \
        * (z <= 1.0) \
        * z_rot >= -0.7

    done = ~notdone
    done = done[...,None]
    return done

def antsafe_c_fn(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape)

    z = next_obs[..., 0]
    body_quat = next_obs[...,1:5]
    z_rot = 1-2*(body_quat[...,1]**2+body_quat[...,2]**2)
    y_dist = next_obs[..., -1:]
    
    obj_cost = np.any(abs(y_dist)>3.2, axis=-1)[...,None]*1.0

    notdone = np.isfinite(next_obs).all(axis=-1) \
        * (z >= 0.2) \
        * (z <= 1.0) \
        * z_rot >= -0.7

    done = ~notdone
    done = done[...,None]

    done_cost = done*1.0
    cost = np.clip(done_cost+obj_cost, 0, 1)
    return cost


TERMS_BY_TASK = {
    'default':no_done,
    'HalfCheetah-v2':no_done,
    'HalfCheetahSafe-v2':no_done,
    'AntSafe-v2':antsafe_term_fn,
}

REWS_BY_TASK = {
    
}

COST_BY_TASK = {
    'HalfCheetahSafe-v2':hcs_cost_f,
    'AntSafe-v2':antsafe_c_fn,
}