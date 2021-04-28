import numpy as np
from utilities.mpi_tools import mpi_statistics_scalar
from utilities.utils import (combined_shape,
                                EPS,
                                keys_as_sorted_list,
                                values_as_sorted_list,
                                discount_cumsum,
                                )

from utilities.utils import (discount_cumsum_weighted,  ## imports for for iv advantages
                                triu_indices_t_h,
                                disc_cumsum_matrix,
                                )

from buffers.cpobuffer import CPOBuffer
import scipy.signal

class ModelBuffer(CPOBuffer):

    def __init__(self, batch_size, env, max_path_length,
                 *args,
                 **kwargs,
                 ):
        
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.env = env
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape
        self.pi_info_shapes = None
        self.reset()

    ''' initialize policy dependendant pi_info shapes, gamma, lam etc.'''
    def initialize(self, pi_info_shapes,
                    gamma=0.99, lam = 0.95,
                    cost_gamma = 0.99, cost_lam = 0.95,
                    ):
        self.pi_info_shapes = pi_info_shapes
        self.pi_info_bufs = {k: np.zeros(shape=[self.batch_size]+[self.max_path_length] + list(v), dtype=np.float32) 
                            for k,v in pi_info_shapes.items()}
        self.sorted_pi_info_keys = keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam

    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        obs_buf_shape = combined_shape(self.batch_size, combined_shape(self.max_path_length, self.obs_shape))
        act_buf_shape = combined_shape(self.batch_size, combined_shape(self.max_path_length, self.act_shape))
        scalar_shape = (self.batch_size, self.max_path_length)

        self.obs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.act_buf = np.zeros(act_buf_shape, dtype=np.float32)

        self.dyn_error_buf = np.zeros(scalar_shape, dtype=np.float32)

        self.nextobs_buf = np.zeros(obs_buf_shape, dtype=np.float32)
        self.adv_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.roll_lengths_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.rew_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.ret_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.val_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.cadv_buf = np.zeros(scalar_shape, dtype=np.float32)    
        self.croll_lengths_buf = np.zeros(scalar_shape, dtype=np.float32)   
        self.cost_buf = np.zeros(scalar_shape, dtype=np.float32)    
        self.cret_buf = np.zeros(scalar_shape, dtype=np.float32)    
        self.cval_buf = np.zeros(scalar_shape, dtype=np.float32)    
        self.logp_buf = np.zeros(scalar_shape, dtype=np.float32)
        self.term_buf = np.zeros(scalar_shape, dtype=np.bool_)
        if self.pi_info_shapes:
            self.pi_info_bufs = {k: np.zeros(shape=[self.batch_size]+[self.max_path_length] + list(v), dtype=np.float32) 
                                for k,v in self.pi_info_shapes.items()}

        self.cutoff_horizons_mean = 0
        # ptr is a scalar to the current position in all paths. You are expected to store at the same timestep 
        #   in all parallel paths
        # path_start_idx is the path starting index, which will actually always be 0, since paths are parallel
        #   and always start at 0, may be removed 
        # max_size is actually also the same for all parallel paths, but a batch sized vector is more convenient
        #   for masked assertion
        # populated_mask shows us which entries in the buffer are valid, meaning they had a value stored in them
        #   and aren't terminated.
        # terminated_paths_mask essentially notes the same thing as populated_mask but is one_dimensional for 
        #   convenience

        self.ptr, self.path_start_idx, self.max_size, self.populated_mask, self.populated_indices, self.terminated_paths_mask = \
                                                            0, \
                                                            0, \
                                                            np.ones(self.batch_size)*self.max_path_length, \
                                                            np.zeros((self.batch_size, self.max_path_length), dtype=np.bool), \
                                                            np.repeat(np.arange(self.max_path_length)[None], axis=0, repeats=self.batch_size), \
                                                            np.zeros(self.batch_size, dtype=np.bool)

    @property
    def size(self):
        return self.populated_mask.sum()

    @property
    def has_room(self):
        room_mask = self.ptr < self.max_size

        return room_mask.all()

    @property
    def alive_paths(self):
        return np.logical_not(self.terminated_paths_mask)

    def store_multiple(self, obs, act, next_obs, rew, val, cost, cval, dyn_error, logp, pi_info, term):
        assert (self.ptr < self.max_size).all()
        alive_paths = self.alive_paths
        
        self.obs_buf[alive_paths, self.ptr] = obs
        self.act_buf[alive_paths, self.ptr] = act
        self.nextobs_buf[alive_paths, self.ptr] = next_obs
        self.rew_buf[alive_paths, self.ptr] = rew
        self.val_buf[alive_paths, self.ptr] = val
        self.cost_buf[alive_paths, self.ptr] = cost
        self.cval_buf[alive_paths, self.ptr] = cval
        self.logp_buf[alive_paths, self.ptr] = logp
        self.term_buf[alive_paths, self.ptr] = term

        for k in self.sorted_pi_info_keys:
            self.pi_info_bufs[k][alive_paths, self.ptr] = pi_info[k]

        self.dyn_error_buf[alive_paths, self.ptr] = dyn_error
        self.populated_mask[alive_paths, self.ptr] = True

        
        self.ptr += 1


    def finish_path_multiple(self, term_mask, last_val=0, last_cval=0):
        """
        finishes multiple paths according to term_mask. 
        Note: if the term_mask indicates to terminate a path that has not yet been populated,
        it will terminate, but samples won't be marked as terminated (they won't be included 
        in get())
        Args:
            term_mask: a bool mask that indicates which paths should be terminated. 
                has to be of same length as currently alive paths.
            last_val: value of the last state in the paths that are to be finished.
                has to be of same length as the number of paths to be terminated (term_mask.sum())
            last_cval: cost value of the last state in the paths that are to be finished.
                has to be of same length as the number of paths to be terminated (term_mask.sum())
        """
        if not term_mask.any(): return                    ### skip if not terminating anything
        assert self.alive_paths.sum() == len(term_mask)   ### terminating a non-alive path!
        alive_paths = self.alive_paths

        ## concat masks for fancy indexing. (expand term_mask to buf dim)
        finish_mask = np.zeros(len(self.alive_paths), dtype=np.bool)
        finish_mask[tuple([alive[term_mask] for alive in np.where(alive_paths)])] = True
        
        if self.ptr>0:
            path_slice = slice(self.path_start_idx, self.ptr)
            
            rews = np.append(self.rew_buf[finish_mask, path_slice], last_val[..., None], axis=-1)
            vals = np.append(self.val_buf[finish_mask, path_slice], last_val[..., None], axis=-1)
            
            deltas = rews[...,:-1] + self.gamma * vals[..., 1:] - vals[..., :-1]
            ### calculate (epistemic) iv-weighted advantages
            self.adv_buf[finish_mask, path_slice] = discount_cumsum(deltas, self.gamma, self.lam, axis=-1)
            #### R_t = A_GAE,t^iv + V_t
            self.ret_buf[finish_mask, path_slice] = self.adv_buf[finish_mask, path_slice] + self.val_buf[finish_mask, path_slice]

            costs = np.append(self.cost_buf[finish_mask, path_slice], last_cval[..., None], axis=-1)
            cvals = np.append(self.cval_buf[finish_mask, path_slice], last_cval[..., None], axis=-1)
            
            cdeltas = costs[...,:-1] + self.cost_gamma * cvals[..., 1:] - cvals[..., :-1]
            ### calculate (epistemic) iv-weighted advantages
            self.cadv_buf[finish_mask, path_slice] = discount_cumsum(cdeltas, self.cost_gamma, self.cost_lam, axis=-1)
            #### R_t = A_GAE,t^iv + V_t
            self.cret_buf[finish_mask, path_slice] = self.cadv_buf[finish_mask, path_slice] + self.cval_buf[finish_mask, path_slice]
                            
        # mark terminated paths
        self.terminated_paths_mask += finish_mask
         
    def get(self):
        """
        Returns a list of predetermined values in the buffer.
        
        Returns:
            list: [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf,
                self.logp_buf] + values_as_sorted_list(self.pi_info_bufs)
        """
        assert self.terminated_paths_mask.all()         ### all paths have to be finished

        if self.size>0:
            # Advantage normalizing trick for policy gradient
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[self.populated_mask].flatten())         # mpi can only handle 1d data
            self.adv_buf[self.populated_mask] = (self.adv_buf[self.populated_mask] - adv_mean) / (adv_std + EPS)

            # Center, but do NOT rescale advantages for cost gradient 
            # (since we're not just minimizing but aiming for a specific c)
            cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf[self.populated_mask].flatten())
            self.cadv_buf[self.populated_mask] -= cadv_mean
            
            ret_mean = self.ret_buf[self.populated_mask].mean()
            cret_mean = self.cret_buf[self.populated_mask].mean()
        else:
            ret_mean = 0
            cret_mean = 0

        res = [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf, 
                self.logp_buf, self.val_buf, 
                self.cval_buf, self.cost_buf] \
                + [v for v in values_as_sorted_list(self.pi_info_bufs)]
        # filter out unpopulated entries / finished paths
        res = [buf[self.populated_mask] for buf in res]
        diagnostics = dict( poolm_batch_size = self.populated_mask.sum(), 
                            poolm_ret_mean=ret_mean, 
                            poolm_cret_mean=cret_mean, 
                            )
        # reset
        self.reset()
        
        return res, diagnostics

