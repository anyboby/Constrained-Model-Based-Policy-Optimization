import numpy as np
import scipy

from utilities.mpi_tools import mpi_statistics_scalar
from utilities.utils import (combined_shape,
                                EPS,
                                keys_as_sorted_list,
                                values_as_sorted_list,
                                discount_cumsum,
                                )

from utilities.utils import (discount_cumsum_weighted,  ## imports for for iv advantages
                                triu_indices_t_h,
                                )

import warnings
import random

class CPOBuffer:

    def __init__(self, size, archive_size, 
                 observation_space, action_space, 
                 rollout_mode = 'schedule',
                 *args,
                 **kwargs,
                 ):
        self.obs_shape = observation_space.shape
        self.act_shape = action_space.shape
        self.archive_size = archive_size
        self.max_size = size

        # _____________________________ #
        # Create buffers and archives   #
        # _____________________________ #

        #### The buffers are for on-policy learning and get erased with 
        # every call to get(). Buffer contents are then copied to an archive where they remain 
        # accessable for model learning or off-policy learning. 

        self.reset_buffers()
        self.reset_arch()

        self.buf_dict = {
            'observations': self.obs_buf,
            'actions':      self.act_buf,
            'next_observations': self.nextobs_buf,
            'advantages':   self.adv_buf,
            'rewards':      self.rew_buf,
            'returns':      self.ret_buf,
            'values':       self.val_buf,
            'cadvantages':  self.cadv_buf,
            'costs':        self.cost_buf,
            'creturns':     self.cret_buf,
            'cvalues':      self.cval_buf,
            'log_policies': self.logp_buf,
            'terminals':    self.term_buf,
            'epochs':        self.epoch_buf,
        }

        self.arch_dict = {
            'observations': self.obs_archive,
            'actions':      self.act_archive,
            'next_observations': self.nextobs_archive,
            'advantages':   self.adv_archive,
            'rewards':      self.rew_archive,
            'returns':      self.ret_archive,
            'values':       self.val_archive,
            'cadvantages':  self.cadv_archive,
            'costs':        self.cost_archive,
            'creturns':     self.cret_archive,
            'cvalues':      self.cval_archive,
            'log_policies': self.logp_archive,
            'terminals':    self.term_archive,
            'epochs':        self.epoch_archive,
        }

    ''' initialize policy dependendant pi_info shapes, gamma, lam etc.'''
    def initialize(self, pi_info_shapes,
                    gamma=0.99, lam = 0.95,
                    cost_gamma = 0.99, cost_lam = 0.95,
                    ):
        self.pi_info_bufs = {k: np.zeros([self.max_size] + list(v), dtype=np.float32) 
                            for k,v in pi_info_shapes.items()}
        self.pi_info_archive = {k: np.zeros([self.archive_size] + list(v), dtype=np.float32) 
                            for k,v in pi_info_shapes.items()}

        self.arch_dict.update({'pi_infos':self.pi_info_archive})
        self.buf_dict.update({'pi_infos':self.pi_info_bufs})

        self.sorted_pi_info_keys = keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam

    def reset_buffers(self):
        size = self.max_size
        self.obs_buf = np.zeros(combined_shape(size, self.obs_shape), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, self.act_shape), dtype=np.float32)
        self.nextobs_buf = np.zeros(combined_shape(size, self.obs_shape), dtype=np.float32)          
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.roll_lengths_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros((size), dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)    # cost advantage
        self.croll_lengths_buf = np.zeros(size, dtype=np.float32)  
        self.cost_buf = np.zeros(size, dtype=np.float32)    # costs
        self.cret_buf = np.zeros(size, dtype=np.float32)    # cost return
        self.cval_buf = np.zeros((size), dtype=np.float32)    # cost value
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.term_buf = np.zeros(size, dtype=np.bool_)
        self.epoch_buf = np.ones(size, dtype=np.float32)*-1
        self.ptr, self.path_start_idx, self.path_finished = 0,0, False

    def reset_arch(self):
        self.archive_full = False
        archive_size = self.archive_size
        self.obs_archive = np.zeros(combined_shape(archive_size, self.obs_shape), dtype=np.float32)
        self.act_archive = np.zeros(combined_shape(archive_size, self.act_shape), dtype=np.float32)
        # bit memory inefficient, but more convenient, to store next_obs
        self.nextobs_archive = np.zeros(combined_shape(archive_size, self.obs_shape), dtype=np.float32)
        self.adv_archive = np.zeros(archive_size, dtype=np.float32)
        self.rew_archive = np.zeros(archive_size, dtype=np.float32)
        self.ret_archive = np.zeros(archive_size, dtype=np.float32)
        self.val_archive = np.zeros(archive_size, dtype=np.float32)
        self.cadv_archive = np.zeros(archive_size, dtype=np.float32)    # cost advantage
        self.cost_archive = np.zeros(archive_size, dtype=np.float32)    # costs
        self.cret_archive = np.zeros(archive_size, dtype=np.float32)    # cost return
        self.cval_archive = np.zeros(archive_size, dtype=np.float32)    # cost value
        self.logp_archive = np.zeros(archive_size, dtype=np.float32)
        self.term_archive = np.zeros(archive_size, dtype=np.bool_)
        self.epoch_archive = np.ones(archive_size, dtype=np.int)*-1
        self.archive_ptr = 0
        self.max_pointer = 0

    @property
    def size(self):
        return self.ptr

    @property
    def max_ep(self):
        return int(np.max(self.epoch_archive))
    @property
    def min_ep(self):
        return int(np.min(self.epoch_archive[self.epoch_archive>-1]))
    @property
    def epochs_list(self):
        bins = np.bincount(self.epoch_archive[self.epoch_archive>=0])
        eps = np.nonzero(bins)

        return np.squeeze(eps, axis=0)
    
    @property
    def arch_size(self):
        return self.max_pointer

    def store(self, obs, act, next_obs, rew, val, cost, cval, logp, pi_info, term, epoch):
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.nextobs_buf[self.ptr] = next_obs
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.cost_buf[self.ptr] = cost
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        self.term_buf[self.ptr] = term
        self.epoch_buf[self.ptr] = epoch

        for k in self.sorted_pi_info_keys:
            self.pi_info_bufs[k][self.ptr] = pi_info[k]
        self.ptr += 1
        self.path_finished = False


    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val, axis=-1)
        vals = np.append(self.val_buf[path_slice], last_val, axis=-1)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        #=====================================================================#
        #  Normal Generalized Advantage                                       #
        #=====================================================================#            
        ### gae
        self.adv_buf[path_slice] = discount_cumsum(deltas, discount=self.gamma, lam=self.lam, axis=-1)
        #### R_t = A_GAE,t^iv + V_t
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]

        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval, axis=-1)

        cdeltas = costs[:-1] + self.cost_gamma * cvals[1:] - cvals[:-1]

        ### calculate advantages
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, discount=self.cost_gamma, lam=self.cost_lam, axis=-1)
        #### R_t = A_GAE,t^iv + V_t
        self.cret_buf[path_slice] = self.cadv_buf[path_slice] + self.cval_buf[path_slice]

        #### useful line for debugging (A should be the same for roughly equal weights):
        #######     np.max((discount_cumsum(cdeltas, self.gamma, self.lam)-self.cadv_buf[path_slice])**2)

        self.path_start_idx = self.ptr
        self.path_finished = True


    def dump_to_archive(self):
        """
        dumps all samples that are currently contained in the buffers to 
        the archive. Careful: use this only, when all paths are finished, 
        otherwise the copied data is incomplete

        resets pointers to zero.

        """
        assert self.path_finished
        
        if self.archive_ptr >= self.archive_size-self.ptr:
            self.archive_full = True
            self.archive_ptr= 0
            warnings.warn('Archive is full, deleting old samples.')

        arch_slice = slice(self.archive_ptr, self.archive_ptr+self.ptr)
        buf_slice = slice(0, self.ptr)

        self.obs_archive[arch_slice] = self.obs_buf[buf_slice]
        self.act_archive[arch_slice] = self.act_buf[buf_slice]
        self.nextobs_archive[arch_slice] = self.nextobs_buf[buf_slice]
        self.rew_archive[arch_slice] = self.rew_buf[buf_slice]
        self.val_archive[arch_slice] = self.val_buf[buf_slice]
        self.cost_archive[arch_slice] = self.cost_buf[buf_slice]
        self.cval_archive[arch_slice] = self.cval_buf[buf_slice]
        self.logp_archive[arch_slice] = self.logp_buf[buf_slice]
        self.term_archive[arch_slice] = self.term_buf[buf_slice]
        self.adv_archive[arch_slice] = self.adv_buf[buf_slice]
        self.ret_archive[arch_slice] = self.ret_buf[buf_slice]
        self.cadv_archive[arch_slice] = self.cadv_buf[buf_slice]
        self.cret_archive[arch_slice] = self.cret_buf[buf_slice]
        self.epoch_archive[arch_slice] = self.epoch_buf[buf_slice]

        for k in self.sorted_pi_info_keys:
            self.pi_info_archive[k][arch_slice] = self.pi_info_bufs[k][buf_slice]
        
        self.archive_ptr+=self.ptr
        self.max_pointer = max(self.archive_ptr, self.max_pointer)
    def get(self):
        """
        Returns a list of predetermined values in the buffer.
        
        Returns:
            list: [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf,
                self.logp_buf] + values_as_sorted_list(self.pi_info_bufs)
        """
        #assert self.ptr == self.max_size    # uffer has to be full before you can get
        
        # Advantage normalizing trick for policy gradient
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[:self.ptr])
        self.adv_buf[:self.ptr] = (self.adv_buf[:self.ptr] - adv_mean) / (adv_std + EPS)

        # Center, but do NOT rescale advantages for cost gradient
        cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf[:self.ptr])
        self.cadv_buf[:self.ptr] -= cadv_mean
        self.dump_to_archive() 


        res = [self.obs_buf, self.act_buf, self.adv_buf,
                self.cadv_buf, self.ret_buf, self.cret_buf, 
                self.logp_buf, self.val_buf, 
                self.cval_buf, 
                self.cost_buf] \
                + values_as_sorted_list(self.pi_info_bufs)
        res = [v.copy()[:self.ptr] for v in res]

        ##### diagnostics        
        ret_mean = self.ret_buf[:self.ptr].mean()
        cret_mean = self.cret_buf[:self.ptr].mean()

        diagnostics = dict(poolr_ret_mean=ret_mean, \
                            poolr_cret_mean=cret_mean, 
                            )

        self.reset_buffers() ### reset

        return res, diagnostics
    
    def get_archive(self, fields=None):
        """
        returns all data currently contained in the buffer. if fields is None a 
        default dictionary containing:
        obs, action, next_obs, r, term
        is returned. 
        Remember, that old values / advantages etc. might come from
        old policies.
        choose from these fields:
            'observations'
            'actions'
            'next_observations'
            'advantages'
            'rewards'
            'returns'
            'values'
            'cadvantages'
            'costs'
            'creturns'
            'cvalues'
            'log_policies'
            'pi_infos'
            'terminals'
            'epochs'
            'all'

        Args:
            fields: a list containing the key words for the desired 
            data types. e.g.: ['observations', 'actions', 'values'], 'all' will return all of the possible fields
        """
        if fields is None:
            archives = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        elif 'all' in fields:
            archives = self.arch_dict.keys() 
        else:
            archives = fields
        if 'pi_infos' in fields:
            pi_info_requested = True
            fields.remove('pi_infos')
        else:
            pi_info_requested = False

        ptr = self.arch_size
        samples = {archive:self.arch_dict[archive][:ptr] for archive in archives}
        
        if pi_info_requested:
            pi_infos = {
                k:self.pi_info_archive[k][:ptr] for k in keys_as_sorted_list(self.pi_info_archive)
                }
            samples.update(pi_infos)
        # add current buffers
        return samples

    def rand_batch_from_archive(self, batch_size, fields=None):
        """
        returns random batch from the archive. if data_type is None a 
        default dictionary containing:
        obs, action, next_obs, r, term
        is returned. 
        Remember, that old values / advantages etc. might come from
        old policies.
        choose from these data types:
            'observations'
            'actions'
            'next_observations'
            'advantages'
            'rewards'
            'returns'
            'values'
            'cadvantages'
            'costs'
            'creturns'
            'cvalues'
            'log_policies'
            'terminals'

        Args:
            fields: a list containing the key words for the desired 
            data types. e.g.: ['observations', 'actions', 'values']
        Returns:
            samples: A dict containining random archive values for default / specified fields 
            as np arrays of batch_size
        """
        if fields is None:
            archives = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        else:
            archives = fields

        rand_indc = self._random_indices(batch_size)
        samples = {archive: self.arch_dict[archive][rand_indc] for archive in archives}
        
        return samples

    def boltz_dist(self, kls, alpha=1):
        '''
        args: kls should be a list of KL divergence means over the epochs contained in the archive. 
        call epochs_list for such a list. 
        '''
        ep_probs = np.exp(alpha * np.negative(kls))
        ep_probs /= np.sum(ep_probs)
        sample_p = np.bincount(self.epoch_archive[self.epoch_archive>=0]).astype(np.float32)
        sample_p[sample_p>0] = ep_probs/sample_p[sample_p>0]

        btz = np.where(self.epoch_archive>=0, sample_p[self.epoch_archive], 0)
        return btz

    def disc_dist(self, disc):
        epochs = np.arange(self.min_ep, self.max_ep+1)
        ep_probs = disc**(self.max_ep-epochs)
        ep_probs /= np.sum(ep_probs)
        sample_n = np.bincount(self.epoch_archive[self.epoch_archive>=0])
        sample_prob = ep_probs/sample_n[sample_n>0]
        disc_dist = np.where(self.epoch_archive>=0, sample_prob[self.epoch_archive-self.min_ep], 0)
        return disc_dist

    def unif_dist(self):
        unif = np.ones(shape=self.archive_size)
        unif[self.epoch_archive<0] = 0
        unif /= np.sum(unif)
        return unif

    def distributed_batch_from_archive(self, batch_size, dist, fields=None):
        """
        returns random batch from the archive. if data_type is None a 
        default dictionary containing:
        obs, action, next_obs, r, term
        is returned. 
        Remember, that old values / advantages etc. might come from
        old policies.
        choose from these data types:
            'observations'
            'actions'
            'next_observations'
            'advantages'
            'rewards'
            'returns'
            'values'
            'cadvantages'
            'costs'
            'creturns'
            'cvalues'
            'log_policies'
            'terminals'

        Args:
            fields: a list containing the key words for the desired 
            data types. e.g.: ['observations', 'actions', 'values']
        Returns:
            samples: A dict containining random archive values for default / specified fields 
            as np arrays of batch_size
        """
        if fields is None:
            archives = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        else:
            archives = fields

        if 'pi_infos' in fields:
            pi_info_requested = True
            fields.remove('pi_infos')
        else:
            pi_info_requested = False

        rand_indc = np.random.choice(np.arange(self.archive_size), size=batch_size, p=dist)
        samples = {archive: self.arch_dict[archive][rand_indc] for archive in archives}
        
        if pi_info_requested:
            pi_infos = {
                k:self.pi_info_archive[k][rand_indc] for k in keys_as_sorted_list(self.pi_info_archive)
                }
            samples.update(pi_infos)

        return samples


    def epoch_batch(self, batch_size, epochs, fields=None):
        """
        returns batch collected under the latest policy from the archive. if fields is None a 
        default dictionary containing:
        obs, action, next_obs, r, term
        is returned. 
        choose from these data types:
            'observations'
            'actions'
            'next_observations'
            'advantages'
            'rewards'
            'returns'
            'values'
            'cadvantages'
            'costs'
            'creturns'
            'cvalues'
            'log_policies'
            'pi_infos'
            'terminals'

        Args:
            fields: a list containing the key words for the desired 
            data types. e.g.: ['observations', 'actions', 'values']
        Returns:
            samples: A dict containining all archive values for default / specified fields 
            as np arrays collected under the latest (available) policy
        """
        assert len(np.shape(epochs))==1
        if fields is None:
            archives = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        else:
            archives = fields

        if 'pi_infos' in fields:
            pi_info_requested = True
            fields.remove('pi_infos')
        else:
            pi_info_requested = False
        max_epoch = self.max_ep
        ep = np.array(epochs)

        ### if epoch not contained
        if np.any(ep>max_epoch) or np.any(ep<self.min_ep):
            print(f'Warning: epoch not contained in buffer.')
            return None
        
        ep[ep<0] = 1+max_epoch+ep[ep<0]
        indc = np.array([np.random.choice(np.squeeze(np.where(self.epoch_archive==ep)), size=batch_size) for ep in epochs])
        samples = {archive: self.arch_dict[archive][indc] for archive in archives}
        
        if pi_info_requested:
            pi_infos = {
                k:self.pi_info_archive[k][indc] for k in keys_as_sorted_list(self.pi_info_archive)
                }
            samples.update(pi_infos)
        
        return samples


    def _random_indices(self, batch_size):
        """
        returns np array of random indices limited to current archive size.
        """

        if self.arch_size == 0: return np.arange(0, 0)
        #return random.sample(range(0, self.arch_size), batch_size)
        return np.random.randint(0, self.arch_size, batch_size)
