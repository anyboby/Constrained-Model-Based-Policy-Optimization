from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import time
import pdb
import itertools
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

import tensorflow.contrib.losses as tf_losses

import numpy as np
from scipy.io import savemat, loadmat

from .utils import get_required_argument, TensorStandardScaler
from .fc import FC
from .logger import Progress, Silent
from models.base_model import EnsembleModel

np.set_printoptions(precision=5)


class PE(EnsembleModel):
    """Neural network ensemble which models aleatoric uncertainty and epistemic uncertainty.
    """
    def __init__(self, params):
        """Initializes a class instance.

        Arguments:
            params (dict): A dict of model parameters:
                name('str): Model name, used for logging/use in variable scopes.
                    Warning: Models with the same name will overwrite each other.
                sess('tf session'): tensorflow session, if not provided newly created
                loss('str'): loss type, choose from 'NLL', 'MSPE', 'MSE', 'Huber', 'CE'
                clip_loss(`bool`): clip losses? performs trust-region like updates, should be 
                    chosen in junction with a probabilistic loss
                kl_cliprange(`float`): if clip_loss, what kl-divergence should losses be clipped at?
                max_logvar(`float`): maximum initial log-var for NLL loss
                min_logvar(`float`): minimum initial log-var for NLL loss
                use_scaler_in(`bool`): determines whether to scale inputs to std 1 and mean 0
                use_scaler_out(`bool`): determines whether to scale outputs to std 1 and mean 0
                    Warning: this heavily affects loss composition
                num_networks(`int`): number of networks in ensemble 
                num_elites(`int`): number of elite-networks 
                load_model(`bool`): load an existing model?
                model_dir(`str`):  directory of model 

        """
        self.name = get_required_argument(params, 'name', 'Must provide name.')
        self.model_dir = params.get('model_dir', None)

        print(f'[ {self.name} ] Initializing model: {self.name} | {params.get("num_networks")} networks | {params.get("num_elites")} elites')
        if params.get('sess', None) is None:
            config = tf.ConfigProto()
            #config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config)
        else:
            self._sess = params.get('sess')
        
        self.loss_type = params.get('loss', 'NLL')

        self.clip_loss = params.get('clip_loss', False)
        self.kl_cliprange = params.get('kl_cliprange', 0.1)   #### Only relevant if clip_loss is True

        # Instance variables
        self.finalized = False
        self.layers, self.max_logvar, self.min_logvar = [], None, None
        self.max_logvar_param = params.get('max_logvar', .5)
        self.min_logvar_param = params.get('min_logvar', -6)
        self.decays, self.optvars, self.nonoptvars = [], [], []
        self.end_act, self.end_act_name = None, None
        self.use_scaler_in = params.get('use_scaler_in', False)
        self.use_scaler_out = params.get('use_scaler_out', False)
        self.scaler_in = None
        self.scaler_out = None

        # Training objects
        self.optimizer = None
        self.sy_train_in, self.sy_train_targ = None, None
        self.train_op, self.loss = None, None

        # Prediction objects
        self.sy_pred_in2d, self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = None, None, None
        self.sy_pred_mean2d, self.sy_pred_var2d = None, None
        self.sy_pred_in3d, self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = None, None, None

        if params.get('load_model', False):
            if self.model_dir is None:
                raise ValueError("Cannot load model without providing model directory.")
            self._load_structure()
            self.num_nets, self.model_loaded = self.layers[0].get_ensemble_size(), True
            print("Model loaded from %s." % self.model_dir)
            self.num_elites = params['num_elites']
            self._model_inds = np.random.randint(self.num_nets, size=self.num_elites)
        else:
            self.num_nets = params.get('num_networks', 1)
            self.num_elites = params['num_elites'] #params.get('num_elites', 1)
            self._model_inds = np.random.randint(self.num_nets, size=self.num_elites)
            self.model_loaded = False

        if self.num_nets == 1:
            print("Created a neural network with variance predictions.")
        else:
            print("Created an ensemble of {} neural networks with variance predictions | Elites: {}".format(self.num_nets, self.num_elites))

    ###################################
    # Network Structure Setup Methods #
    ###################################

    def add(self, layer):
        """Adds a new layer to the network.

        Arguments:
            layer: (layer) The new layer to be added to the network.
                   If this is the first layer, the input dimension of the layer must be set.

        Returns: None.
        """
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        if len(self.layers) == 0 and layer.get_input_dim() is None:
            raise ValueError("Must set input dimension for the first layer.")
        if self.model_loaded:
            raise RuntimeError("Cannot add layers to a loaded model.")

        layer.set_ensemble_size(self.num_nets)
        if len(self.layers) > 0:
            layer.set_input_dim(self.layers[-1].get_output_dim())
        self.layers.append(layer.copy())

    def pop(self):
        """Removes and returns the most recently added layer to the network.

        Returns: (layer) The removed layer.     
        """
        if len(self.layers) == 0:
            raise RuntimeError("Network is empty.")
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        if self.model_loaded:
            raise RuntimeError("Cannot remove layers from a loaded model.")

        return self.layers.pop()

    def finalize(self, optimizer, optimizer_args=None, weighted=False, *args, **kwargs):
        """Finalizes the network.

        Arguments:
            optimizer: (tf.train.Optimizer) An optimizer class from those available at tf.train.Optimizer.
            optimizer_args: (dict) A dictionary of arguments for the __init__ method of the chosen optimizer.

        Returns: None
        """
        if len(self.layers) == 0:
            raise RuntimeError("Cannot finalize an empty network.")
        if self.finalized:
            raise RuntimeError("Can only finalize a network once.")

        optimizer_args = {} if optimizer_args is None else optimizer_args
        global_step = tf.Variable(0, trainable=False)
        if "learning_rate_decay" in optimizer_args:
            starter_learning_rate = optimizer_args['learning_rate']
            learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                                                                global_step=global_step,
                                                                decay_steps=optimizer_args.get('decay_steps', 10000),
                                                                decay_rate=optimizer_args['learning_rate_decay'], 
                                                                staircase=True)            
            optimizer_args.pop('learning_rate_decay', None)
            optimizer_args.pop('decay_steps', None)
            optimizer_args['learning_rate'] = learning_rate    

        
        # Add variance output.
        if self.is_probabilistic:
            self.layers[-1].set_output_dim(2 * self.layers[-1].get_output_dim())

        # Remove last activation (later applied manually, to isolate variance if included)
        self.end_act = self.layers[-1].get_activation()
        self.end_act_name = self.layers[-1].get_activation(as_func=False)
        self.layers[-1].unset_activation()

        # Construct all variables.
        with self.sess.as_default():
            with tf.variable_scope(self.name):
                if self.use_scaler_in:
                    self.scaler_in = TensorStandardScaler(self.layers[0].get_input_dim(), name='scaler_in')
                if self.use_scaler_out:
                    if self.is_probabilistic:
                        self.scaler_out = TensorStandardScaler(self.layers[-1].get_output_dim() // 2, name='scaler_out')
                    else:
                        self.scaler_out = TensorStandardScaler(self.layers[-1].get_output_dim(), name='scaler_out')
                        
                if self.loss_type=='NLL':
                    self.max_logvar = tf.Variable(np.ones([1, self.layers[-1].get_output_dim() // 2])*self.max_logvar_param, dtype=tf.float32,
                                                name="max_log_var")
                    self.min_logvar = tf.Variable(np.ones([1, self.layers[-1].get_output_dim() // 2])*self.min_logvar_param, dtype=tf.float32,
                                                name="min_log_var")
                for i, layer in enumerate(self.layers):
                    with tf.variable_scope("Layer%i" % i):
                        layer.construct_vars()
                        self.decays.extend(layer.get_decays())
                        self.optvars.extend(layer.get_vars())
        if self.loss_type=='NLL':
            self.optvars.extend([self.max_logvar, self.min_logvar])
        if self.use_scaler_in:
            self.nonoptvars.extend(self.scaler_in.get_vars())
        if self.use_scaler_out:
            self.nonoptvars.extend(self.scaler_out.get_vars())


        # Set up training
        with tf.variable_scope(self.name):
            self.optimizer = optimizer(**optimizer_args)
            self.sy_train_in = tf.placeholder(dtype=tf.float32,
                                              shape=[self.num_nets, None, self.layers[0].get_input_dim()],
                                              name="training_inputs")
            if self.is_probabilistic:
                self.sy_train_targ = tf.placeholder(dtype=tf.float32,
                                                shape=[self.num_nets, None, self.layers[-1].get_output_dim() // 2],
                                                name="training_targets")
                
            else:
                self.sy_train_targ = tf.placeholder(dtype=tf.float32,
                                                shape=[self.num_nets, None, self.layers[-1].get_output_dim()],
                                                name="training_targets")

            if weighted:
                self.weights = tf.placeholder(dtype=tf.float32,
                                                shape=[self.num_nets, None],
                                                name="weighted_loss")
            else:
                self.weights=None
            
            #### set up losses
            if self.loss_type == 'NLL':
                if self.clip_loss:
                    self.kl_cliprange_ph = tf.placeholder(dtype=tf.float32,
                                    shape=(),
                                    name="kl_cliprange_ph")
                    self.old_pred_ph = tf.placeholder(dtype=tf.float32,
                                    shape=[self.num_nets, None, self.layers[-1].get_output_dim() // 2],
                                    name="old_predictions_v")
                    self.old_pred_var_ph = tf.placeholder(dtype=tf.float32,
                                    shape=[self.num_nets, None, self.layers[-1].get_output_dim() // 2],
                                    name="old_predictions_var")
                else:
                    self.old_pred_ph = None
                    self.old_pred_var_ph = None

                train_loss = tf.reduce_sum(self._nll_loss(self.sy_train_in, 
                                                            self.sy_train_targ, 
                                                            inc_var_loss=True, 
                                                            weights=self.weights, 
                                                            oldpred_v=self.old_pred_ph, 
                                                            oldpred_var = self.old_pred_var_ph)
                                                            )
                train_loss += tf.add_n(self.decays)
                train_loss += 0.01 * tf.reduce_sum(self.max_logvar) - 0.01 * tf.reduce_sum(self.min_logvar)
                self.loss = self._nll_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, weights=self.weights)
                self.tensor_loss, self.debug_mean = self._nll_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, tensor_loss=True, weights=self.weights)            
            elif self.loss_type == 'MSPE':
                train_loss = tf.reduce_sum(self._mspe_loss(self.sy_train_in, 
                                                            self.sy_train_targ, 
                                                            inc_var_loss=True,)
                                                            )
                train_loss += tf.add_n(self.decays)
                self.loss = self._nll_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, weights=self.weights)
                self.tensor_loss, self.debug_mean = self._nll_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, tensor_loss=True, weights=self.weights)            

            elif self.loss_type == 'MSE':
                if self.clip_loss:
                    self.kl_cliprange_ph = tf.placeholder(dtype=tf.float32,
                                    shape=(),
                                    name="kl_cliprange_ph")                    
                    self.old_pred_ph = tf.placeholder(dtype=tf.float32,
                                    shape=[self.num_nets, None, self.layers[-1].get_output_dim()],
                                    name="old_predictions")
                else:
                    self.old_pred_ph = None
                train_loss = tf.reduce_sum(self._nll_loss(self.sy_train_in, 
                                                            self.sy_train_targ, 
                                                            inc_var_loss=False, 
                                                            weights=self.weights, 
                                                            oldpred_v=self.old_pred_ph)
                                                            )
                train_loss += tf.add_n(self.decays)
                self.loss = self._nll_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, weights=self.weights)
                self.tensor_loss, self.debug_mean = self._nll_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, tensor_loss=True, weights=self.weights)            
            
            elif self.loss_type == 'Huber':
                train_loss = tf.reduce_sum(self._huber_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, weights=self.weights, delta=0.3))
                train_loss += tf.add_n(self.decays)
                train_loss += 0.01 * tf.reduce_sum(self.max_logvar) - 0.01 * tf.reduce_sum(self.min_logvar)
                self.loss = self._huber_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, weights=self.weights, delta=0.3)
                self.tensor_loss, self.debug_mean = self._huber_loss(self.sy_train_in, self.sy_train_targ, inc_var_loss=False, tensor_loss=True, weights=self.weights, delta=0.3)
            
            elif self.loss_type == 'CE':
                train_loss = tf.reduce_sum(self._ce_loss(self.sy_train_in, self.sy_train_targ, weights=self.weights))
                train_loss += tf.add_n(self.decays)
                self.loss = self._ce_loss(self.sy_train_in, self.sy_train_targ, weights=self.weights)
                self.tensor_loss, self.debug_mean = self._ce_loss(self.sy_train_in, self.sy_train_targ, tensor_loss=True, weights=self.weights)

            #### _________________________ ####  
            ####      Optimization Ops     ####
            #### _________________________ ####
            
            # self.train_op = self.optimizer.minimize(train_loss, var_list=self.optvars)
            grads_a_vars = self.optimizer.compute_gradients(train_loss, var_list=self.optvars)
            # grads_a_vars = [(tf.clip_by_value(grad, -1, 1.), var) for grad, var in grads_a_vars]
            self.train_op = self.optimizer.apply_gradients(grads_and_vars=grads_a_vars, global_step=global_step)
                    
        # Initialize all variables
        self.sess.run(tf.variables_initializer(self.optvars + self.nonoptvars + self.optimizer.variables()))

        # Set up prediction
        with tf.variable_scope(self.name):
            if self.is_probabilistic:
                self.sy_pred_in2d = tf.placeholder(dtype=tf.float32,
                                                shape=[None, self.layers[0].get_input_dim()],
                                                name="2D_training_inputs")
                self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = \
                    self.create_prediction_tensors(self.sy_pred_in2d, factored=True)
                self.sy_pred_mean2d = tf.reduce_mean(self.sy_pred_mean2d_fac, axis=0)
                self.sy_pred_var2d = tf.reduce_mean(self.sy_pred_var2d_fac, axis=0) + \
                    tf.reduce_mean(tf.square(self.sy_pred_mean2d_fac - self.sy_pred_mean2d), axis=0)

                self.sy_pred_in3d = tf.placeholder(dtype=tf.float32,
                                                shape=[self.num_nets, None, self.layers[0].get_input_dim()],
                                                name="3D_training_inputs")
                self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = \
                    self.create_prediction_tensors(self.sy_pred_in3d, factored=True)
            else:
                self.sy_pred_in2d = tf.placeholder(dtype=tf.float32,
                                                shape=[None, self.layers[0].get_input_dim()],
                                                name="2D_training_inputs")
                self.sy_pred_mean2d_fac  = \
                    self.create_prediction_tensors(self.sy_pred_in2d, factored=True)
                self.sy_pred_mean2d = tf.reduce_mean(self.sy_pred_mean2d_fac, axis=0)

                self.sy_pred_in3d = tf.placeholder(dtype=tf.float32,
                                                shape=[self.num_nets, None, self.layers[0].get_input_dim()],
                                                name="3D_training_inputs")
                self.sy_pred_mean3d_fac = \
                    self.create_prediction_tensors(self.sy_pred_in3d, factored=True)

        # Load model if needed
        if self.model_loaded:
            with self.sess.as_default():
                params_dict = loadmat(os.path.join(self.model_dir, "%s.mat" % self.name))
                all_vars = self.nonoptvars + self.optvars
                for i, var in enumerate(all_vars):
                    var.load(params_dict[str(i)])
        self.finalized = True

    ##################
    # Custom Methods #
    ##################

    def _save_state(self, idx):
        self._state[idx] = [layer.get_model_vars(idx, self.sess) for layer in self.layers]

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                self._save_state(i)
                updated = True
                improvement = (best - current) / best
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))
        
        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            # print('[ BNN ] Breaking at epoch {}: {} epochs since update ({} max)'.format(epoch, self._epochs_since_update, self._max_epochs_since_update))
            return True
        else:
            return False

    def _start_train(self):
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.num_nets)}
        self._epochs_since_update = 0

    def _end_train(self, holdout_losses):
        sorted_inds = np.argsort(holdout_losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()
        print('Using {} / {} models: {}'.format(self.num_elites, self.num_nets, self._model_inds))

    @property
    def elite_inds(self):
        return self._model_inds

    def reset(self):
        print(f'[ {self.name}] ] Resetting model')
        # Initialize all variables
        
        #### @anyboby debug: trying different initializer since this thing below doesn't seem to reset
        [layer.reset(self.sess) for layer in self.layers]
        #self.sess.run(tf.variables_initializer(self.optvars + self.nonoptvars + self.optimizer.variables()))
    
    @property
    def is_probabilistic(self):
        return True if 'NLL' in self.loss_type or 'MSPE' in self.loss_type else False

    @property
    def is_ensemble(self):
        return self.num_nets>1

    @property
    def in_dim(self):
        return self.layers[0].get_input_dim()
    
    @property
    def out_dim(self):
        if self.is_probabilistic:
            self.layers[-1].get_output_dim()//2
        else:
            return self.layers[0].get_output_dim()

    @property
    def is_tf_model(self):
        return True

    @property
    def sess(self):
        return self._sess

    def validate(self, inputs, targets):
        inputs = np.tile(inputs[None], [self.num_nets, 1, 1])
        targets = np.tile(targets[None], [self.num_nets, 1, 1])
        losses = self.sess.run(
            self.loss,
            feed_dict={
                self.sy_train_in: inputs,
                self.sy_train_targ: targets
                }
        )
        mean_elite_loss = np.sort(losses)[:self.num_elites].mean()
        return mean_elite_loss

    #################
    # Model Methods #
    #################

    def train(self, inputs, targets,
              batch_size=32, max_epochs=None, max_epochs_since_update=5, min_epoch_before_break = 0,
              hide_progress=False, holdout_ratio=0.0, max_logging=5000, max_grad_updates=None, timer=None, max_t=None, **kwargs):
        """Trains/Continues network training

        Arguments:
            inputs (`np.ndarray`): Network inputs in the training dataset in rows.
            targets (`np.ndarray`): Network target outputs in the training dataset in rows corresponding
                to the rows in inputs.
            batch_size (`int`): The minibatch size to be used for training.
            epochs (`int`): Number of epochs (full network passes that will be done.
            max_epochs(`int`): Maximum number of epochs
            max_epochs_since_update(`int`): Number of epochs of not updating best model before terminating 
            min_epoch_before_break(`int`): Number of minimum epochs before training is terminated
            holdout_ratio(`float`<1): ratio of evaluation samples to assess performance improvement
            hide_progress (bool): If True, hides the progress bar shown at the beginning of training.
        kwargs:
            'kl_cliprange': if clip_loss: kl-divergence for clipping range
            'old_pred': if clip_loss: old prediction for input
            'old_pred_var' if clip_loss: old variance prediction for input
        Returns: None
        """
        self._max_epochs_since_update = max_epochs_since_update
        self._start_train()
        break_train = False

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        # Split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        if self.clip_loss:
            kl_cliprange = kwargs.get('kl_cliprange', self.kl_cliprange)

        if self.loss_type=='MSE' and self.clip_loss:
            old_pred = kwargs['old_pred']
            old_pred = old_pred[permutation[num_holdout:]]
        
        if self.loss_type=='NLL' and self.clip_loss:
            old_pred = kwargs['old_pred']
            old_pred = old_pred[permutation[num_holdout:]]
            
            old_pred_var = kwargs['old_pred_var']
            old_pred_var = old_pred_var[permutation[num_holdout:]]
        
        if self.weights is not None:
            weights = kwargs['weights']
            weights, holdout_weights = weights[permutation[num_holdout:]], weights[permutation[:num_holdout]]
            holdout_weights = np.tile(holdout_weights[None], [self.num_nets, 1])

        print(f'[ {self.name} ] Training {inputs.shape} | Holdout: {holdout_inputs.shape}')

        if self.use_scaler_in:
            with self.sess.as_default():
                self.scaler_in.fit(inputs)
        if self.use_scaler_out:
            with self.sess.as_default():
                self.scaler_out.fit(targets)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])
        if hide_progress:
            progress = Silent()
        else:
            progress = Progress(max_epochs)

        if max_epochs:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        t0 = time.time()
        grad_updates = 0
        for epoch in epoch_iter:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                
                feed_dict={
                    self.sy_train_in: inputs[batch_idxs],
                    self.sy_train_targ: targets[batch_idxs]
                    }
                
                if self.loss_type=='MSE' and self.clip_loss:
                    feed_dict.update({self.old_pred_ph:old_pred[batch_idxs], self.kl_cliprange_ph:kl_cliprange})

                elif self.loss_type=='NLL' and self.clip_loss:
                    feed_dict.update({self.old_pred_ph:old_pred[batch_idxs], self.kl_cliprange_ph:kl_cliprange})
                    feed_dict.update({self.old_pred_var_ph:old_pred_var[batch_idxs]})

                if self.weights is not None:
                    feed_dict.update({self.weights:weights[batch_idxs]})

                self.sess.run(
                    self.train_op,
                    feed_dict=feed_dict
                )
                grad_updates += 1

            idxs = shuffle_rows(idxs)
            if not hide_progress:
                ### no holdout evaluation
                progress_feed_dict = {
                        self.sy_train_in: inputs[idxs[:, :max_logging]],
                        self.sy_train_targ: targets[idxs[:, :max_logging]]
                    }
                if self.weights is not None:
                    progress_feed_dict.update({self.weights:weights[idxs[:, :max_logging]]})

                if holdout_ratio < 1e-12:
                    feed_dict={
                        self.sy_train_in: inputs[idxs[:, :max_logging]],
                        self.sy_train_targ: targets[idxs[:, :max_logging]]
                    }                    
                    losses = self.sess.run(
                            self.loss,
                            feed_dict=progress_feed_dict
                        )
                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    progress.set_description(named_losses)
                ### eval_runs with holdout
                else:
                    holdout_feed_dict = {
                                self.sy_train_in: holdout_inputs,
                                self.sy_train_targ: holdout_targets
                            }
                    if self.weights is not None:
                        holdout_feed_dict.update({self.weights:holdout_weights})
                        
                    losses = self.sess.run(
                            self.loss,
                            feed_dict = progress_feed_dict
                        )
                    holdout_losses = self.sess.run(
                            self.loss,
                            feed_dict = holdout_feed_dict
                        )

                    ### just to debug, which parts of the output generate most losses ###
                    holdout_tensor_losses, holdout_outputs = self.sess.run(
                            [self.tensor_loss, self.debug_mean],
                            feed_dict = holdout_feed_dict
                        )

                    ### just to debug, which parts of the output generate most losses ###
                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    named_holdout_losses = [['V{}'.format(i), holdout_losses[i]] for i in range(len(holdout_losses))]
                    named_losses = named_losses + named_holdout_losses + [['T', time.time() - t0]]
                    progress.set_description(named_losses)

                    #### returns true if the best model hasn't been updated for more than 
                    #               ._max_epochs_since_update model-bnn train epochs
                    break_train = self._save_best(epoch, holdout_losses)
                    pass
            progress.update()
            t = time.time() - t0
            if (break_train and epoch>min_epoch_before_break) or (max_grad_updates and grad_updates > max_grad_updates):
                break
            if max_t and t > max_t:
                descr = 'Breaking because of timeout: {}! (max: {})'.format(t, max_t)
                progress.append_description(descr)
                # print('Breaking because of timeout: {}! | (max: {})\n'.format(t, max_t))
                # time.sleep(5)
                break

        progress.stamp()
        if timer: timer.stamp('bnn_train')

        holdout_losses = self.sess.run(
            self.loss,
            feed_dict={
                self.sy_train_in: holdout_inputs,
                self.sy_train_targ: holdout_targets
            }
        )

        if timer: timer.stamp('bnn_holdout')

        self._end_train(holdout_losses)
        if timer: timer.stamp('bnn_end')

        val_loss = (np.sort(holdout_losses)[:self.num_elites]).mean()
        model_metrics = {f'{self.name}/val_loss': val_loss}
        print(f'[ {self.name} ] Holdout', np.sort(holdout_losses), model_metrics)
        return OrderedDict(model_metrics)

    def predict(self, inputs, *args, **kwargs):
        """Returns the mean distribution predicted by the ensemble for each input vector in inputs.
        Behavior is affected by the configuration of the model:

        returns mean predictions between models, if ensemble

        self._is_probabilistic = True: returns (mean, variance) predictions
        self._is_probabilistic = False: returns mean predictions
        """

        assert len(inputs.shape) == 2
        
        if self.is_probabilistic:
            return self.sess.run(
                [self.sy_pred_mean2d, self.sy_pred_var2d],
                feed_dict={self.sy_pred_in2d: inputs}
            )
        else: ### only deterministic predictions
            return self.sess.run(
                self.sy_pred_mean2d,
                feed_dict={self.sy_pred_in2d: inputs}
            )

    def predict_ensemble(self, inputs, *args, **kwargs):
        """Returns the mean distribution predicted by the ensemble for each input vector in inputs.
        Behavior is affected by the configuration of the model and dimensionality of inputs:
        
        prob(row)/ensemble(col)     | True  | False |
                            True    |   x   |       |       
                            False   |       |       |
        
        self._is_probabilistic and inputs 2D: returns (mean, variance) tensors with (num_networks, ...) per input
        self._is_probabilistic and inputs 3D: returns (mean, variance) tensors with (num_networks, ...), one model predicts per input dim
        not self._is_probabilistic and inputs 2D: returns mean tensor with (num_networks, ...) per input
        not self._is_probabilistic and inputs 3D: returns mean tensor with (num_networks, ...) one model predicts per input dim

        Arguments:
            inputs (np.ndarray): An array of input vectors in rows. See above for behavior.
        """

        if len(inputs.shape) == 2:
            if self.is_probabilistic:
                mean, var = self.sess.run(
                        [self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac],
                        feed_dict={self.sy_pred_in2d: inputs}
                    )
                # if self.loss_type=="MSPE":
                #     disagreement_penalty = np.exp(-10 * np.var(mean, axis=0)/self.scaler_out.cached_var)[None]
                #     var *= disagreement_penalty
                return mean, var
            else:
                return self.sess.run(
                            self.sy_pred_mean2d_fac,
                            feed_dict={self.sy_pred_in2d: inputs}
                        )
        else: #### inputs are 3d!
                if self.is_probabilistic:
                    return self.sess.run(
                        [self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac],
                        feed_dict={self.sy_pred_in3d: inputs}
                    )
                else:
                    return self.sess.run(
                            self.sy_pred_mean3d_fac,
                            feed_dict={self.sy_pred_in3d: inputs}
                        )

    def create_prediction_tensors(self, inputs, factored=False, *args, **kwargs):
        """See predict() above for documentation.
        """
        if self.is_probabilistic:
            factored_mean, factored_variance = self._compile_outputs(inputs, scale_output=True)
            if inputs.shape.ndims == 2 and not factored:
                mean = tf.reduce_mean(factored_mean, axis=0)
                variance = tf.reduce_mean(tf.square(factored_mean), axis=0) - tf.reduce_mean(tf.square(factored_mean), axis=0) \
                        + tf.reduce_mean(factored_variance, axis=0)
                # variance = tf.reduce_mean(tf.square(factored_mean - mean), axis=0) + \
                #         tf.reduce_mean(factored_variance, axis=0)
                return mean, variance
            return factored_mean, factored_variance
        else: 
            factored_mean = self._compile_outputs(inputs, scale_output=True)
            if inputs.shape.ndims == 2 and not factored:
                mean = tf.reduce_mean(factored_mean, axis=0)
                return mean
            return factored_mean
        

    def save(self, savedir, timestep):
        """Saves all information required to recreate this model in two files in savedir
        (or self.model_dir if savedir is None), one containing the model structure and the other
        containing all variables in the network.

        savedir (str): (Optional) Path to which files will be saved. If not provided, self.model_dir
            (the directory provided at initialization) will be used.
        """
        if not self.finalized:
            raise RuntimeError()
        model_dir = self.model_dir if savedir is None else savedir

        # Write structure to file
        with open(os.path.join(model_dir, '{}_{}.nns'.format(self.name, timestep)), "w+") as f:
            for layer in self.layers[:-1]:
                f.write("%s\n" % repr(layer))
            last_layer_copy = self.layers[-1].copy()
            last_layer_copy.set_activation(self.end_act_name)
            if self.is_probabilistic:
                last_layer_copy.set_output_dim(last_layer_copy.get_output_dim() // 2)
            else:
                last_layer_copy.set_output_dim(last_layer_copy.get_output_dim())
            f.write("%s\n" % repr(last_layer_copy))

        # Save network parameters (including scalers) in a .mat file
        var_vals = {}
        for i, var_val in enumerate(self.sess.run(self.nonoptvars + self.optvars)):
            var_vals[str(i)] = var_val
        savemat(os.path.join(model_dir, '{}_{}.mat'.format(self.name, timestep)), var_vals)

    def _load_structure(self):
        """Uses the saved structure in self.model_dir with the name of this network to initialize
        the structure of this network.
        """
        structure = []
        with open(os.path.join(self.model_dir, "%s.nns" % self.name), "r") as f:
            for line in f:
                kwargs = {
                    key: val for (key, val) in
                    [argval.split("=") for argval in line[3:-2].split(", ")]
                }
                kwargs["input_dim"] = int(kwargs["input_dim"])
                kwargs["output_dim"] = int(kwargs["output_dim"])
                kwargs["weight_decay"] = None if kwargs["weight_decay"] == "None" else float(kwargs["weight_decay"])
                kwargs["activation"] = None if kwargs["activation"] == "None" else kwargs["activation"][1:-1]
                kwargs["ensemble_size"] = int(kwargs["ensemble_size"])
                structure.append(FC(**kwargs))
        self.layers = structure

    #######################
    # Compilation methods #
    #######################

    def _compile_outputs(self, inputs, ret_log_var=False, raw_output=False, scale_output=False, debug=False):
        """Compiles the output of the network at the given inputs.

        If inputs is 2D, returns a 3D tensor where output[i] is the output of the ith network in the ensemble.
        If inputs is 3D, returns a 3D tensor where output[i] is the output of the ith network on the ith input matrix.

        Arguments:
            inputs: (tf.Tensor) A tensor representing the inputs to the network
            ret_log_var: (bool) If True, returns the log variance instead of the variance.

        Returns: (tf.Tensors) The mean and variance/log variance predictions at inputs for each network
            in the ensemble.
        """
        dim_output = self.layers[-1].get_output_dim()
        cur_out = inputs
        if self.use_scaler_in:
            cur_out = self.scaler_in.transform(cur_out)
            self.deb_scaler = self.scaler_in.transform(cur_out)

        if debug:
            self.layers_deb = []
        for layer in self.layers:
            cur_out = layer.compute_output_tensor(cur_out)
            if debug:
                self.layers_deb.append(cur_out)

        mean = cur_out[..., :dim_output//2] if self.is_probabilistic else cur_out
    
        if self.end_act is not None and not raw_output:
            mean = self.end_act(mean)

        if self.use_scaler_out and scale_output:
            mean = self.scaler_out.inverse_transform(mean)
            self.mean_deb = mean

        if self.is_probabilistic:
            # logvar = 5 * tf.tanh(cur_out[..., dim_output//2:]) + .1 * cur_out[..., dim_output//2:]
            logvar = cur_out[..., dim_output//2:]
            if self.use_scaler_out and scale_output:
                logvar = self.scaler_out.inverse_transform_logvar(logvar)

            if ret_log_var:
                var = logvar
            else:
                var = tf.exp(logvar)

            return mean, var

        else:
            return mean

    def _nll_loss(self, inputs, targets, inc_var_loss=True, tensor_loss=False, weights=None, oldpred_v=None, oldpred_var=None):
        """Helper method for compiling the NLL loss function.

        The loss function is obtained from the log likelihood, assuming that the output
        distribution is Gaussian, with both mean and (diagonal) covariance matrix being determined
        by network outputs.

        if inc_var_loss is False, becomes standard MSE

        Arguments:
            inputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            inc_var_loss: (bool) If True, includes log variance loss.
                            will throw an error if set to True in a model without variances included

        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """

        #### just for debugging        
        if tensor_loss: 
            if self.is_probabilistic:
                mean, _ = self._compile_outputs(inputs)
            else:
                mean = self._compile_outputs(inputs)
            return 0.5 * tf.square(mean - targets), mean

        #### weighting if provided
        if weights is None:
            weights = 1

        if self.is_probabilistic:
            mean, log_var = self._compile_outputs(inputs, ret_log_var=True)
            inv_var = tf.exp(-log_var)
        else: 
            mean, log_var = self._compile_outputs(inputs), 0.0
            inv_var = 1.0

        #### rescale targets if set to true
        if self.use_scaler_out:
            targets = self.scaler_out.transform(targets)

        #=====================================================================#
        #  Loss Clipping if provided                                          #
        #=====================================================================#

        ### clipped loss as Kullback leibler divergence constraint approximation
        if oldpred_v is not None:
            if oldpred_var is not None:
                old_var = oldpred_var
            else: 
                old_var = tf.reduce_mean(0.5 * (tf.square(oldpred_v - targets)))        ### overall empirical var

            kl_cliprange = tf.sqrt(self.kl_cliprange_ph*old_var)
            mean = oldpred_v + tf.clip_by_value(mean-oldpred_v, -tf.sqrt(2.0)*kl_cliprange, tf.sqrt(2.0)*kl_cliprange)
            
            if inc_var_loss:
                assert self.is_probabilistic
                varpred_cl = oldpred_var + tf.clip_by_value(tf.exp(log_var)-oldpred_var, -kl_cliprange, kl_cliprange)
            
                inv_var = 1/varpred_cl
                log_var = tf.log(varpred_cl)
            else:
                var_losses = 0
                inv_var = 1.0

            self.deb_cliprange = kl_cliprange

        if inc_var_loss:
            assert self.is_probabilistic
            var_losses = tf.reduce_mean(weights * tf.reduce_mean(0.5 * log_var, axis=-1), axis=-1) 
            
            #### debug
            self.deb_var_loss = var_losses
        else:            
            var_losses = 0
            inv_var = 1.0
            
        mse_losses = tf.reduce_mean(weights * tf.reduce_mean(0.5 * inv_var *  (tf.square(mean - targets)), axis=-1), axis=-1)
        total_losses = mse_losses + var_losses
        return total_losses

    def _mspe_loss(self, inputs, targets, inc_var_loss=True):
        """Helper method for compiling the MSPE loss function.

        Mean Squared Prediction error is essentially similar to MSE but aims to predict
        the MSE conditional on x. Outputs are thus probabilistic with slight abuse of the 
        maximum likelihood estimator on gaussians.
        The mean loss aims to minimize the MSE : E[(y-x)**2]
        The variance loss aims to minimize the MSE : E[(v-(y-x)**2)**2]
    
        Arguments:
            inputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            inc_var_loss: (bool) If True, includes log variance loss.
                            will throw an error if set to True in a model without variances included

        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """

        if self.is_probabilistic:
            mean, log_var = self._compile_outputs(inputs, ret_log_var=True)

        #### rescale targets if set to true
        if self.use_scaler_out:
            targets = self.scaler_out.transform(targets)

        var_pred = tf.exp(log_var)
        inv_var = tf.stop_gradient(var_pred)

        mse_losses_logit = tf.square(mean - targets)

        # predictor_var_logit = tf.stop_gradient(tf.math.reduce_variance(mean, axis=0))[tf.newaxis]
        # var_losses_logit = tf.square(var_pred - tf.stop_gradient(mse_losses_logit))
        # ratio = 0.1 * tf.reduce_mean(tf.stop_gradient(mse_losses_logit))/tf.reduce_mean(tf.stop_gradient(var_losses_logit))

        # mse_losses = tf.reduce_mean(tf.reduce_mean(mse_losses_logit, axis=-1),axis=-1)
        # var_losses = tf.reduce_mean(tf.reduce_mean(var_losses_logit * ratio, axis=-1), axis=-1)
        # # mean_logit_logvar = tf.stop_gradient(tf.reduce_mean(tf.reduce_mean(log_var, axis=0), axis=0)[tf.newaxis, tf.newaxis])
        # var_reg_loss_logit = tf.abs(log_var)
        # reg_ratio = 5e-3 * tf.reduce_mean(tf.stop_gradient(mse_losses_logit))/tf.reduce_mean(tf.stop_gradient(var_reg_loss_logit))
        # var_reg_loss = tf.reduce_mean(var_reg_loss_logit * reg_ratio)
        
        # total_losses = mse_losses + var_losses + var_reg_loss
        
        predictor_var_logit = tf.stop_gradient(tf.math.reduce_variance(mean, axis=0))[tf.newaxis]
        var_losses_logit = tf.square(var_pred - tf.stop_gradient(mse_losses_logit))
        ratio = 0.05 * tf.reduce_mean(tf.stop_gradient(mse_losses_logit))/tf.reduce_mean(tf.stop_gradient(var_losses_logit))

        mse_losses = tf.reduce_mean(tf.reduce_mean(mse_losses_logit, axis=-1),axis=-1)
        var_losses = tf.reduce_mean(tf.reduce_mean(var_losses_logit * ratio, axis=-1), axis=-1)
        # mean_logit_logvar = tf.stop_gradient(tf.reduce_mean(tf.reduce_mean(log_var, axis=0), axis=0)[tf.newaxis, tf.newaxis])
        var_reg_loss = 0.05 * tf.reduce_mean(tf.square(log_var))
        total_losses = mse_losses + var_losses + var_reg_loss
        return total_losses


    def _ce_loss(self, inputs, targets, tensor_loss=False, weights=None):
        """Helper method for compiling the NLL loss function.

        The loss function is obtained from the log likelihood, assuming that the output
        distribution is Gaussian, with both mean and (diagonal) covariance matrix being determined
        by network outputs.

        if inc_var_loss is False, becomes standard MSE

        Arguments:
            inputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            inc_var_loss: (bool) If True, includes log variance loss.

        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """

        assert not self.is_probabilistic

        if weights is not None:
            weights_tensor = tf.convert_to_tensor(weights, np.float32)
        else:
            weights_tensor = 1

        #### just for debugging        
        if tensor_loss: 
            mean = self._compile_outputs(inputs)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits = mean)
            loss = tf.reduce_mean(loss, axis=-2)

            return loss, tf.reduce_mean(mean, axis=-2)

        else:
            mean = self._compile_outputs(inputs, raw_output=True)
            total_losses = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits = mean)
            total_losses = tf.reduce_mean(total_losses, axis=-1)
        return total_losses



    def _huber_loss(self, inputs, targets, inc_var_loss=True, tensor_loss=False, weights=None, delta=1.0, scope=None):
        """Define a huber loss  https://en.wikipedia.org/wiki/Huber_loss
        Args:
            inputs: The input placeholders
            targets: The targets to hit
            inc_var_loss: True/False whether to include loss for variance
            tensor_loss: True/False whether to reduce losses over batch size. mainly for 
                debugging purposes to see which target parts generate most losses
            weights: a weight array with same dims as targets. if None, weights will be set to 1
            delta: The cutoff at which Huber loss turns linear
            scope: Optional scope for op_scope

        Returns: The huber loss op

        Huber loss:
        f(x) = if |x| <= delta:
                0.5 * x^2
            else:
                delta * |x| - 0.5 * delta^2
        
        """
        if self.is_probabilistic:
            mean, log_var = self._compile_outputs(inputs, ret_log_var=True)
            inv_var = tf.exp(-log_var)
        else: 
            mean = self._compile_outputs(inputs, ret_log_var=True)
            
        if weights is not None:
            weights_tensor = tf.convert_to_tensor(weights, np.float32)
        else:
            weights_tensor = 1

        with ops.name_scope(scope, "HuberL",
                            [mean, targets]) as scope:

            mean.get_shape().assert_is_compatible_with(targets.get_shape())
            # mean = math_ops.to_float(mean)
            # targets = math_ops.to_float(targets)
            #diff = math_ops.subtract(mean, targets)
            #abs_diff = tf.abs(diff)

            if tensor_loss: 
                total_losses = tf.reduce_mean(tf.where(tf.abs(targets-mean) < delta , 
                                                    0.5*tf.square(mean - targets),
                                                    delta*tf.abs(targets - mean) - 0.5*(delta**2)) * weights_tensor,
                                                axis=-2)
                return total_losses, mean

            if inc_var_loss:
                assert self.is_probabilistic
                
                mean_losses = tf.reduce_mean(tf.reduce_mean(tf.where(tf.abs(targets-mean) < delta , 
                                                    0.5*tf.square(mean - targets),
                                                    delta*tf.abs(targets - mean) - 0.5*(delta**2)) * inv_var * weights_tensor,
                                                axis=-1), axis=-1)
                var_losses = tf.reduce_mean(tf.reduce_mean(0.5 * log_var * weights_tensor, axis=-1), axis=-1)
                total_losses = mean_losses + var_losses
            else:
                total_losses = tf.reduce_mean(tf.reduce_mean(tf.where(tf.abs(targets-mean) < delta , 
                                                    0.5*tf.square(mean - targets),
                                                    delta*tf.abs(targets - mean) - 0.5*(delta**2)) * weights_tensor,
                                                axis=-1), axis=-1)

            return total_losses

# huber loss
def huber(true, pred, delta):
    loss = tf.where(tf.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*tf.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)
