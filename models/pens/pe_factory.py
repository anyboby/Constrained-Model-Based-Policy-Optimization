import numpy as np
import numpy.ma as ma
import tensorflow as tf

import copy
from .fc import FC
from .pe import PE

def build_PE(in_dim, 
					out_dim,
					name='BNN',
					hidden_dims=(200, 200, 200), 
					num_networks=7, 
					num_elites=5,
					loss = 'MSPE', 
					activation = 'swish',
					output_activation = None,
					decay=1e-4,
					lr = 1e-3,
					lr_decay = None,
					decay_steps=None,
					use_scaler_in = False,
					use_scaler_out = False,
					clip_loss = False,
					kl_cliprange = 0.1,
					max_logvar = .5,
					min_logvar = -6,
					session=None):
	"""
	Constructs a tf probabilistic ensemble model.
	Args:
		loss: Choose from 'MSPE', 'NLL', 'MSE', 'Huber', or 'CE'. 
				choosing MSPE or NLL will construct a model with variance output
	"""
	print('[PE] dim in / out: {} / {} | Hidden dim: {}'.format(in_dim, out_dim, hidden_dims))
	#print('[ BNN ] Input Layer dim: {} | Output Layer dim: {} '.format(obs_dim_in+act_dim+prior_dim, obs_dim_out+rew_dim))
	params = {'name': name, 
				'loss':loss, 
				'num_networks': num_networks, 
				'num_elites': num_elites, 
				'sess': session,
				'use_scaler_in': use_scaler_in,
				'use_scaler_out': use_scaler_out,
				'clip_loss': clip_loss,
				'kl_cliprange':kl_cliprange,
				'max_logvar':max_logvar,
				'min_logvar':min_logvar,
				}
	model = PE(params)
	model.add(FC(hidden_dims[0], input_dim=in_dim, activation=activation, weight_decay=decay/4))	# def dec: 0.000025))
	
	for hidden_dim in hidden_dims[1:]:
		model.add(FC(hidden_dim, activation=activation, weight_decay=decay/2))						# def dec: 0.00005))
	
	model.add(FC(out_dim, activation=output_activation, weight_decay=decay))						# def dec: 0.0001
	
	opt_params = {"learning_rate":lr} if lr_decay is None else {"learning_rate":lr, 
																"learning_rate_decay":lr_decay,
																"decay_steps":decay_steps}
	model.finalize(tf.train.AdamOptimizer, opt_params, lr_decay=lr_decay)

	total_parameters = 0
	for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name):
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	print('[ Probabilistic Ensemble ] Total trainable Parameteres: {} '.format(total_parameters))

	return model

def format_samples_for_dyn(samples, append_r=True, append_c=False, noise=None):
	"""
	formats samples to fit training, specifically returns: 
	inputs, outputs:

	inputs = np.concatenate((observations, act, priors), axis=-1)
	outputs = np.concatenate(delta_observations, rewards ,costs), axis=-1)  

	where rewards and costs are optional
	"""
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	terms = np.squeeze(samples['terminals'])[..., None]

	delta_obs = next_obs - obs

	#### ----END preprocess samples for model training in safety gym -----####
	inputs = np.concatenate((obs, act), axis=-1)
	
	outputs = delta_obs
	
	if append_r:
		rew = np.squeeze(samples['rewards'])[..., None]
		outputs = np.concatenate((outputs, rew), axis=-1)
	
	if append_c:
		costs = np.squeeze(samples['costs'])[..., None]
		outputs = np.concatenate((outputs, costs), axis=-1)

	# add noise
	if noise:
		inputs = _add_noise(inputs, noise)		### noise helps (sometimes)

	return inputs, outputs


### @anyboby, try to include this in the model rather than separately
def format_samples_for_cost(samples, oversampling=False, one_hot = True, num_classes=2, noise=None):
	"""
	formats samples to fit training for cost, specifically returns: 
	(obs, act, next_obs)

	Args:
		one_hot: determines whether targets are structured as classification or regression
					one_hot: True will output targets with shape [batch_size, num_classes]
					one_hot: False wil output targets with shape [batch_size,] and scalar targets
	"""
	next_obs = samples['next_observations']
	obs = samples['observations']
	cost = samples['costs']
	act = samples['actions']

	if one_hot:
		cost_one_hot = np.zeros(shape=(len(cost), num_classes))
		batch_indcs = np.arange(0, len(cost))
		costs = cost.astype(int)
		cost_one_hot[(batch_indcs, costs)] = 1
		outputs = cost_one_hot
	else:
		outputs = cost[:, None]

	inputs = np.concatenate((obs, act, next_obs), axis=-1)
	## ________________________________ ##
	##      oversample cost classes     ##
	## ________________________________ ##
	if oversampling:
		if len(outputs[np.where(costs>0)[0]])>0:
			imbalance_ratio = len(outputs[np.where(costs==0)[0]])//len(outputs[np.where(costs>0)[0]])
			extra_outputs = np.tile(outputs[np.where(costs>0)[0]], (1+imbalance_ratio//3,1))		## don't need to overdo it
			outputs = np.concatenate((outputs, extra_outputs), axis=0)
			extra_inputs = np.tile(inputs[np.where(costs>0)[0]], (1+imbalance_ratio//3,1))
			extra_inputs = _add_noise(extra_inputs, 0.0001)
			inputs = np.concatenate((inputs, extra_inputs), axis=0)
	
	### ______ add noise _____ ###
	if noise:
		inputs = _add_noise(inputs, noise)		### noise helps 
	
	return inputs, outputs

def _add_noise(data_inp, noiseToSignal):
    data= copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis = 0)
    std_of_noise = mean_data*noiseToSignal
    for j in range(mean_data.shape[0]):
        if(std_of_noise[j]>0):
            data[:,j] = np.copy(data[:,j]+np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data

def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))
