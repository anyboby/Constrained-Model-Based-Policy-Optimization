"""

Some simple logging functionality, vendored from Spinning Up's utilities.

"""
import json
import joblib
import shutil
import numpy as np
import time
import tensorflow as tf
from copy import deepcopy
import os.path as osp, time, atexit, os
from utilities.mpi_tools import proc_id, mpi_statistics_scalar
from utilities.serialization_utils import convert_json
from envs.utils import get_env_from_params
# from softlearning.environments.gym import mujoco_safety_gym

from tensorflow.python.lib.io import file_io        ###@anyboby not good!
import traceback
from gym import wrappers

DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__)))),'data')

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)
    sess = tf.keras.backend.get_session()
    
    #sess = tf.Session(graph=tf.Graph())
    
    saver = Saver()
    model = saver.restore_tf_graph(sess, fpath)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x})

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        environment_params = {}
        environment_params['universe'] = 'gym'
        environment_params['task'] = 'v2'
        environment_params['domain'] = 'HumanoidSafe'
        environment_params['kwargs'] = {}
        env = get_env_from_params(environment_params)
        # env = wrappers.Monitor(env, '/home/uvday/ray_mbpo/AntSafe/', force = True)

    return env, get_action, sess

class Saver:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def init_saver(self, scope):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.saver = tf.train.Saver(
            var_list= var_list,
            sharded=True,
            allow_empty=True)
        self.builder = None        

    def restore_tf_graph(self, sess, fpath):
        """
        Loads graphs saved by Logger.

        Will output a dictionary whose keys and values are from the 'inputs' 
        and 'outputs' dict you specified with logger.setup_tf_saver().

        Args:
            sess: A Tensorflow session.
            fpath: Filepath to save directory.

        Returns:
            A dictionary mapping from keys to tensors in the computation graph
            loaded from ``fpath``. 
        """
        tf.saved_model.loader.load(
                    sess,
                    [tf.saved_model.tag_constants.SERVING],
                    fpath
                )
        model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
        graph = sess.graph #tf.get_default_graph()
        model = dict()
        model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
        model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})
        return model

    def save_config(self, config, config_dir, exp_name='CPO_config'):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if exp_name is not None:
            config_json['exp_name'] = exp_name
        if proc_id()==0:
            output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(config_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, output_dir, itr=None):
        """

        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            output_dir: the target directory

            itr: An int, or None. Current iteration of training.

        Return:
            state_path: returns the full path to the saved file (the target 
            directory + extension)
        """

        ### dump state
        fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
        state_path = osp.join(output_dir, fname)
        try:
            joblib.dump(state_dict, state_path)
        except:
            print('Warning: could not pickle state_dict.')

        return state_path
                                
    def save_tf (self, sess, inputs, outputs, output_dir, shards=1):
        """
        Uses simple_save to save a trained model, plus info to make it easy
        to associate tensors to variables after restore. 

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.

            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the 
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!

            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.

            output_dir: the target directory

            #itr: An int, or None. Current iteration of training.

        Returns: 
            fpath: the path to the saved tf model

            model_info_path('*.pkl'): the path to the saved model_info dict
        """
        ### save tf
        tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
        tf_saver_info = {'inputs': {k:v.name for k,v in inputs.items()},
                                'outputs': {k:v.name for k,v in outputs.items()}}

        fpath = ''#'simple_save' + ('%d'%itr if itr is not None else '')
        fpath = osp.join(output_dir, fpath)
        if osp.exists(fpath):
            # simple_save refuses to be useful if fpath already exists,
            # so just delete fpath if it's there.
            shutil.rmtree(fpath)
        
        ##### @anyboby saving with builder since simple_save seemed to increase
            #    chkpt size by adding save op every time
        try:
            builder = self._maybe_create_builder(self.builder, sess, fpath, inputs, outputs)
            builder.save(as_text=False)
            if self.verbose:
                print("  SavedModel graph written successfully. " )
            success = True
        except Exception as e:
            print("       WARNING::SavedModel write FAILED. " )
            traceback.print_tb(e.__traceback__)
            success = False

        #tf.saved_model.simple_save(export_dir=fpath, **tf_saver_elements)
        
        ### save model info
        model_info_path = osp.join(fpath, 'model_info.pkl')
        joblib.dump(tf_saver_info, model_info_path)

        return fpath, model_info_path, success

    def _maybe_create_builder(self, builder, sess, export_dir, inputs, outputs):
        """
        hacky, but doesn't create a new savedmodelbuilder witch each call, but instead 
        overwrites export_dir in the SavedModelBuilder. 
        """
        if builder:
            if file_io.file_exists(export_dir):
                if file_io.list_directory(export_dir):
                    raise AssertionError(
                        "Export directory already exists, and isn't empty. Please choose "
                        "a different export directory, or delete all the contents of the "
                        "specified directory: %s" % export_dir)
            else:
                file_io.recursive_create_dir(export_dir)
            
            builder._export_dir = export_dir
            return builder
        else:
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
            signature_def_map = {
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(inputs, outputs)
            }
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
            builder.add_meta_graph_and_variables(sess,
                                            tags= [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map=signature_def_map,
                                            assets_collection=assets_collection,
                                            saver=self.saver)
        return builder

class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        # if proc_id()==0:
        #     self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
        #     if osp.exists(self.output_dir):
        #         print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        #     else:
        #         os.makedirs(self.output_dir)
        #     self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        #     atexit.register(self.output_file.close)
        #     print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))
        # else:
        #     self.output_dir = None
        #     self.output_file = None
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id()==0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    
    def dump_tabular(self, output_dir, print_out=True):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.

        Returns the current dictionary, if needed for other diagnostic 
        purposes. 

        Be sure to log all diagnostics you want before calling this!

        Args:
            fpath: path to the output directory

            print_out: set to False if you don't need the prints
        Returns:
            current_diagnostics: dictionary of the current diagnostics status
        """
        if proc_id()==0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15,max(key_lens))
            keystr = '%'+'%d'%max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            if print_out:
                print("-"*n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g"%val if hasattr(val, "__float__") else val
                if print_out:
                    print(fmt%(key, valstr))
                vals.append(val)
            if print_out:
                print("-"*n_slashes, flush=True)

            output_file = open(osp.join(output_dir, 'diagnostics.txt'), 'w')
            if self.first_row:
                output_file.write("\t".join(self.log_headers)+"\n")
            output_file.write("\t".join(map(str,vals))+"\n")
            output_file.flush()
            output_file.close()

        current_diagnostics = deepcopy(self.log_current_row)
        self.log_current_row.clear()
        self.first_row=False
        return current_diagnostics

class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use 

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        @anyboby: this should eventually be replaced by an overall diagnostics method !!!
        
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            if v:
                vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
                stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
                super().log_tabular(key if average_only else key+'Average', stats[0])
                if not(average_only):
                    super().log_tabular(key+'Std', stats[1])
                if with_min_and_max:
                    super().log_tabular(key+'Max', stats[3])
                    super().log_tabular(key+'Min', stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return mpi_statistics_scalar(vals)


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=True):

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs