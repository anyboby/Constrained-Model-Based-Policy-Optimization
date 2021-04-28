import os
from gym import utils
from mujoco_safety_gym.envs.fetch_env import FetchEnvNew


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(FetchEnvNew, utils.EzPickle):
    def __init__(self, reward_type='sparse', additional_objects=False, number_of_objects=5):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.]          
        }
        FetchEnvNew.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.10, target_range=0.15, distance_threshold=0.05, additional_objects=additional_objects,
            number_of_objects = number_of_objects, initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
