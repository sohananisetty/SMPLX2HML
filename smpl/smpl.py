# This code is based on https://github.com/Mathux/ACTOR.git
import contextlib

import numpy as np
import torch
from smplx import SMPLLayer as _SMPLLayer
from smplx.lbs import vertices2joints
from utils.smpl_config import JOINT_REGRESSOR_TRAIN_EXTRA, SMPL_MODEL_PATH

SMPLX_JOINT_NAMES = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "jaw": 22,
    "left_eye_smplhf": 23,
    "right_eye_smplhf": 24,
    "left_index1": 25,
    "left_index2": 26,
    "left_index3": 27,
    "left_middle1": 28,
    "left_middle2": 29,
    "left_middle3": 30,
    "left_pinky1": 31,
    "left_pinky2": 32,
    "left_pinky3": 33,
    "left_ring1": 34,
    "left_ring2": 35,
    "left_ring3": 36,
    "left_thumb1": 37,
    "left_thumb2": 38,
    "left_thumb3": 39,
    "right_index1": 40,
    "right_index2": 41,
    "right_index3": 42,
    "right_middle1": 43,
    "right_middle2": 44,
    "right_middle3": 45,
    "right_pinky1": 46,
    "right_pinky2": 47,
    "right_pinky3": 48,
    "right_ring1": 49,
    "right_ring2": 50,
    "right_ring3": 51,
    "right_thumb1": 52,
    "right_thumb2": 53,
    "right_thumb3": 54,
}

NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

class SMPL(_SMPLLayer):
    """Extension of the official SMPL implementation to support more joints"""

    def __init__(self, model_path=SMPL_MODEL_PATH, **kwargs):
        kwargs["model_path"] = model_path

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(**kwargs)

        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            "J_regressor_extra", torch.tensor(J_regressor_extra, dtype=torch.float32)
        )
        smpl_indexes = np.arange(24)

        self.maps = {
            "smpl": smpl_indexes,
        }

    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        all_joints = torch.cat([smpl_output.joints, extra_joints], dim=1)

        output = {"vertices": smpl_output.vertices}

        for joinstype, indexes in self.maps.items():
            output[joinstype] = all_joints[:, indexes]

        return output
