from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *

# from paramUtil import *
from paramUtil import t2m_raw_offsets as smplx_raw_offsets
from paramUtil import t2m_body_hand_kinematic_chain as smplx_full_kinematic_chain

import torch
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

example_id = "000021"
# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# body,hand joint idx
# 2*3*5=30, left first, then right
hand_joints_id = [i for i in range(25, 55)]
body_joints_id = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
]  # 22 joints
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 52
n_raw_offsets = torch.from_numpy(smplx_raw_offsets)
kinematic_chain = smplx_full_kinematic_chain


from human_body_prior.body_model.body_model import BodyModel

male_bm_path = "./smplx/SMPLX_NEUTRAL.npz"
male_bm = BodyModel(bm_fname=male_bm_path).to("cuda").eval()


def findAllFile(base):
    """
    Recursively find all files in the specified directory.

    Args:
        base (str): The base directory to start the search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def uniform_skeleton(positions, target_offset):
    """
    Uniformly scales a skeleton to match a target offset.

    Args:
        positions (numpy.ndarray): Input skeleton joint positions.
        target_offset (torch.Tensor): Target offset for the skeleton.

    Returns:
        numpy.ndarray: New joint positions after scaling and inverse/forward kinematics.
    """
    # Creating a skeleton with a predefined kinematic chain
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")

    # Calculate the global offset of the source skeleton
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Calculate Scale Ratio as the ratio of legs
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    # Scale ratio for uniform scaling
    scale_rt = tgt_leg_len / src_leg_len

    # Extract the root position of the source skeleton
    src_root_pos = positions[:, 0]
    # Scale the root position based on the calculated ratio
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse Kinematics to get quaternion parameters
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    # Forward Kinematics with the new root position and target offset
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions, feet_thre, tgt_offsets=None):
    # (seq_len, joints_num, 3)

    """Uniform Skeleton"""
    positions = uniform_skeleton(positions, tgt_offsets)

    """Put on Floor"""
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    """XZ at origin"""
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    """All initially face Z+"""
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = (
        forward_init / np.sqrt((forward_init**2).sum(axis=-1))[..., np.newaxis]
    )

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    """New ground truth positions"""
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    """Quaternion and Cartesian representation"""
    r_rot = None

    def get_rifke(positions):
        """Local pose"""
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        """All pose face Z+"""
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions
        )
        return positions

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True
        )

        """Quaternion to continuous 6D"""
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        """Root Linear Velocity"""
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        """Root Angular Velocity"""
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    """Root height"""
    root_y = positions[:, 0, 1:2]

    """Root rotation and linear velocity"""
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    """Get Joint Rotation Representation"""
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    """Get Joint Rotation Invariant Position Represention"""
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    """Get Joint Velocity Representation"""
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    """Add Y-axis rotation to local joints"""
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    """Concate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def pose168_to_322(path):
    pose = np.load(path)
    n = pose.shape[0]
    pose_trans = pose[:, :3]

    pose_root = pose[:, 3:6]
    pose_body = pose[:, 6:69]
    pose_jaw = pose[:, 69:72]

    pose_expression = np.zeros((n, 50))
    pose_face_shape = np.zeros((n, 100))

    pose_hand = pose[:, -90:]

    pose_body_shape = np.zeros((n, 10))

    pose2 = np.concatenate(
        [
            pose_root,
            pose_body,
            pose_hand,
            pose_jaw,
            pose_expression,
            pose_face_shape,
            pose_trans,
            pose_body_shape,
        ],
        -1,
    )

    return pose2


def motionx2positions(
    path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    pose = torch.Tensor(np.load(path)).to(device)

    if pose.shape[0] > 4000:
        body = male_bm(
            pose_body=pose[:4000, 3 : 3 + 63],
            pose_hand=pose[:4000, 66 : 66 + 90],
            pose_jaw=pose[:4000, 66 + 90 : 66 + 93],
            betas=torch.zeros((4000, 10)).to(device),
            root_orient=pose[:4000, :3],
        )
        joint_loc1 = body.Jtr + pose[:4000, 309 : 309 + 3][:, None, :]
        joint_loc1 = joint_loc1.cpu().numpy()

        body = male_bm(
            pose_body=pose[4000:, 3 : 3 + 63],
            pose_hand=pose[4000:, 66 : 66 + 90],
            pose_jaw=pose[4000:, 66 + 90 : 66 + 93],
            betas=torch.zeros((pose.shape[0] - 4000, 10)).to(device),
            root_orient=pose[4000:, :3],
        )
        joint_loc2 = body.Jtr + pose[4000:, 309 : 309 + 3][:, None, :]
        joint_loc2 = joint_loc2.cpu().numpy()

        return np.concatenate([joint_loc1, joint_loc2], 0)

    body = male_bm(
        pose_body=pose[:, 3 : 3 + 63],
        pose_hand=pose[:, 66 : 66 + 90],
        pose_jaw=pose[:, 66 + 90 : 66 + 93],
        betas=torch.zeros((pose.shape[0], 10)).to(device),
        root_orient=pose[:, :3],
    )
    joint_loc = body.Jtr + pose[:, 309 : 309 + 3][:, None, :]
    joint_loc = joint_loc.cpu().numpy()

    return joint_loc


def beatx2positions(
    path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    beat_bm = (
        BodyModel(bm_fname=male_bm_path, num_betas=300, num_expressions=100)
        .to("cuda")
        .eval()
    )

    beat_motion = np.load(path)
    pose_trans = torch.Tensor(beat_motion["trans"]).to(device)
    pose = torch.Tensor(beat_motion["poses"]).to(device)
    n = pose_trans.shape[0]
    pose_expression = torch.Tensor(beat_motion["expressions"]).to(device)
    pose_body_shape = torch.Tensor(beat_motion["betas"]).to(device)

    if pose.shape[0] > 4000:

        joint_locs = []
        for s, e in enumerate(range(0, n, 4000)):

            body = beat_bm(
                pose_body=pose[s * 4000 : e + 4000, 3 : 3 + 63],
                pose_hand=pose[s * 4000 : e + 4000, 66 : 66 + 90],
                pose_jaw=pose[s * 4000 : e + 4000, 66 + 90 : 66 + 93],
                expression=pose_expression[s * 4000 : e + 4000],
                betas=pose_body_shape.repeat(pose[s * 4000 : e + 4000].shape[0], 1),
                root_orient=pose[s * 4000 : e + 4000, :3],
            )
            joint_loc = body.Jtr + pose_trans[s * 4000 : e + 4000][:, None, :]
            joint_loc = joint_loc.cpu().numpy()
            joint_locs.append(joint_loc)

        return np.concatenate(joint_locs, 0)

    body = beat_bm(
        pose_body=pose[:, 3 : 3 + 63],
        pose_hand=pose[:, 66 : 66 + 90],
        pose_jaw=pose[:, 66 + 90 : 66 + 93],
        expression=pose_expression,
        betas=pose_body_shape.repeat(pose.shape[0], 1),
        root_orient=pose[:, :3],
    )
    joint_loc = body.Jtr + pose_trans[:, None, :]
    joint_loc = joint_loc.cpu().numpy()

    return joint_loc


"""
For HumanML3D Dataset
"""

if __name__ == "__main__":
    """
    This script processes motion capture data, performs a recovery operation on the joint positions,
    and saves the recovered joint positions along with the original joint vectors. The main steps include:

    Note: Exception handling is implemented to identify and print any issues encountered during processing.

    Output:
    - Recovered joint positions are saved in the 'new_joints' directory.
    - Original joint vectors are saved in the 'new_joint_vecs' directory.
    """

    data_dir = "/srv/hays-lab/scratch/sanisetty3/motionx/motion_data/smplx_322"
    save_dir1 = "/srv/hays-lab/scratch/sanisetty3/motionx/motion_data/new_joints/"
    save_dir2 = "/srv/hays-lab/scratch/sanisetty3/motionx/motion_data/new_joint_vecs/"

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    example_data = np.array(
        motionx2positions(
            "/srv/hays-lab/scratch/sanisetty3/motionx/motion_data/smplx_322/humanml/000021.npy"
        )
    )
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)[:, body_joints_id + hand_joints_id, :]

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    source_list = findAllFile(
        "/srv/hays-lab/scratch/sanisetty3/motionx/motion_data/smplx_322/"
    )
    frame_num = 0
    for source_file in tqdm(source_list):

        save_path1 = pjoin(save_dir1, source_file.replace(data_dir + "/", ""))
        save_path2 = pjoin(save_dir2, source_file.replace(data_dir + "/", ""))

        if os.path.exists(save_path1) and os.path.exists(save_path2):
            continue

        try:
            source_data = motionx2positions(os.path.join(data_dir, source_file))[
                :, body_joints_id + hand_joints_id, :
            ]
            data, ground_positions, positions, l_velocity = process_file(
                source_data, 0.002
            )
            rec_ric_data = recover_from_ric(
                torch.from_numpy(data).unsqueeze(0).float(), joints_num
            )

            os.makedirs(os.path.dirname(save_path1), exist_ok=True)
            os.makedirs(os.path.dirname(save_path2), exist_ok=True)

            np.save(save_path1, rec_ric_data.squeeze().numpy())
            np.save(save_path2, data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print(
        "Total clips: %d, Frames: %d, Duration: %fm"
        % (len(source_list), frame_num, frame_num / 30 / 60)
    )
