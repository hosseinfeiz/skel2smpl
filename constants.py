"""SMPL-22 topology + source-skeleton name maps for the skeleton→SMPL fit.

The SMPL-standard 22-joint kinematic tree (the body joints, no hands/face), the
Z-up↔Y-up frame rotation, and the BVH/Mixamo source-joint name maps used by the
input adapters. Vendored from koopman `constants.py` / `motion_bridge.py` /
`preprocess_bridges.py` — only the SMPL-standard bits (no koopman/Ninjutsu pairing
or torus constants).
"""

import torch

# SMPL-22 body joints in canonical order (pelvis-rooted kinematic tree).
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

N_JOINTS = 22
HEAD_INDEX = 15

# parents[j] = parent joint index (parents[0] = -1 for the root).
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Kinematic edges (parent, child) over the movable joints — used by the limb-length
# shape solve.
_SMPL22_BONES = [(int(SMPL_PARENTS[j]), j) for j in range(1, N_JOINTS)]

# Joints whose ARRIVING bone length is corrupted by the BVH 5-spine→SMPL 3-spine
# topology mismatch (downweighted in the shape data term). §P9/§P12.
SPINE_NECK_COLLAR = (3, 6, 9, 12, 13, 14)

# Z-up → Y-up (Rx(-90°)); SMPL.forward applies this in reverse to return Z-up.
# Identical to smpl._TRANS; the native↔proper frame conjugation pivots on it.
_RX_NEG90 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# Primary child per SMPL-22 joint (the bone whose direction defines the rest frame);
# -1 = leaf. Used by the BVH-rest Procrustes alignment.
SMPL22_PRIMARY_CHILD = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17,
                        -1, 18, 19, 20, 21, -1, -1]

# SMPL-22 joint name -> BVH/Mixamo source joint name (rotation retarget).
SMPL_TO_BVH = {
    'pelvis': 'Hips', 'left_hip': 'LeftUpLeg', 'right_hip': 'RightUpLeg',
    'spine1': 'Spine', 'left_knee': 'LeftLeg', 'right_knee': 'RightLeg',
    'spine2': 'Spine2', 'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot',
    'spine3': 'Spine3', 'left_foot': 'LeftToeBase', 'right_foot': 'RightToeBase',
    'neck': 'Neck', 'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder',
    'head': 'Head', 'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm',
    'left_elbow': 'LeftForeArm', 'right_elbow': 'RightForeArm',
    'left_wrist': 'LeftHand', 'right_wrist': 'RightHand',
}

# SMPL-22 joint name -> mocap *_worldpos.csv joint name (position ingest).
SMPL_TO_MOCAP = {
    'pelvis': 'Hips',
    'left_hip': 'LeftUpLeg', 'right_hip': 'RightUpLeg',
    'spine1': 'Spine',
    'left_knee': 'LeftLeg', 'right_knee': 'RightLeg',
    'spine2': 'Spine2',
    'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot',
    'spine3': 'Spine3',
    'left_foot': 'LeftToeBase', 'right_foot': 'RightToeBase',
    'neck': 'Neck',
    'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder',
    'head': 'Head',
    'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm',
    'left_elbow': 'LeftForeArm', 'right_elbow': 'RightForeArm',
    'left_wrist': 'LeftHand', 'right_wrist': 'RightHand',
}
