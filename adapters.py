"""Input adapters: BVH/mocap → SMPL-22 observations for the fitter.

Generic mocap I/O vendored from koopman `motion_bridge.py` / `preprocess_bridges.py`
and `smpl_fit.compute_Q`. The §P11 convention-correct rotation retarget (pure
conjugation) and the rigid native→proper map are kept; the koopman-specific
proper-frame `Fp`/convrot glue does NOT live here.

  compute_Q                    — rigid SMPL-native→proper map Q = Rx(+90)·Mᵀ·Rx(+90)
  bvh_to_smpl_world_rotations  — BVH source rotations → SMPL-native world rotations
  body_to_world_transforms     — head-frame body-log + head SE(3) → world joint transforms
  _read_worldpos / _smpl22_positions — *_worldpos.csv → [T,22,3] Z-up metres
"""

import csv
import pickle

import torch

from skel2smpl.constants import (
    N_JOINTS, JOINT_NAMES, SMPL_TO_BVH, SMPL_TO_MOCAP, SMPL22_PRIMARY_CHILD,
    _RX_NEG90,
)
from skel2smpl.geometry import (
    exp_so3, se3_exp, se3_compose, project_se3, rotation_matrix_to_axis_angle,
)

MM_TO_M = 0.001


def body_to_world_transforms(body_log, head_global, device, head_index: int = 15):
    """[T,132] head-frame body-log + [T,4,4] head world SE(3) → [T,N_JOINTS,4,4]
    world joint transforms (joint position = T_world[...,:3,3])."""
    n_frames = body_log.shape[0]
    body_log = body_log.to(device).double()
    head_global = head_global.to(device).double()

    T_rel = se3_exp(body_log.reshape(-1, 6)).reshape(n_frames, N_JOINTS, 4, 4)
    eye = torch.eye(4, device=device, dtype=T_rel.dtype).unsqueeze(0).repeat(n_frames, 1, 1)
    T_rel[:, head_index] = eye

    head_exp = head_global.unsqueeze(1).expand(-1, N_JOINTS, -1, -1)
    T_world = se3_compose(
        head_exp.reshape(-1, 4, 4), T_rel.reshape(-1, 4, 4)
    ).reshape(n_frames, N_JOINTS, 4, 4)
    T_world = project_se3(T_world.reshape(-1, 4, 4)).reshape(n_frames, N_JOINTS, 4, 4)
    return T_world.cpu()


def parse_bvh_skeleton(bvh_path):
    """BVH HIERARCHY → (names[list, DFS order], parent[dict: name->parent|None]).
    End Sites carry no rotation channels and are skipped."""
    names, parent, stack = [], {}, []
    with open(bvh_path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            kw = tok[0]
            if kw in ('ROOT', 'JOINT'):
                nm = tok[1]
                parent[nm] = stack[-1] if stack else None
                names.append(nm)
                stack.append(nm)
            elif kw == 'End':                 # 'End Site' — push placeholder leaf
                stack.append(None)
            elif kw == '}':
                if stack:
                    stack.pop()
            elif kw == 'MOTION':
                break
    return names, parent


def _read_euler_csv(csv_path):
    """*_rotations.csv → {joint: [T,3] tensor} Euler degrees in (X,Y,Z) channel order."""
    with open(csv_path, newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [[float(v) for v in r] for r in rdr if r and r[0] != '']
    cols = {}
    for i, h in enumerate(header):
        if '.' not in h:
            continue
        name, axis = h.rsplit('.', 1)
        cols.setdefault(name, {})[axis] = i
    data = torch.tensor(rows, dtype=torch.float64)
    out = {}
    for name, ax in cols.items():
        if {'X', 'Y', 'Z'} <= set(ax):
            out[name] = data[:, [ax['X'], ax['Y'], ax['Z']]]
    return out


def _euler_to_R(euler_deg, order='XYZ'):
    """[...,3] Euler degrees → [...,3,3], composed R = R[o0] @ R[o1] @ R[o2]
    (intrinsic). BVH channel order is Xrotation Yrotation Zrotation ⇒ 'XYZ'."""
    e = torch.deg2rad(torch.as_tensor(euler_deg, dtype=torch.float64))
    axis_idx = {'X': 0, 'Y': 1, 'Z': 2}
    R = None
    for k, ch in enumerate(order):
        ax = torch.zeros(e.shape[:-1] + (3,), dtype=e.dtype)
        ax[..., axis_idx[ch]] = e[..., k]
        Rk = exp_so3(ax)
        R = Rk if R is None else R @ Rk
    return R


def source_world_rotations(bvh_path, rot_csv, euler_order='XYZ'):
    """Forward-kinematics the BVH source skeleton → {joint: [T,3,3]} world
    rotations in the BVH (Y-up) frame. Positions are NOT needed (rotation FK is
    offset-independent)."""
    names, parent = parse_bvh_skeleton(bvh_path)
    euler = _read_euler_csv(rot_csv)
    T = next(iter(euler.values())).shape[0]
    Rw = {}
    for nm in names:                          # DFS order ⇒ parent computed first
        if nm in euler:
            Rl = _euler_to_R(euler[nm], euler_order)
        else:
            Rl = torch.eye(3, dtype=torch.float64).expand(T, 3, 3)
        p = parent[nm]
        Rw[nm] = Rl if p is None else Rw[p] @ Rl
    return Rw


def _source_rest_positions(bvh_path, offsets_pkl):
    """BVH rest-pose joint positions {name:[3]} via offset FK (zero rotation).
    Frame-only (units cancel under normalization)."""
    names, parent = parse_bvh_skeleton(bvh_path)
    off = pickle.load(open(offsets_pkl, 'rb'))
    rest = {}
    for nm in names:
        o = torch.tensor(off[nm], dtype=torch.float64) if nm in off \
            else torch.zeros(3, dtype=torch.float64)
        p = parent[nm]
        rest[nm] = o if p is None else rest[p] + o
    return rest


def _rest_frame_alignment(srest, rest_smpl):
    """Single global rotation [3,3] mapping source-rest bone directions to SMPL-rest
    bone directions (Procrustes over the primary bones). Reconciles the BVH-native vs
    SMPL-native frames so a source world rotation can be transported as M·R·Mᵀ."""
    import torch.nn.functional as F
    A, B = [], []
    for j in range(N_JOINTS):
        c = SMPL22_PRIMARY_CHILD[j]
        if c < 0:
            continue
        s, sc = SMPL_TO_BVH[JOINT_NAMES[j]], SMPL_TO_BVH[JOINT_NAMES[c]]
        A.append(F.normalize((srest[sc] - srest[s]).unsqueeze(0), dim=-1)[0])
        B.append(F.normalize((rest_smpl[c] - rest_smpl[j]).unsqueeze(0), dim=-1)[0])
    A, B = torch.stack(A), torch.stack(B)
    U, S, Vt = torch.linalg.svd(A.transpose(-1, -2) @ B)
    d = torch.sign(torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    return Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)


def bvh_to_smpl_world_rotations(bvh_path, rot_csv, offsets_pkl, smpl_model,
                                euler_order='XYZ'):
    """Convention-correct BVH→SMPL world-rotation retarget — NO position fitting.

    Builds the SMPL skeleton purely from the SOURCE joint ROTATIONS by PURE
    CONJUGATION  R_smpl_world_j(t) = M · R_src_world_b(t) · Mᵀ  where b = SMPL_TO_BVH[j]
    and M = _rest_frame_alignment(source-rest, SMPL-rest). This transports the source's
    anatomically-correct LOCAL joint rotations (full roll/twist) straight into SMPL's
    frame ⇒ a clean mesh under standard LBS; the 5-spine→3-spine topology self-resolves
    (world rotations are absolute). Returns [T,22,3,3] SMPL-NATIVE (Y-up) world
    rotations. KOOPMAN_MASTER_PLAN.md §P11."""
    Rsrc = source_world_rotations(bvh_path, rot_csv, euler_order)
    srest = _source_rest_positions(bvh_path, offsets_pkl)
    rest = (smpl_model.J_regressor @ smpl_model.v_template)[:N_JOINTS].double()
    M = _rest_frame_alignment(srest, rest)                        # BVH frame → SMPL frame
    T = next(iter(Rsrc.values())).shape[0]
    Rn = torch.eye(3, dtype=torch.float64).repeat(T, N_JOINTS, 1, 1)
    for j in range(N_JOINTS):
        b = SMPL_TO_BVH[JOINT_NAMES[j]]
        if b in Rsrc:
            Rn[:, j] = M @ Rsrc[b][:T] @ M.transpose(-1, -2)     # pure conjugation (no C_j)
    return Rn


def compute_Q(bvh_path, offsets_pkl, smpl_model):
    """The single rigid SMPL-native→proper map Q = Rx(+90)·Mᵀ·Rx(+90).

    The rest-frame alignment M (BVH↔SMPL Procrustes) conjugated by the up-preserving
    proper rotation Rx(+90)=_RX_NEG90ᵀ. Returns [3,3] in smpl_model.dtype."""
    dt = smpl_model.dtype
    srest = _source_rest_positions(bvh_path, offsets_pkl)
    rest = (smpl_model.J_regressor @ smpl_model.v_template)[:N_JOINTS].double()
    M = _rest_frame_alignment(srest, rest)
    RXP = _RX_NEG90.double().t()                    # Rx(+90): up-preserving proper
    return (RXP @ M.t() @ RXP).to(dt)


def _read_worldpos(csv_path):
    """Read a *_worldpos.csv → {joint_name: [T,3] tensor} (Y-up→Z-up, mm→m)."""
    with open(csv_path, newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = [[float(v) for v in r] for r in rdr if r and r[0] != '']
    # header like Time,Hips.X,Hips.Y,Hips.Z,...
    cols = {}  # joint -> (ix, iy, iz)
    for i, h in enumerate(header):
        if '.' not in h:
            continue
        name, axis = h.rsplit('.', 1)
        cols.setdefault(name, {})[axis] = i
    data = torch.tensor(rows, dtype=torch.float32)  # [T, ncol]
    out = {}
    for name, ax in cols.items():
        if {'X', 'Y', 'Z'} <= set(ax):
            xyz = data[:, [ax['X'], ax['Y'], ax['Z']]] * MM_TO_M
            # mocap Y-up → Z-up: (x, y, z) -> (x, z, y)
            out[name] = torch.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], dim=-1)
    return out


def _smpl22_positions(worldpos):
    """{joint:[T,3]} mocap → [T, 22, 3] in SMPL joint order."""
    cols = []
    for jn in JOINT_NAMES:
        mocap = SMPL_TO_MOCAP[jn]
        if mocap not in worldpos:
            raise KeyError(f"mocap joint '{mocap}' (for SMPL '{jn}') not in CSV")
        cols.append(worldpos[mocap])
    return torch.stack(cols, dim=1)  # [T, 22, 3]
