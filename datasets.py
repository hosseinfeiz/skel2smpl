import os
import csv as _csv

import numpy as np
import torch
import torch.nn.functional as Fn

from skel2smpl.smpl import SMPL
from skel2smpl.fit import (solve_shape, solve_pose, forward_proper, fit_markers,
                           resample_spine_targets, measure_bone_ratios, _fk)
from skel2smpl.adapters import bvh_to_smpl_world_rotations, compute_Q
from skel2smpl.constants import _RX_NEG90, N_JOINTS, SMPL_PARENTS, SMPL22_PRIMARY_CHILD
from skel2smpl.geometry import rotation_matrix_to_axis_angle
from skel2smpl import skeletons as sk

RXP = _RX_NEG90.t().double()                 # SMPL native (Y-up) -> Z-up
PAR = list(SMPL_PARENTS)
RETBONES = (4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21)   # legs+feet+arms length retarget
_SMPL_NAME = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
              'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
              'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
              'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']

_SMPL_CACHE = None


def _smpl():
    global _SMPL_CACHE
    if _SMPL_CACHE is None:
        _SMPL_CACHE = SMPL()
    return _SMPL_CACHE


def _world_to_local_aa(Rw):
    """[T,22,3,3] SMPL-native WORLD rotations -> [T,22,3] LOCAL axis-angle (root local=world)."""
    T = Rw.shape[0]
    Rl = torch.eye(3, dtype=torch.float64).repeat(T, N_JOINTS, 1, 1)
    for j in range(N_JOINTS):
        Rl[:, j] = Rw[:, j] if PAR[j] < 0 else Rw[:, PAR[j]].transpose(-1, -2) @ Rw[:, j]
    return rotation_matrix_to_axis_angle(Rl.reshape(-1, 3, 3)).reshape(T, N_JOINTS, 3)


# ───────────────────────── 2C (embedded BVH, 20 joints) ─────────────────────────
_C2_MAP = {"Hips": 0, "Spine": 3, "Spine1": 6, "HEad": 15, "LeftShoulder1": 13,
           "LeftArm": 16, "LeftForeArm": 18, "LeftHand": 20, "RightShoulder": 14,
           "RightArm": 17, "RightForeArm": 19, "RightHand": 21, "LeftUpLeg": 1,
           "LeftLeg": 4, "LeftFoot": 7, "LeftToes": 10, "RightUpLeg": 2,
           "RightLeg": 5, "RightFoot": 8, "RightToes": 11}
_C2_NAME = {_SMPL_NAME[j]: nm for nm, j in _C2_MAP.items()}


def _obs_2c(c_bvh, nf):
    names, parents, offs, ch, frames, _ = sk.load_bvh(c_bvh)
    pos = sk.bvh_fk_world(names, parents, offs, ch, frames, scale=0.01)  # [T,J,3] Y-up m
    pos = torch.from_numpy(pos[:nf].astype(np.float64))
    obs = torch.zeros(pos.shape[0], 22, 3, dtype=torch.float64)
    w = torch.zeros(22)
    for nm, j in _C2_MAP.items():
        if nm in names:
            obs[:, j] = pos[:, names.index(nm)]; w[j] = 1.0
    obs[:, 12] = obs[:, 6] + 0.65 * (obs[:, 15] - obs[:, 6])   # neck seed (chest->head)
    obs[:, 9] = 0.5 * (obs[:, 6] + obs[:, 12])                 # spine3 seed
    return obs, w


def _bvh_world_rots(names, parents, offsets, channels, frames):
    T, J = frames.shape[0], len(names)
    gR = np.zeros((T, J, 3, 3), np.float64)
    rest = np.zeros((J, 3), np.float64)
    for j in range(J):
        rest[j] = offsets[j] if parents[j] < 0 else rest[parents[j]] + offsets[j]
    for t in range(T):
        vals = frames[t]; k = 0; g = [None] * J
        for j in range(J):
            loc = np.eye(3)
            for cn in channels[j]:
                v = vals[k]; k += 1
                if not cn.endswith("position"):
                    loc = loc @ sk._axis_rot(cn[0], v)
            g[j] = loc if parents[j] < 0 else g[parents[j]] @ loc
        gR[t] = np.asarray(g)
    return gR, rest


def _pose0_2c(c_bvh, nf):
    names, parents, offs, ch, frames, _ = sk.load_bvh(c_bvh)
    gR, rest = _bvh_world_rots(names, parents, offs, ch, frames)
    gR = torch.from_numpy(gR[:nf]); rest = torch.from_numpy(rest)
    smpl = _smpl()
    Js = (smpl.J_regressor @ smpl.v_template)[:N_JOINTS].double()
    A, B = [], []
    for j in range(N_JOINTS):
        c = SMPL22_PRIMARY_CHILD[j]
        sj, sc = _SMPL_NAME[j], (_SMPL_NAME[c] if c >= 0 else None)
        if c < 0 or sj not in _C2_NAME or sc not in _C2_NAME:
            continue
        bj, bc = names.index(_C2_NAME[sj]), names.index(_C2_NAME[sc])
        A.append(Fn.normalize((rest[bc] - rest[bj]).unsqueeze(0), dim=-1)[0])
        B.append(Fn.normalize((Js[c] - Js[j]).unsqueeze(0), dim=-1)[0])
    A, B = torch.stack(A), torch.stack(B)
    U, _, Vt = torch.linalg.svd(A.t() @ B)
    d = torch.sign(torch.linalg.det(Vt.t() @ U.t()))
    M = Vt.t() @ torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d])) @ U.t()
    T = gR.shape[0]
    Rw = torch.eye(3, dtype=torch.float64).repeat(T, N_JOINTS, 1, 1)
    for j in range(N_JOINTS):                  # topological order (parent idx < j)
        nm = _C2_NAME.get(_SMPL_NAME[j])
        if nm in names:
            Rw[:, j] = M @ gR[:, names.index(nm)] @ M.t()
        elif PAR[j] >= 0:
            Rw[:, j] = Rw[:, PAR[j]]           # INHERIT parent world-rot -> local identity (no twist)
    return _world_to_local_aa(Rw)


# ───────────────────────── ExPI (18 surface markers) ─────────────────────────
# §P23 — ExPI 18 markers are SURFACE points (README: fhead,lhead,rhead, back,
# l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r heel, l/r toes).
# The SMPL BONE each marker is rigidly attached to (free MoSh offset absorbs the
# surface→joint vector). `back` → spine3 (upper back), NOT spine1 (the 415mm bug).
# hips → PELVIS (0), not the femur joint (1/2): the iliac-crest marker is rigid to the
# pelvis, so it must NOT swing with the thigh when the leg lifts (§P23 215mm hip bug).
_EXPI_BONE = [15, 15, 15, 9, 16, 17, 18, 19, 20, 21, 0, 0, 4, 5, 7, 8, 10, 11]


# §P23 — which marker pair measures each SMPL LONG bone (child joint -> markerA,markerB).
# ONLY the long limb bones whose two markers bracket the bone end-to-end (thigh, shin,
# upper-arm, forearm). The FEET are EXCLUDED: the heel→toes marker spans the whole foot
# LENGTH, not SMPL's short ankle→ball bone, so its ratio balloons to ~1.8× → alien giant
# feet (user 2026-06-18). Ratios are CLAMPED to [0.85,1.30]: marker segments are
# offset-contaminated (the iliac-crest hip marker inflates the thigh), and §P14 warns
# extreme bone_scale tears the mesh; clamping keeps the retarget anthropometric, not alien.
# (The real proportion+girth fix is §P23.v surface-vertex attach.)
_EXPI_BONE_SEG = {4: (10, 12), 5: (11, 13), 7: (12, 14), 8: (13, 15),
                  18: (4, 6), 19: (5, 7), 20: (6, 8), 21: (7, 9)}


def _expi_bone_scale(obs_markers):
    """Per-bone length ratios from marker segments -> bone_scale [24] (§P14), clamped."""
    smpl = _smpl()
    J = (smpl.J_regressor @ smpl.v_template)[:N_JOINTS].double()
    bs = torch.ones(24, dtype=torch.float64)
    for c, (a, b) in _EXPI_BONE_SEG.items():
        Lm = (obs_markers[:, a] - obs_markers[:, b]).norm(dim=-1).mean()
        Ls = (J[c] - J[PAR[c]]).norm()
        bs[c] = float((Lm / Ls).clamp(0.85, 1.30))
    return bs


def _markers_expi(tsv, nf, follower=False):
    """ExPI surface markers in the proper (Y-up) frame — RAW 18 points, NO synthesis.
    Returns (obs_markers [T,18,3], marker_bone [18] long)."""
    with open(tsv) as f:
        f.readline()
        rows = [[float(x) for x in ln.strip().split(",")] for ln in f if ln.strip()]
    a = np.asarray(rows, np.float32).reshape(len(rows), 36, 3) * 0.001
    p = torch.from_numpy(a[:nf, 18:].astype(np.float64) if follower
                         else a[:nf, :18].astype(np.float64))
    p = torch.einsum("mn,tjn->tjm", _RX_NEG90.double(), p)        # Z-up -> Y-up proper
    return p, torch.tensor(_EXPI_BONE, dtype=torch.long)


# ───────────────── ReMoCap (LindyHop / Ninjutsu) worldpos CSV, all 22 SMAP ─────────────────
_REMO_NAME = {'pelvis': 'Hips', 'left_hip': 'LeftUpLeg', 'right_hip': 'RightUpLeg',
              'spine1': 'Spine', 'left_knee': 'LeftLeg', 'right_knee': 'RightLeg',
              'spine2': 'Spine2', 'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot',
              'spine3': 'Spine4', 'left_foot': 'LeftToeBase', 'right_foot': 'RightToeBase',
              'neck': 'Neck', 'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder',
              'head': 'Head', 'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm',
              'left_elbow': 'LeftForeArm', 'right_elbow': 'RightForeArm',
              'left_wrist': 'LeftHand', 'right_wrist': 'RightHand'}


def _fit_remocap(wp_path, nf):
    """ReMoCap SMPL body: MoSh position fit that MATCHES the worldpos GT (torso upright, no
    lean) initialized from the exact BVH rotations (correct twist). The Mixamo retarget bakes
    the rest-frame M into its rotations, so the fit runs with compute_Q in the Z-up proper
    frame; outputs are rotated back to Y-up. resample_spine_targets fixes the 5-spine->3-spine
    torso-lean (pure rotation transport leans the torso ~16deg vs GT). Returns (joints, verts,
    faces, err) Y-up, or None if bvh/rotations/offsets absent."""
    base = wp_path[:-len("_worldpos.csv")]
    bvh, rot, off = base + ".bvh", base + "_rotations.csv", base + "_offsets.pkl"
    if not (os.path.exists(bvh) and os.path.exists(rot) and os.path.exists(off)):
        return None
    smpl = _smpl()
    pose0 = _world_to_local_aa(bvh_to_smpl_world_rotations(bvh, rot, off, smpl)[:nf])
    Q = compute_Q(bvh, off, smpl)                                  # honors the Mixamo rest M
    T = pose0.shape[0]
    obs, w = _obs_remocap(wp_path, nf); obs = obs[:T]
    obsZ = torch.einsum("mn,tjn->tjm", RXP, obs)                   # native Y-up -> Z-up proper
    jp, vp, faces, _ = fit_smpl(obsZ, w, pose0=pose0, lam_contact=8.0, Q=Q)
    RY = _RX_NEG90.float()                                          # proper Z-up -> Y-up native
    jp = torch.einsum("mn,tjn->tjm", RY, jp)
    vp = torch.einsum("mn,tvn->tvm", RY, vp)
    err = (jp[:, w > 0.5] - obs[:, w > 0.5].float()).norm(dim=-1).mean().item() * 1000.0
    return jp, vp, faces, err


def _obs_remocap(csv_path, nf):
    with open(csv_path, newline='') as f:
        rdr = _csv.reader(f); header = next(rdr)
        rows = [[float(v) for v in r] for r in rdr if r and r[0] != '']
    col = {}
    for i, h in enumerate(header):
        if '.' in h:
            nm, ax = h.rsplit('.', 1); col.setdefault(nm, {})[ax] = i
    data = torch.tensor(rows[:nf], dtype=torch.float64)
    obs = torch.zeros(data.shape[0], 22, 3, dtype=torch.float64)
    w = torch.zeros(22)
    for j, sn in enumerate(_SMPL_NAME):
        nm = _REMO_NAME[sn]
        if nm in col and {'X', 'Y', 'Z'} <= set(col[nm]):
            obs[:, j] = data[:, [col[nm]['X'], col[nm]['Y'], col[nm]['Z']]] * 0.001  # mm->m Y-up
            w[j] = 1.0
    return obs, w


def _remocap_csvs(seq_dir):
    """seq_dir -> (actor_worldpos.csv, reactor_worldpos.csv) for ReMoCap layout."""
    flat0 = os.path.join(seq_dir, "0_worldpos.csv")
    if os.path.exists(flat0):
        return flat0, os.path.join(seq_dir, "1_worldpos.csv")
    return (os.path.join(seq_dir, "0", "motion_worldpos.csv"),
            os.path.join(seq_dir, "1", "motion_worldpos.csv"))


# ───────────────────────────── unified fit ─────────────────────────────
def fit_smpl(obs, w, pose0=None, lam_contact=8.0, iters=600, Q=None):
    """obs [T,22,3] + per-joint weight w -> (joints[T,22,3], verts, faces, err) in obs's OWN
    frame, via §P16 spine-resample (fixes torso lean) + §P14 bone_scale + MoSh position fit.
    Q maps SMPL-native->obs frame: default RXP (obs native Y-up, for 2C/ExPI); ReMoCap passes
    compute_Q with obs in Z-up so the Mixamo rest-frame M is honored."""
    smpl = _smpl()
    if Q is None:
        Q = RXP
    T = obs.shape[0]
    if pose0 is None:
        pose0 = torch.zeros(T, 22, 3, dtype=torch.float64)
    beta, s = solve_shape(obs, pose0, smpl, lam_reg=0.8)
    obs2 = resample_spine_targets(obs, smpl, betas=beta, scale=s)
    bones = tuple(c for c in RETBONES if w[PAR[c]] > 0.5 and w[c] > 0.5)
    bs = measure_bone_ratios(obs2, smpl, betas=beta, scale=s, bones=bones) if bones else None
    pose = solve_pose(obs2, pose0, smpl, Q, betas=beta, scale=s, iters=iters,
                      lam_contact=lam_contact, bone_scale=bs)
    jp, vp = forward_proper(smpl, pose, Q, betas=beta, scale=s,
                            root_target=obs2[:, 0], want_verts=True, bone_scale=bs)
    m = w > 0.5
    err = (jp[:, m] - obs2[:, m]).norm(dim=-1).mean().item() * 1000.0
    return jp.float(), vp.float(), smpl.faces, err


def fit_smpl_markers(obs_markers, marker_bone, lam_contact=0.0, bone_scale=None):
    """§P23 — true-MoSh surface-marker fit (ExPI). obs_markers [T,M,3] proper-frame +
    marker_bone [M] -> (joints[T,22,3], verts, faces, marker_err_mm, virtual_markers).
    Q=RXP (obs is native Y-up, as for 2C/ExPI). bone_scale defaults to the per-bone
    anthropometric retarget measured from the marker segments (§P14)."""
    smpl = _smpl()
    if bone_scale is None:
        bone_scale = _expi_bone_scale(obs_markers)
    pose, trans, beta, s, delta = fit_markers(obs_markers, marker_bone, smpl, RXP,
                                              lam_contact=lam_contact, bone_scale=bone_scale)
    jp, vp = forward_proper(smpl, pose, RXP, betas=beta, scale=s, root_target=trans,
                            want_verts=True, bone_scale=bone_scale)
    dt = smpl.dtype
    QM = (RXP @ _RX_NEG90.double()).to(dt)
    jnat, Rnat = _fk(smpl, pose, beta, bone_scale=bone_scale)
    Reff = torch.einsum("mn,tjnk->tjmk", QM, Rnat)
    mb = torch.as_tensor(marker_bone, dtype=torch.long)
    jpd = jp.to(dt)
    vm = jpd[:, mb] + s * torch.einsum("tmkl,ml->tmk", Reff[:, mb], delta.to(dt))
    err = (vm - obs_markers.to(dt)).norm(dim=-1).mean().item() * 1000.0
    return jp.float(), vp.float(), smpl.faces, err, vm.float()
