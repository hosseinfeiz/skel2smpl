"""Per-dataset skeleton definitions + SMPL-fitting approaches.

Each motion-capture source has its own skeleton/marker layout and its own fit
recipe; both live here so they can be imported directly:

    from skel2smpl.datasets import fit_2c, fit_expi, fit_remocap, fit_dataset
    from skel2smpl.datasets import C2_MAP, EXPI_BONE, REMO_NAME

The 5 datasets:
  - 2c        — 2-character boxing BVH (20 joints, rotational). `C2_MAP`; fit_2c:
                BVH-rotation init (`_pose0_2c`) + unified MoSh++ position fit.
  - expi      — Extreme-Pose-Interaction TSV (18 SURFACE markers). `EXPI_BONE`;
                fit_expi: true-MoSh surface-marker solve (`fit_smpl_markers`, §P23).
  - lindyhop  — ReMoCap worldpos CSV (22 joints). `REMO_NAME`; fit_remocap:
  - ninjutsu  — ReMoCap worldpos CSV (22 joints). `REMO_NAME`; fit_remocap:
                BVH-rotation init (when *_rotations.csv present) + unified fit.
  - boxing    — INS cache, already SMPL-22 (Z-up). NOT a fit target — the reference
                pose is the GT; loaded/rendered directly by the vr pipeline.

`fit_*` each fit ONE performer (a file path) and return a `Fit` (joints, verts,
faces, marker/joint err mm, GT points, GT edges). Dyad pairing (actor/reactor) is
the caller's job — see `remocap_csvs` and the `_C0/_C1` (2C) / leader-follower
(ExPI) conventions.
"""
import os
import csv as _csv
from collections import namedtuple

import numpy as np
import torch

from skel2smpl.smpl import SMPL
from skel2smpl.fit import (solve_shape, solve_pose, forward_proper, fit_markers,
                           resample_spine_targets, measure_bone_ratios, _fk)
from skel2smpl.adapters import bvh_to_smpl_world_rotations, compute_Q
from skel2smpl.constants import _RX_NEG90, N_JOINTS, SMPL_PARENTS, JOINT_NAMES
from skel2smpl.geometry import rotation_matrix_to_axis_angle
from skel2smpl import skeletons as sk

RXP = _RX_NEG90.t().double()                 # SMPL native (Y-up) -> Z-up
PAR = list(SMPL_PARENTS)
SMPL_NAME = list(JOINT_NAMES)                # SMPL-22 joint names (== constants.JOINT_NAMES)
RETBONES = (4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21)   # legs+feet+arms length retarget

Fit = namedtuple("Fit", "joints verts faces err gt gt_edges")

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


# ═══════════════════════ generic solvers (any obs) ═══════════════════════
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
    """§P23 — true-MoSh surface-marker fit. obs_markers [T,M,3] proper-frame + marker_bone [M]
    -> (joints[T,22,3], verts, faces, marker_err_mm, virtual_markers). Q=RXP (obs native Y-up).
    bone_scale defaults to the per-bone anthropometric retarget measured from marker segments."""
    smpl = _smpl()
    if bone_scale is None:
        bone_scale = expi_bone_scale(obs_markers)
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


def marker_bone_scale(markers, bone_seg, clamp=(0.85, 1.30)):
    """Per-bone length ratios from marker segments -> bone_scale [24] (§P14), clamped.
    `bone_seg` maps SMPL child joint -> (markerA_idx, markerB_idx) that bracket the bone."""
    smpl = _smpl()
    J = (smpl.J_regressor @ smpl.v_template)[:N_JOINTS].double()
    bs = torch.ones(24, dtype=torch.float64)
    for c, (a, b) in bone_seg.items():
        Lm = (markers[:, a] - markers[:, b]).norm(dim=-1).mean()
        Ls = (J[c] - J[PAR[c]]).norm()
        bs[c] = float((Lm / Ls).clamp(*clamp))
    return bs


# ═══════════════════════ 2C — boxing BVH (20 joints, rotational) ═══════════════════════
# The 2C skeleton's joint centres are treated as MARKERS attached to SMPL bones and fit with
# the SAME true-MoSh solve as ExPI (graduated-σ from zero, free per-marker offset, torso anchor,
# joint limits, per-bone retarget) — more robust than the old BVH-rotation-init position fit.
C2_MAP = {"Hips": 0, "Spine": 3, "Spine1": 6, "HEad": 15, "LeftShoulder1": 13,
          "LeftArm": 16, "LeftForeArm": 18, "LeftHand": 20, "RightShoulder": 14,
          "RightArm": 17, "RightForeArm": 19, "RightHand": 21, "LeftUpLeg": 1,
          "LeftLeg": 4, "LeftFoot": 7, "LeftToes": 10, "RightUpLeg": 2,
          "RightLeg": 5, "RightFoot": 8, "RightToes": 11}
# fixed marker order (20 BVH joints) + the SMPL bone each attaches to.
C2_MARKERS = ("Hips", "Spine", "Spine1", "HEad", "LeftShoulder1", "LeftArm", "LeftForeArm",
              "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
              "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToes", "RightUpLeg", "RightLeg",
              "RightFoot", "RightToes")
C2_MARKER_BONE = [C2_MAP[nm] for nm in C2_MARKERS]
# marker-pair -> SMPL long bone (child) for the §P14 length retarget (limbs only; feet excluded).
C2_BONE_SEG = {18: (5, 6), 20: (6, 7), 19: (9, 10), 21: (10, 11),
               4: (12, 13), 7: (13, 14), 5: (16, 17), 8: (17, 18)}
# viz connectivity: markers whose SMPL bones are parent↔child in the SMPL tree.
C2_MARKER_EDGES = [(i, j) for i in range(len(C2_MARKERS)) for j in range(len(C2_MARKERS))
                   if SMPL_PARENTS[C2_MARKER_BONE[j]] == C2_MARKER_BONE[i]]


def _markers_2c(c_bvh, nf):
    """2C BVH -> (markers [T,20,3] Y-up m, marker_bone [20]). Joint centres as MoSh markers."""
    names, parents, offs, ch, frames, _ = sk.load_bvh(c_bvh)
    pos = sk.bvh_fk_world(names, parents, offs, ch, frames, scale=0.01)  # [T,J,3] Y-up m
    pos = torch.from_numpy(pos[:nf].astype(np.float64))
    markers = torch.stack([pos[:, names.index(nm)] for nm in C2_MARKERS], dim=1)
    return markers, torch.tensor(C2_MARKER_BONE, dtype=torch.long)


def fit_2c(bvh_path, max_frames=200):
    """Fit one 2C performer with the ExPI true-MoSh surface-marker solve (§P23)."""
    markers, mb = _markers_2c(bvh_path, max_frames)
    bs = marker_bone_scale(markers, C2_BONE_SEG)
    jp, vp, faces, err, _ = fit_smpl_markers(markers, mb, bone_scale=bs)
    return Fit(jp, vp, faces, err, markers.float(), C2_MARKER_EDGES)


# ═══════════════════════ ExPI — 18 SURFACE markers ═══════════════════════
# README marker order. The SMPL BONE each marker is rigidly attached to (free MoSh offset
# absorbs the surface→joint vector). `back` → spine3 (upper back), NOT spine1 (the 415mm bug).
# hips → PELVIS (0), not the femur (1/2): iliac-crest marker is rigid to the pelvis (§P23).
EXPI_MARKERS = ("fhead", "lhead", "rhead", "back", "lshoulder", "rshoulder", "lelbow",
                "relbow", "lwrist", "rwrist", "lhip", "rhip", "lknee", "rknee",
                "lheel", "rheel", "ltoes", "rtoes")
EXPI_BONE = [15, 15, 15, 9, 16, 17, 18, 19, 20, 21, 0, 0, 4, 5, 7, 8, 10, 11]
# marker connectivity (viz only) — head triangle, shoulders→elbows→wrists, back→hips→knees→heels→toes.
EXPI_EDGES = [(0, 1), (0, 2), (1, 2), (3, 0), (3, 4), (3, 5), (4, 6), (6, 8),
              (5, 7), (7, 9), (3, 10), (3, 11), (10, 12), (12, 14), (14, 16),
              (11, 13), (13, 15), (15, 17), (10, 11)]
# which marker pair measures each SMPL LONG bone (child joint -> markerA,markerB) for the
# §P14 length retarget. ONLY long limb bones (thigh/shin/upper-arm/forearm); FEET EXCLUDED
# (heel→toes = foot LENGTH not the ankle→ball bone → 1.8× alien feet). Ratios clamped [0.85,1.30].
EXPI_BONE_SEG = {4: (10, 12), 5: (11, 13), 7: (12, 14), 8: (13, 15),
                 18: (4, 6), 19: (5, 7), 20: (6, 8), 21: (7, 9)}


def expi_bone_scale(obs_markers):
    return marker_bone_scale(obs_markers, EXPI_BONE_SEG)


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
    return p, torch.tensor(EXPI_BONE, dtype=torch.long)


def fit_expi(tsv_path, max_frames=200, follower=False):
    """Fit one ExPI performer (leader / follower): true-MoSh surface-marker solve (§P23)."""
    markers, mb = _markers_expi(tsv_path, max_frames, follower=follower)
    jp, vp, faces, err, _ = fit_smpl_markers(markers, mb)
    return Fit(jp, vp, faces, err, markers.float(), EXPI_EDGES)


# ═══════════════════ ReMoCap (LindyHop / Ninjutsu) — 22-joint worldpos CSV ═══════════════════
REMO_NAME = {'pelvis': 'Hips', 'left_hip': 'LeftUpLeg', 'right_hip': 'RightUpLeg',
             'spine1': 'Spine', 'left_knee': 'LeftLeg', 'right_knee': 'RightLeg',
             'spine2': 'Spine2', 'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot',
             'spine3': 'Spine4', 'left_foot': 'LeftToeBase', 'right_foot': 'RightToeBase',
             'neck': 'Neck', 'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder',
             'head': 'Head', 'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm',
             'left_elbow': 'LeftForeArm', 'right_elbow': 'RightForeArm',
             'left_wrist': 'LeftHand', 'right_wrist': 'RightHand'}


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
    for j, sn in enumerate(SMPL_NAME):
        nm = REMO_NAME[sn]
        if nm in col and {'X', 'Y', 'Z'} <= set(col[nm]):
            obs[:, j] = data[:, [col[nm]['X'], col[nm]['Y'], col[nm]['Z']]] * 0.001  # mm->m Y-up
            w[j] = 1.0
    return obs, w


def _fit_remocap(wp_path, nf):
    """ReMoCap SMPL body: MoSh position fit MATCHING the worldpos GT, initialized from the exact
    BVH rotations (correct twist) when *_rotations.csv/_offsets.pkl are present. resample_spine_targets
    fixes the 5-spine->3-spine torso-lean. Returns (joints, verts, faces, err) Y-up, or None if the
    BVH/rotations/offsets are absent (caller falls back to zero-init position fit)."""
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


def remocap_csvs(seq_dir):
    """ReMoCap dyad layout -> (actor_worldpos.csv, reactor_worldpos.csv)."""
    flat0 = os.path.join(seq_dir, "0_worldpos.csv")
    if os.path.exists(flat0):
        return flat0, os.path.join(seq_dir, "1_worldpos.csv")
    return (os.path.join(seq_dir, "0", "motion_worldpos.csv"),
            os.path.join(seq_dir, "1", "motion_worldpos.csv"))


def fit_remocap(worldpos_csv, max_frames=200):
    """Fit one ReMoCap performer (LindyHop / Ninjutsu): BVH-rotation init when available,
    else zero-init position fit to the 22-joint worldpos GT."""
    res = _fit_remocap(worldpos_csv, max_frames)
    if res is not None:
        jp, vp, faces, err = res
        gt = _obs_remocap(worldpos_csv, max_frames)[0][:jp.shape[0]].float()
    else:
        obs, w = _obs_remocap(worldpos_csv, max_frames)
        jp, vp, faces, err = fit_smpl(obs, w, lam_contact=8.0)
        gt = obs.float()
    return Fit(jp, vp, faces, err, gt, sk.smpl22_edges())


# ═══════════════════════ registry ═══════════════════════
# name -> (fit-one-performer fn). LindyHop and Ninjutsu share the ReMoCap recipe.
DATASETS = {"2c": fit_2c, "expi": fit_expi, "lindyhop": fit_remocap, "ninjutsu": fit_remocap}


def fit_dataset(name, path, **kw):
    """Generic entry: fit one performer of `name` at `path` -> Fit. Boxing is not a fit
    target (the INS cache is already SMPL-22 GT); fit the others via the registry."""
    if name not in DATASETS:
        raise ValueError(f"no fit recipe for {name!r}; known: {sorted(DATASETS)} "
                         f"(boxing is reference SMPL, not a fit target)")
    return DATASETS[name](path, **kw)
