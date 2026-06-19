"""P13 — Principled unified SMPL fit (MoSh++/EasyMocap, ONE optimization).

Supersedes the accreted P8/P10/P11/P12 fitting stack (four functions, three
`--gt-pose` modes, a deterministic foot-snap, a post-hoc render scale, hand-tuned
per-joint weight tables, and separate GT/pred paths). Here a SINGLE MAP estimate
per subject — a shared shape beta + uniform size scale s, per-frame pose theta_t —
is solved against the observed 3D joints, and a SINGLE `forward_proper` is used by
both the objective and the render, for both GT and pred.

The principle (KOOPMAN_MASTER_PLAN.md §P13):

    E(beta,s,{theta_t}) = lam_D·E_data + lam_b·E_beta + lam_th·E_pose
                          + lam_u·E_smooth + lam_c·E_contact

  E_data    robust GMOF 3D-joint distance, spine downweighted (BVH 5-spine→SMPL
            3-spine topology, §P9), feet up-weighted for contact.
  E_beta    ‖beta‖² shape prior (beta~N(0,I)).
  E_pose    ‖theta_t − theta_init_t‖²  — the BVH-retarget anchor (our principled
            substitute for MoSh++'s GMM/VPoser prior; carries the source twist).
  E_smooth  2nd-order acceleration, torso>limb weighted (EasyMocap smooth_poses).
  E_contact ground-penetration + in-contact anti-skate (EasyMocap contact.py) —
            replaces the deterministic foot-snap.

References (verified line refs in pose_estimation/kinematics/optimization.py):
  optimizeShape 785-836, gmof 61-62, 3D data term 361-379, multi_stage 838-864,
  smoothness 441-511, ground/skate 387-397; MoSh++/AMASS arXiv:1904.03278;
  SMPLify-X/VPoser arXiv:1904.05866; SMPL Loper et al. 2015.

The frame: `vizsys.smpl.SMPL.forward_R` runs FK then applies `_TRANS=Rx(−90)`, so
its output reaches the up-preserving "proper" render frame only via the rigid
Q = Rx(+90)·Mᵀ·Rx(+90) (M = rest-frame Procrustes alignment, motion_bridge
`_rest_frame_alignment`). Q is correct and kept — applied ONCE inside
`forward_proper`. Observed joints are un-mirrored to proper once at ingress
(F=diag(1,−1,1)) by the caller.
"""

import numpy as np
import torch
from scipy.signal import savgol_filter

from skel2smpl.constants import (
    N_JOINTS, SMPL_PARENTS, SPINE_NECK_COLLAR, _SMPL22_BONES, _RX_NEG90,
)
from skel2smpl.geometry import exp_so3, log_so3, hat_so3
from skel2smpl.smpl import retarget_joints

# SMPL-22 joint-index groups used by the data/contact terms.
_FEET = [7, 8, 10, 11]      # ankles + feet  (grounded by the fit)
_LEGS = [1, 2, 4, 5]        # hips + knees   (stance stability)
_FOOT_GROUND = [10, 11]     # foot joints used for the contact/ground term
_TORSO = [0, 3, 6, 9, 12, 15]   # root + spine1/2/3 + neck + head (smooth heavier)
# Joints allowed a latent offset (MoSh markers): the upper-body convention-mismatch
# region — spine1/2/3, neck, L/R collar (the 5→3 spine gap, §P9) + head + L/R shoulder
# (Mixamo vs SMPL clavicle/neck/head anatomy). Everything else keeps δ≈0 (matches by pose).
_DELTA_FREE = (3, 6, 9, 12, 13, 14, 15, 16, 17)

# ── §P17 anatomical kinematics regularizers (joint limits + passive damping) ──────
# Per-joint per-axis [x,y,z] range-of-motion (rad) + passive damping, ported VERBATIM
# from pose_estimation/kinematics/phys_body.py (get_rom_array / get_passive_damping_array,
# the JOINT_DATA table). SMPL-22 indexed. The same anatomical bounds the avbd/kintrack
# solvers use; koopman has the 3D-joint observation but no 2D term. See §P17.
_LIMB_LEFT = {1, 4, 7, 10, 13, 16, 18, 20}              # SMPL left-side joints (lateral sign)
_ROM_LO = torch.tensor([
    [-0.524, -0.785, -0.524], [-0.524, -0.785, -0.436], [-0.524, -0.611, -0.785],
    [-0.349, -0.175, -0.349], [-0.087, -0.175, -0.087], [-0.087, -0.524, -0.087],
    [-0.262, -0.262, -0.262], [-0.698, -0.087, -0.611], [-0.698, -0.087, -0.349],
    [-0.175, -0.349, -0.175], [-0.524, -0.087, -0.087], [-0.524, -0.087, -0.087],
    [-0.611, -0.349, -0.349], [-0.175, -0.262, -0.436], [-0.175, -0.262, -0.436],
    [-0.436, -0.785, -0.262], [-0.873, -1.396, -1.571], [-0.873, -1.222, -1.571],
    [-0.087, -1.396, -0.175], [-0.087, -1.396, -0.175], [-1.222, -0.087, -0.524],
    [-1.222, -0.087, -0.349]], dtype=torch.float64)
_ROM_HI = torch.tensor([
    [0.524, 0.785, 0.524], [2.094, 0.611, 0.785], [2.094, 0.785, 0.436],
    [0.698, 0.175, 0.349], [2.269, 0.524, 0.087], [2.269, 0.175, 0.087],
    [0.524, 0.262, 0.262], [0.349, 0.087, 0.349], [0.349, 0.087, 0.611],
    [0.349, 0.349, 0.175], [1.222, 0.087, 0.087], [1.222, 0.087, 0.087],
    [0.524, 0.349, 0.349], [0.611, 0.262, 0.436], [0.611, 0.262, 0.436],
    [0.524, 0.785, 0.262], [3.142, 1.222, 1.571], [3.142, 1.396, 1.571],
    [2.531, 1.396, 0.175], [2.531, 1.396, 0.175], [1.396, 0.087, 0.349],
    [1.396, 0.087, 0.524]], dtype=torch.float64)
_JOINT_DAMP = torch.tensor([
    [1.5, 1.5, 1.5], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5],
    [0.5, 1.0, 1.0], [0.5, 1.0, 1.0], [1.5, 1.5, 1.5], [1.0, 1.0, 0.7],
    [1.0, 1.0, 0.7], [1.5, 1.5, 1.5], [0.5, 0.8, 0.8], [0.5, 0.8, 0.8],
    [0.7, 0.7, 0.7], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5],
    [0.8, 0.6, 0.8], [0.8, 0.6, 0.8], [0.4, 0.3, 0.5], [0.4, 0.3, 0.5],
    [0.3, 0.5, 0.3], [0.3, 0.5, 0.3]], dtype=torch.float64)
# Twist (axial) axis index per joint (phys_body _anatomical_limit_axes spec): collars
# 13/14 twist about z=idx2; every other limited joint about y=idx1. Joint 0 unused.
_LIMIT_AXIAL = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1],
                            dtype=torch.long)
_LIMIT_JOINTS = list(range(1, N_JOINTS))                # all movable joints (root 0 is free)
# Per-joint SCALAR passive damping (phys_model.py:46 collapses to mean-over-axes, root→0):
# the §P17 velocity-damping weight — LOW on fast distal joints, HIGH on the slow spine/proximal.
_DAMP_SCALAR = _JOINT_DAMP.mean(dim=1).clone()
_DAMP_SCALAR[0] = 0.0

# §P14 leg-retarget bone set: the long bones thigh/shin/foot (child-indexed), L+R.
# Hips 1,2 are EXCLUDED (stance-width, not the ankle-drop residual; they drove the
# worst mesh shear — see §P14 BUILD DECISION). All other bones keep r=1 (identity).
LEG_BONES = (4, 5, 7, 8, 10, 11)


# compute_Q (the rigid SMPL-native→proper map) lives in skel2smpl.adapters — it
# reads a BVH/offsets file, so it belongs with the input adapters, not the solvers.


def _fk(smpl_model, pose, betas=None, bone_scale=None):
    """Fast joints-only FK (no LBS verts) — the hot-loop forward for the solver.

    Returns (j_native [T,22,3], R_world [T,22,3,3]) in SMPL's pre-_TRANS native
    frame. `forward_proper`'s verts path (used by the render) computes the same
    joints with the LBS verts attached; this skips the 6890-vertex skinning that
    dominates the optimisation cost. `bone_scale` [24] (§P14) retargets the rest
    JOINTS before FK so the solver sees the anthropometric bone lengths — joints
    only here (no mesh), matching forward_R's joint output; None ⇒ unchanged."""
    dt = smpl_model.dtype
    pose = torch.as_tensor(pose, dtype=dt)
    T = pose.shape[0]
    betas = (torch.zeros(smpl_model.n_betas, dtype=dt) if betas is None
             else torch.as_tensor(betas, dtype=dt))
    v_shaped = smpl_model.v_template + torch.einsum(
        "vxb,b->vx", smpl_model.shapedirs, betas)
    J = smpl_model.J_regressor @ v_shaped                        # [24,3] native
    if bone_scale is not None:                                   # §P14 rest-joint retarget
        J = retarget_joints(J, torch.as_tensor(bone_scale, dtype=dt), smpl_model.parents)
    eye = torch.eye(3, dtype=dt).expand(T, 24 - N_JOINTS, 3, 3)
    R = torch.cat([exp_so3(pose.reshape(-1, 3)).reshape(T, N_JOINTS, 3, 3), eye], dim=1)
    G = smpl_model._global_transforms(R, J)                      # [T,24,4,4]
    return G[:, :N_JOINTS, :3, 3], G[:, :N_JOINTS, :3, :3]


def forward_proper(smpl_model, pose, Q, betas=None, scale=1.0, root_target=None,
                   want_verts=False, skinning="lbs", bone_scale=None):
    """THE single SMPL forward, used by the objective AND the render, GT and pred.

    forward_R(theta) → Q-conjugate to proper → place root at `root_target` → uniform
    `scale` about that root. This is exactly the transform the §P12 global fit applied
    in its inner `fwd` (motion_bridge.py:499-504), promoted to the one shared path so
    verts and joints, GT and pred, all live in the same proper frame with no snap and
    no post-hoc scale.

    pose:        [T,22,3] SMPL LOCAL axis-angle.
    Q:           [3,3] from compute_Q.
    root_target: [T,3] proper-frame target for joint 0 (the observed root). None ⇒ no
                 grounding/scale (raw Q-mapped output).
    Returns joints [T,22,3]; if want_verts also verts [T,6890,3] (proper frame)."""
    dt = smpl_model.dtype
    pose = torch.as_tensor(pose, dtype=dt)
    T = pose.shape[0]
    betas = None if betas is None else torch.as_tensor(betas, dtype=dt)
    Qd = torch.as_tensor(Q, dtype=dt)
    eye = torch.eye(3, dtype=dt).expand(T, 24 - N_JOINTS, 3, 3)
    R = torch.cat([exp_so3(pose.reshape(-1, 3)).reshape(T, N_JOINTS, 3, 3), eye], dim=1)
    verts, j = smpl_model.forward_R(R, torch.zeros(T, 3, dtype=dt), betas=betas,
                                    skinning=skinning, bone_scale=bone_scale)
    j = j[:, :N_JOINTS].to(dt)
    jp = torch.einsum("mn,tjn->tjm", Qd, j)                       # SMPL native → proper
    if want_verts:
        vp = torch.einsum("mn,tvn->tvm", Qd, verts.to(dt))
    if root_target is not None:
        rt = torch.as_tensor(root_target, dtype=dt)
        shift = rt[:, None, :] - jp[:, :1]                        # ground root to target
        jp = jp + shift
        if want_verts:
            vp = vp + shift
        s = float(scale)
        if abs(s - 1.0) > 1e-9:
            c = rt[:, None, :]
            jp = c + s * (jp - c)                                 # uniform scale about root
            if want_verts:
                vp = c + s * (vp - c)
    return (jp, vp) if want_verts else jp


def detect_contacts(obs_joints, h_thresh: float = 0.08, v_thresh: float = 0.02,
                    med: int = 5):
    """Foot-contact mask from observed joints (EasyMocap contact.py adapted).

    A foot is in contact when it is near the floor (height < h_thresh above the
    sequence-min foot height) AND moving slowly (per-frame xy displacement <
    v_thresh m). The boolean mask is median-filtered (window `med`) to remove
    single-frame flicker. v_thresh is in metres/frame (EasyMocap's 0.25 m/s at
    typical mocap fps ≈ a couple cm/frame).

    obs_joints: [T,22,3] proper-frame Z-up observed joints.
    Returns (contact [T, len(_FOOT_GROUND)] bool, floor scalar z)."""
    obs = torch.as_tensor(obs_joints, dtype=torch.float64)
    T = obs.shape[0]
    z = obs[:, _FOOT_GROUND, 2]                                   # [T,F]
    floor = z.min()
    height_ok = (z - floor) < h_thresh
    vel = torch.zeros_like(z)
    if T > 1:
        d = (obs[1:, _FOOT_GROUND, :2] - obs[:-1, _FOOT_GROUND, :2]).norm(dim=-1)
        vel[1:] = d
        vel[0] = d[0]
    vel_ok = vel < v_thresh
    contact = height_ok & vel_ok
    if med > 1 and T >= med:                                      # median de-flicker
        pad = med // 2
        cf = contact.float()
        cp = torch.nn.functional.pad(cf.t()[None], (pad, pad), mode='replicate')[0]
        win = cp.unfold(-1, med, 1)                               # [F,T,med]
        contact = (win.median(dim=-1).values > 0.5).t()
    return contact, floor


def gmof(sq, sigma_sq: float = 0.04):
    """Geman-McClure / GMOF robustifier (EasyMocap optimization.py:61-62).
    sq: squared residual (sum over xyz per joint). rho = sigma²·sq/(sigma²+sq)."""
    return sigma_sq * sq / (sigma_sq + sq)


def measure_bone_ratios(obs_joints, smpl_model, betas=None, scale: float = 1.0,
                        bones=LEG_BONES):
    """§P14 per-bone retarget ratios r[c] = L_data[c] / (s · L_smpl_β[c]).

    Frame-invariant bone lengths (mean over frames) of the OBSERVED skeleton vs the
    β-shaped SMPL rest skeleton, per kinematic edge parent→child, divided by the §P13
    uniform scale s so the retarget is ORTHOGONAL to (β,s): the forward's `scale=s`
    then renders bone length s·r·L_smpl_β = L_data. Only `bones` (default the long leg
    bones) get a ratio; every other bone — root, hips, hands, upper body — stays r=1
    (identity there). Read-only in β. Bone lengths are rotation/mirror invariant, so the
    proper-frame obs and native-frame J are directly comparable. Returns bone_scale [24]
    for forward_R / _fk / forward_proper. Pred reuses the GT subject's r (like β/s)."""
    dt = smpl_model.dtype
    obs = torch.as_tensor(obs_joints, dtype=dt)[:, :N_JOINTS]
    betas = (torch.zeros(smpl_model.n_betas, dtype=dt) if betas is None
             else torch.as_tensor(betas, dtype=dt))
    v_shaped = smpl_model.v_template + torch.einsum(
        "vxb,b->vx", smpl_model.shapedirs, betas)
    J = (smpl_model.J_regressor @ v_shaped)[:N_JOINTS]           # β-shaped native rest
    s = float(scale)
    r = torch.ones(24, dtype=dt)
    for c in bones:
        p = SMPL_PARENTS[c]
        L_data = (obs[:, c] - obs[:, p]).norm(dim=-1).mean()
        L_smpl = (J[c] - J[p]).norm().clamp_min(1e-8)
        r[c] = L_data / (s * L_smpl)
    return r


# §P16: SMPL-22 torso chain (pelvis→spine1→spine2→spine3→neck) — kinematic, each
# the parent of the next (SMPL_PARENTS[3]=0,[6]=3,[9]=6,[12]=9). The middle three
# are the resampled joints.
_TORSO_CHAIN = (0, 3, 6, 9, 12)
_SPINE_MID = (3, 6, 9)


def resample_spine_targets(obs_joints, smpl_model, betas=None, scale: float = 1.0):
    """§P16: redistribute the mid-spine targets along the SOURCE spine polyline at
    SMPL's β-rest arc-length fractions, so spine2/spine3 sit at REACHABLE heights.

    The 5-spine(source)→3-spine(SMPL) topology mismatch is a CORRESPONDENCE problem,
    not a length problem (the net torso length is already ~right, §P16 step-1 re-measure):
    SMPL's rest spine over-weights the lumbar (pelvis→spine1) and under-weights the
    thoracic, so the OBSERVED spine2/spine3 sit at heights the SMPL chain cannot reach
    (60–97 mm fit-err). Fix: trace the observed spine POLYLINE through the 5 torso
    points pelvis/spine1/spine2/spine3/neck (real source joints — the bend is in them)
    and resample spine1/2/3 at the fractional arc-lengths where SMPL's β-rest spine1/2/3
    sit along its own pelvis→neck chain. The points still lie on the source curve (lean
    preserved) but at SMPL's natural heights (reachable). Pelvis/neck/limbs unchanged.

    Then `measure_bone_ratios(obs', bones={3,6,9,12})` yields MILD, near-uniform ratios
    (each ≈ the net torso-length ratio L_src/(s·L_smpl_β,total) ≈ 1.0±0.1) — safe for the
    §P14 translation-form retarget (no spine3 balloon). Returns obs' [T,22,3] (a clone)."""
    dt = smpl_model.dtype
    obs = torch.as_tensor(obs_joints, dtype=dt).clone()
    betas = (torch.zeros(smpl_model.n_betas, dtype=dt) if betas is None
             else torch.as_tensor(betas, dtype=dt))
    v_shaped = smpl_model.v_template + torch.einsum(
        "vxb,b->vx", smpl_model.shapedirs, betas)
    J = (smpl_model.J_regressor @ v_shaped)[:N_JOINTS]           # β-shaped native rest
    chain = list(_TORSO_CHAIN)
    # SMPL β-rest cumulative arc-length fractions along pelvis→…→neck (scale-invariant)
    seg_smpl = torch.stack([(J[chain[i + 1]] - J[chain[i]]).norm()
                            for i in range(len(chain) - 1)])     # [4]
    cum_smpl = torch.cat([torch.zeros(1, dtype=dt), torch.cumsum(seg_smpl, 0)])
    frac = cum_smpl / cum_smpl[-1].clamp_min(1e-9)               # [5]: 0 … 1
    # source polyline through the OBSERVED torso points, per frame
    pts = obs[:, chain]                                          # [T,5,3]
    seg_src = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)           # [T,4]
    cum_src = torch.cat([torch.zeros(pts.shape[0], 1, dtype=dt),
                         torch.cumsum(seg_src, 1)], dim=1)       # [T,5]
    L_src = cum_src[:, -1]                                       # [T]
    for ci, jidx in zip((1, 2, 3), _SPINE_MID):                 # spine1/2/3
        tlen = frac[ci] * L_src                                  # [T] target arc-length
        idx = (cum_src <= tlen[:, None]).sum(1) - 1             # segment start
        idx = idx.clamp(0, len(chain) - 2)
        c0 = cum_src.gather(1, idx[:, None]).squeeze(1)
        c1 = cum_src.gather(1, (idx + 1)[:, None]).squeeze(1)
        w = ((tlen - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0, 1)
        p0 = pts.gather(1, idx[:, None, None].expand(-1, 1, 3)).squeeze(1)
        p1 = pts.gather(1, (idx + 1)[:, None, None].expand(-1, 1, 3)).squeeze(1)
        obs[:, jidx] = p0 + w[:, None] * (p1 - p0)
    return obs


def fit_smpl_betas_limblen(joints_obs, pose, smpl_model, n_frames: int = 12,
                           max_iter: int = 50, w_data: float = 1000.0,
                           lam_reg: float = 0.5, spine_weight: float = 0.1,
                           return_scale: bool = False, return_bone_err: bool = False):
    """P12: EasyMocap-style limb-LENGTH SMPL shape solve — a port of optimizeShape
    (pose_estimation/kinematics/optimization.py:785–836). β is a SINGLE free variable
    for the whole sequence, solved by LBFGS(strong_wolfe, max_iter=50) so SMPL's bone
    LENGTHS match the observed-keypoint bone lengths. Matching LENGTHS (not joint
    POSITIONS) DECOUPLES shape from pose — the bone DIRECTION is detached — so it is
    far more robust than a joint-position β-fit, which conflates shape with the
    topology-corrupted spine pose and inflates ‖β‖.

    Objective (optimizeShape closure): for each bone (p,c) and frame f,
        direct = unit(J_c − J_p).detach();  err = (J_c − J_p) − direct · ‖obs_c − obs_p‖
        loss = w_data · Σ_{f,bone} conf·‖err‖² / nFrames  +  lam_reg · ‖β‖²  (L2-β prior).
    w_data scales the metre²-scale length term into EasyMocap's high-weight s3d regime
    so the prior bites the girth null-space and NOT the length-determined β components.
    Bones ARRIVING at a SPINE_NECK_COLLAR joint are downweighted (spine_weight) because
    the BVH 5-spine→SMPL 3-spine topology mismatch corrupts those observed lengths.

    joints_obs: [T,22,3] observed Z-up joints (the subject's bone lengths).
    pose:       [T,22,3] SMPL LOCAL axis-angle for the current bone directions
                (detached in the length term; the GT/init pose is a fine choice).
    Returns β [n_betas]; with return_scale, (β, s) where s is the uniform SIZE scale;
    with return_bone_err, also the mean per-bone length error in m. §P12 (EasyMocap)."""
    dt = smpl_model.dtype
    obs = torch.as_tensor(joints_obs, dtype=dt)[:, :N_JOINTS]
    th = torch.as_tensor(pose, dtype=dt)
    T = obs.shape[0]
    idx = torch.linspace(0, T - 1, min(n_frames, T)).round().long()
    obs = obs[idx]
    th = th[idx]
    F = obs.shape[0]
    R = exp_so3(th.reshape(-1, 3)).reshape(F, N_JOINTS, 3, 3)
    eye = torch.eye(3, dtype=dt).expand(F, 24 - N_JOINTS, 3, 3)
    R = torch.cat([R, eye], dim=1)                                 # [F,24,3,3]
    M = _RX_NEG90.to(dt)                                           # SMPL native → Z-up

    par = torch.tensor([p for p, _ in _SMPL22_BONES])
    chi = torch.tensor([c for _, c in _SMPL22_BONES])
    limb_obs = (obs[:, chi] - obs[:, par]).norm(dim=-1)            # [F,B] observed lengths
    spine = set(SPINE_NECK_COLLAR)
    conf = torch.tensor([spine_weight if c in spine else 1.0       # downweight corrupted
                         for _, c in _SMPL22_BONES], dtype=dt)     #   arrival bones

    beta = torch.zeros(smpl_model.n_betas, dtype=dt, requires_grad=True)
    # §P9-redux: a uniform SIZE scale s (subject overall size, which β cannot carry
    # without inflating girth). When fitted, s absorbs the size and β stays small;
    # applied at render as a uniform vertex resize about the root. s=1 when not fitted.
    s = torch.ones((), dtype=dt, requires_grad=True)
    opt = torch.optim.LBFGS([beta, s] if return_scale else [beta],
                            line_search_fn='strong_wolfe', max_iter=max_iter)

    def closure():
        opt.zero_grad()
        v_shaped = smpl_model.v_template + torch.einsum(
            "vxb,b->vx", smpl_model.shapedirs, beta)
        J = smpl_model.J_regressor @ v_shaped                      # [24,3] native
        G = smpl_model._global_transforms(R, J)                    # [F,24,4,4]
        j = (M @ G[:, :N_JOINTS, :3, 3].transpose(-1, -2)).transpose(-1, -2)  # [F,22,3] Z-up
        src = j[:, par]
        dst = j[:, chi]
        direct = (dst - src).detach()
        direct = direct / (direct.norm(dim=-1, keepdim=True) + 1e-4)
        # s scales the SMPL bone vector: |s·(dst−src)| = s·smpl_len matched to obs_len,
        # so a uniform s captures the subject's SIZE and β only carries residual shape.
        err = (s if return_scale else 1.0) * (dst - src) - direct * limb_obs.unsqueeze(-1)
        data = ((err ** 2).sum(-1) * conf).sum() / F
        loss = w_data * data + lam_reg * (beta ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    beta = beta.detach()
    s_val = float(s.detach())
    out = (beta,)
    if return_scale:
        out = out + (s_val,)
    if return_bone_err:
        with torch.no_grad():
            v_shaped = smpl_model.v_template + torch.einsum(
                "vxb,b->vx", smpl_model.shapedirs, beta)
            J = smpl_model.J_regressor @ v_shaped
            blen = s_val * (J[chi] - J[par]).norm(dim=-1)          # s·SMPL bone lengths
            err_bone = (blen - limb_obs.mean(0)).abs()
            mask = conf > 0.5                                      # full-weight bones only
            out = out + (float(err_bone[mask].mean()),)
    return out if len(out) > 1 else beta


def solve_shape(obs_joints, init_pose, smpl_model, lam_reg: float = 0.3,
                spine_weight: float = 0.1):
    """Stage 1 (shape): the EasyMocap limb-length (beta, s) solve.

    A thin wrapper over the already-faithful port `fit_smpl_betas_limblen`
    (optimizeShape:785-836, LBFGS strong-Wolfe). Bone LENGTHS are frame-invariant,
    so the proper-frame obs is a valid observation; the prior keeps the girth
    null-space near the mean (§P9) while s carries the subject's true size.

    lam_reg=0.3 — REVERTED 2026-06-15 from the 2026-06-14 "fighter build" retune
    (lam_reg=0.1). The 0.1 value inflated ‖β‖→4.1 to deepen the torso, but β's girth
    null-space is UNCONSTRAINED by skeleton-only data, so it DISTORTED the buttock/hip
    mesh ~26 mm (the user's "jiggly ass" — §P17). MEASURED (shot_091/031 lam_reg sweep):
    the joint/bone fit-err is INVARIANT to β (≈flat across lam_reg 0.1→2.0) because §P14
    bone_scale carries lengths and s carries size — so β can be small at ~ZERO joint-fit
    cost. 0.3 ⇒ ‖β‖≈1.4 (in-distribution), buttock disp 26→6 mm. Build/size now come from
    s (≈1.15) + the §P14 leg retarget, NOT β's girth. This supersedes the §P9/motion_bridge
    "no lam_reg gives both length-match and in-distribution β" claim (which predated §P14).
    ONE coupling: smaller β ⇒ smaller s ⇒ ~2 mm more ankle lift (shot_031 60-frame gap
    23→24 mm, still ≤ G14c 25 mm). ↑lam_reg → cleaner shape but watch the ankle (G14c).
    Returns (beta [n_betas], s float)."""
    beta, s = fit_smpl_betas_limblen(
        obs_joints, init_pose, smpl_model, lam_reg=lam_reg,
        spine_weight=spine_weight, return_scale=True)
    return beta, s


# ── §P17 joint-limit machinery (port of phys_kintrack._add_limits + phys_body axes) ──
def _anatomical_limit_axes(joints_native):
    """Per-joint [22,3,3] anatomical frame (rows = the 3 ROM axes; the `axial` row is
    the rest bone direction) — port of phys_body._anatomical_limit_axes. The body axes
    are DERIVED from this model's rest geometry (up=pelvis→head, left=R-hip→L-hip,
    anterior=left×up) so it is frame-convention-agnostic. At the SMPL rest pose every
    joint's local frame equals the world frame, so these axes apply to the LOCAL pose
    rotation that `solve_pose` optimises. §P17."""
    J = joints_native
    dt = J.dtype
    P = list(SMPL_PARENTS)
    children = [[k for k in range(N_JOINTS) if P[k] == i] for i in range(N_JOINTS)]
    up = J[15] - J[0]; up = up / up.norm()                 # pelvis→head ≈ +Y
    lx = J[1] - J[2]; lx = lx / lx.norm()                  # R-hip→L-hip ≈ +X (LEFT)
    ant = torch.linalg.cross(lx, up); ant = ant / ant.norm()   # anterior ≈ +Z

    def bone(j):
        ch = children[j]
        v = (J[ch[0]] - J[j]) if ch else (J[j] - J[P[j]])
        nv = v.norm()
        return v / nv if nv > 1e-9 else torch.tensor([0.0, 0.0, 1.0], dtype=dt)

    def lateral(j):
        return lx if j in _LIMB_LEFT else -lx

    AX = None                                              # marks the axial (twist) slot
    spec = {}
    for j in (1, 2):   spec[j] = [ant, AX, lateral(j)]
    for j in (4, 5):   spec[j] = [-ant, AX, lateral(j)]
    for j in (7, 8):   spec[j] = [up, AX, lateral(j)]
    for j in (10, 11): spec[j] = [up, AX, lateral(j)]
    for j in (3, 6, 9): spec[j] = [ant, AX, lx]
    for j in (12, 15): spec[j] = [ant, AX, lx]
    for j in (13, 14): spec[j] = [up, ant, AX]
    for j in (16, 17): spec[j] = [ant, AX, up]
    for j in (18, 19): spec[j] = [ant, AX, up]
    for j in (20, 21): spec[j] = [ant, AX, up]
    axes = torch.eye(3, dtype=dt).repeat(N_JOINTS, 1, 1)
    for j in range(1, N_JOINTS):
        b = bone(j); s = spec[j]
        axial_ax = [k for k in range(3) if s[k] is AX][0]
        p0, p1 = [k for k in range(3) if s[k] is not AX]
        e0 = torch.linalg.cross(b, s[p0]); e0 = e0 / e0.norm()
        e1 = torch.linalg.cross(b, e0)
        e1 = e1 * torch.sign((e1 * torch.linalg.cross(b, s[p1])).sum())
        rows = torch.zeros(3, 3, dtype=dt)
        rows[axial_ax] = b; rows[p0] = e0; rows[p1] = e1
        axes[j] = rows
    return axes


def smpl_joint_limits(smpl_model, betas=None, bone_scale=None):
    """§P17 anatomical joint-limit data for this SMPL rest skeleton:
    (limit_axes [22,3,3], axial [22], lo [22,3], hi [22,3]). ROM bounds ported from
    phys_body; axes derived from the β/bone_scale-shaped rest joints."""
    dt = smpl_model.dtype
    J = _rest_joints_native(smpl_model, betas, bone_scale, N_JOINTS).to(dt)
    return (_anatomical_limit_axes(J), _LIMIT_AXIAL, _ROM_LO.to(dt), _ROM_HI.to(dt))


def _quat_mul(ax, ay, az, aw, bx, by, bz, bw):
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _quat_log(x, y, z, w):
    """Differentiable log of a unit quaternion → axis·angle [...,3], double-cover safe."""
    v = torch.stack([x, y, z], dim=-1)                     # [...,3]
    flip = (w < 0).to(v.dtype)
    v = v * (1.0 - 2.0 * flip)[..., None]
    w = w.abs()
    nv = v.norm(dim=-1).clamp_min(1e-12)                   # [...]
    return v * (2.0 * torch.atan2(nv, w) / nv)[..., None]


def _limit_residual(pose, limit_axes, axial, lo, hi, joints=_LIMIT_JOINTS):
    """§P17 differentiable joint-limit penalty — port of phys_kintrack._add_limits.
    Decompose each LOCAL joint rotation into TWIST (about the anatomical axial axis) +
    SWING (log of the swing quat), hinge-penalize each ROM axis outside [lo,hi]. Returns
    the mean-over-frames sum of squared violations (zero for in-ROM poses). VECTORISED
    over the limited joints (no Python joint loop — keeps the 900-iter Adam solve fast)."""
    T = pose.shape[0]
    js = torch.as_tensor(list(joints), dtype=torch.long)   # [J] limited joints
    jr = torch.arange(js.shape[0])
    ax = limit_axes[js]                                     # [J,3,3] anatomical axes
    axi = axial[js]                                         # [J] axial (twist) slot
    lo_j, hi_j = lo[js], hi[js]                             # [J,3]
    v = pose[:, js]                                         # [T,J,3] axis-angle
    ang = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * ang
    qxyz = v / ang * torch.sin(half)                       # [T,J,3]
    qw = torch.cos(half).squeeze(-1)                       # [T,J]
    a = ax[jr, axi]                                         # [J,3] twist axis per joint
    proj = (qxyz * a).sum(-1)                              # [T,J]
    twist = 2.0 * torch.atan2(proj, qw)                   # [T,J] twist angle
    tn = torch.sqrt(proj * proj + qw * qw).clamp_min(1e-12)
    tw = (proj / tn)[..., None] * a                        # [T,J,3] twist-quat vec
    tww = qw / tn                                          # [T,J] twist-quat w
    sx, sy, sz, sw = _quat_mul(qxyz[..., 0], qxyz[..., 1], qxyz[..., 2], qw,
                               -tw[..., 0], -tw[..., 1], -tw[..., 2], tww)   # q ⊗ conj(twist)
    swing = _quat_log(sx, sy, sz, sw)                     # [T,J,3] swing log
    swing_proj = torch.einsum('jkc,tjc->tjk', ax, swing)  # [T,J,3] swing onto each axis
    onehot = torch.zeros(js.shape[0], 3, dtype=pose.dtype)
    onehot[jr, axi] = 1.0                                  # [J,3] axial-slot mask
    theta = swing_proj * (1.0 - onehot)[None] + twist[..., None] * onehot[None]  # [T,J,3]
    viol = torch.relu(theta - hi_j[None]) + torch.relu(lo_j[None] - theta)
    return (viol * viol).sum() / T


def solve_pose(obs_joints, init_pose, smpl_model, Q, betas=None, scale=1.0,
               iters: int = 900, lr: float = 0.01, lam_anchor: float = 0.01,
               lam_smooth: float = 0.20, lam_contact: float = 8.0,
               lam_delta: float = 5.0, lam_delta_spine: float = 0.05,
               lam_rest: float = 0.1, rest_joints=(13, 14),
               lam_ankle: float = 0.5, ankle_joints=(7, 8),
               lam_knee: float = 0.15, knee_joints=(4, 5),
               sigma_sq: float = 0.25, foot_weight: float = 12.0,
               leg_weight: float = 2.0, return_offsets: bool = False,
               bone_scale=None, lam_limit: float = 0.0, lam_damp: float = 0.0):
    """Stage 2 (pose): the unified per-frame MAP pose fit with latent joint offsets.

    ONE optimization that replaces fit_smpl_global_convrot + refine_upper_body_convrot
    + fit_smpl_pose. The principled core is the SKELETON-MISMATCH handling: the observed
    skeleton's joints are NOT the same anatomical points as SMPL's J_regressor joints
    (most severely the BVH 5-spine collapsed into SMPL's 3-spine), so forcing
    ‖SMPL_j − obs_j‖→0 would DISTORT the pose to chase a convention gap. Following
    MoSh++'s LATENT MARKERS, we solve a per-joint body-attached offset delta_j (shared
    across all frames, L2-regularized) so the "virtual marker" marker_j = SMPL_j +
    s·R_world_j·delta_j matches obs_j while the BODY pose stays anatomically natural.

    Variables: theta_t (all joint rotations incl. root) and delta (22×3 latent offsets).
    Objective (all in the proper frame via the fast joints-only FK):
      data    = mean_j w_j · gmof(‖marker_j − obs_j‖², sigma²)        (robust, MoSh marker)
      anchor  = mean ‖theta − init‖²                       (BVH/pred orientation prior)
      smooth  = mean torso>limb · ‖R_{t+1}−2R_t+R_{t-1}‖²_F  (2nd-order chordal accel)
      contact = ground-penetration + in-contact anti-skate on the BODY feet
      delta   = ‖delta‖²                                    (latent-offset prior, MoSh E_R)
    Adam (stable on the long sequence; LBFGS is reserved for the shape stage).

    obs_joints: [T,22,3] proper-frame observed joints (GT skeleton OR pred joints).
    init_pose:  [T,22,3] LOCAL axis-angle anchor (BVH retarget for GT; world_transforms_to_smpl
                of the predicted body_log for pred).
    Returns pose [T,22,3] (and, with return_offsets, the latent delta [22,3]) for
    `forward_proper` — which renders the natural BODY (delta is the data correspondence,
    not the mesh)."""
    dt = smpl_model.dtype
    obs = torch.as_tensor(obs_joints, dtype=dt)[:, :N_JOINTS]
    T = obs.shape[0]
    root_target = obs[:, 0]
    init = torch.as_tensor(init_pose, dtype=dt).clone()
    QM = torch.as_tensor(Q, dtype=dt) @ _RX_NEG90.to(dt)         # native → proper
    s = float(scale)
    # Per-joint pose anchor. Most joints anchor to the BVH init (orientation incl. twist).
    # The CLAVICLE (collars) is special: the Mixamo clavicle sits FORWARD of SMPL's, so
    # anchoring it to the BVH transfer rotates the SMPL clavicle forward → HUNCH. Since δ
    # already absorbs the clavicle's forward POSITION (marker), its ROTATION must anchor to
    # REST (the anatomically-natural, near-static clavicle) instead — decoupling the
    # convention gap (→δ) from the body orientation (→rest). aw weights the anchor per joint.
    atgt = init.clone()
    aw = torch.full((N_JOINTS,), lam_anchor, dtype=dt)
    for jdx in rest_joints:
        atgt[:, jdx] = 0.0                                       # rest = natural clavicle
        aw[jdx] = lam_rest
    # The ANKLES carry the foot PITCH. The BVH init already has the correct ball-of-foot
    # pitch (≈ the dataset's heel-raised stance). But at scale s>1 the over-long SMPL shin
    # would otherwise DROP the ankle to keep the toe grounded → FLATTENED foot, buried heel
    # mesh, "toe in the air" (measured: pitch −23°→−8°, ankle 40mm low, sole −20mm under the
    # floor). Anchoring the ankle ROTATION to its (correct) init preserves the pitch while
    # the strong foot data weight still grounds the toe (pitch −8°→−17°, sole −20→−4mm).
    for jdx in ankle_joints:
        aw[jdx] = lam_ankle                                      # keep correct init foot pitch
    # The KNEES carry the leg BEND. The dataset legs are near-straight (knee ≈167°) and the BVH
    # init already has that bend (≈161°), but the strong foot data terms (foot_weight on ankle+toe)
    # pull the lower leg so hard that the optimiser OVER-BENDS the knee (≈134°, 33° off) and shoves
    # it ~122mm off its obs position to nail the feet — the "foot fitting ruins the knee" artifact.
    # Anchoring the knee ROTATION to its (good) init bend — exactly as lam_ankle does for the foot
    # pitch and lam_rest for the collar — keeps the leg natural while the feet still ground.
    # lam_knee=0.15 (user 2026-06-14: "ankle/toe fitting ruined the knee, fix that"): knee angle err
    # 33.7°→24.9°, knee pos 122→94mm. NOTE a real TRADEOFF — knee bend and foot grounding are coupled
    # through SMPL's fixed (retargeted) bone lengths, so straightening the knee lifts the ankle (0.15 →
    # ankle ~21mm, still within G13d/G13k; 0.2→29mm, 0.3→knee 16° but ankle 40mm/foot floats). 0.15 is
    # the balance that de-ruins the knee while keeping the feet grounded; raise it for a straighter knee.
    for jdx in knee_joints:
        aw[jdx] = lam_knee                                       # keep correct init knee bend

    w = torch.ones(N_JOINTS, dtype=dt)                           # data weights
    w[_FEET] = foot_weight
    w[_LEGS] = leg_weight
    sw = torch.ones(N_JOINTS, dtype=dt)                          # smoothness weights
    # The slow PROXIMAL body — torso AND the whole lower body (hips/knees/ankles/feet) — is a
    # near-static stance and must be smoothed STRONGLY so it does not wobble while the data term
    # chases noisy per-frame foot/leg targets. Only the FAST distal arms stay lightly smoothed
    # (sw=1) so strikes keep their speed. The previous sw=1 on the legs (vs torso 3) under-smoothed
    # the lower body → the whole-lower-body wobble; this is the structural fix (not a weight hack).
    sw[_TORSO] = 3.0
    sw[_LEGS] = 3.0
    sw[_FEET] = 3.0
    # Per-joint latent-offset prior: WEAK on the whole UPPER-BODY convention-mismatch
    # region — the spine/neck/collars (BVH 5-spine→SMPL 3-spine gap, §P9) PLUS the head
    # and shoulders, whose Mixamo placement differs from SMPL's clavicle/neck/head
    # anatomy. Letting δ absorb that mismatch keeps the BODY head/neck/shoulders natural
    # (no hunch/neck-stretch) while the markers still reach obs. STRONG elsewhere (root,
    # hips/knees/ankles/feet, elbows/wrists) so those match by POSE (δ≈0) — arms reach
    # the dataset via the elbow/wrist data term, not by floating the shoulder. A tensor
    # lam_delta is used verbatim (caller override).
    if torch.is_tensor(lam_delta) and lam_delta.numel() == N_JOINTS:
        dw = lam_delta.to(dt)
    else:
        dw = torch.full((N_JOINTS,), float(lam_delta), dtype=dt)
        for jdx in _DELTA_FREE:
            dw[jdx] = lam_delta_spine
    contact, floor = detect_contacts(obs)                       # [T,F] bool, floor z
    cm = contact.to(dt)

    def marker_and_body(pose, delta):
        jnat, Rnat = _fk(smpl_model, pose, betas, bone_scale=bone_scale)   # native + world-rot (§P14 retarget)
        jp = torch.einsum("mn,tjn->tjm", QM, jnat)              # → proper
        Reff = torch.einsum("mn,tjnk->tjmk", QM, Rnat)          # joint world-rot in proper
        jp = jp + (root_target[:, None, :] - jp[:, :1])         # ground root to obs root
        jp = root_target[:, None, :] + s * (jp - root_target[:, None, :])   # scale
        off = torch.einsum("tjmk,jk->tjm", Reff, delta)         # body-attached offset
        return jp + s * off, jp                                 # (markers, body joints)

    pose = init.clone().requires_grad_(True)
    delta = torch.zeros(N_JOINTS, 3, dtype=dt, requires_grad=True)
    opt = torch.optim.Adam([pose, delta], lr=lr)
    # §P17 anatomical kinematics regularizers (OFF by default ⇒ §P13 unchanged, G17e).
    if lam_limit > 0:
        _lim_axes, _lim_ax, _lim_lo, _lim_hi = smpl_joint_limits(smpl_model, betas, bone_scale)
    damp_w = _DAMP_SCALAR.to(dt)
    for _ in range(iters):
        opt.zero_grad()
        marker, jp = marker_and_body(pose, delta)
        sq = ((marker - obs) ** 2).sum(-1)                      # [T,22] marker residual
        data = (gmof(sq, sigma_sq) * w).mean()
        anchor = (aw * ((pose - atgt) ** 2).sum(-1)).mean()
        # GEODESIC temporal smoothness — 2nd-order chordal ACCELERATION on the joint
        # rotations: ‖R_{t+1}−2R_t+R_{t-1}‖²_F. Chordal (not axis-angle) keeps it
        # double-cover-safe, so it still forbids the distal-joint 180° TWIST FLIP (a flip is
        # a huge frame-to-frame acceleration). Crucially this is ACCELERATION (2nd order), not
        # VELOCITY (1st order ‖R_t−R_{t+1}‖²): a velocity penalty's loss-minimizer is a STATIC
        # pose, so it drains real upper-body speed (measured on shot_031: arms→57%, torso/
        # shoulders→76% of the BVH source). Acceleration removes only JERK and passes sustained
        # fast motion (arms→82%, torso→102% at the SAME lam_smooth, ~0 data-fit cost). Restores
        # the §P13 / docstring spec. sw [N_JOINTS] broadcasts over the [T-2, N_JOINTS] term.
        # lam_smooth=0.20 (was 0.05) — user retune 2026-06-14 ("lower body wobbles"). The WOBBLE FIX is
        # primarily the sw change above (legs+feet smoothed 3× like the torso, was 1× → under-smoothed
        # → the whole-lower-body wobble); raising the global lam_smooth 0.05→0.20 compounds it. Together:
        # lower-body 2nd-order accel 4.5×→~0.5× source (smoother than source) while arms stay fast
        # (G13j ≥0.75 — accel-smoothing passes sustained motion). Lowering foot_weight to "use the same
        # weights everywhere" was TESTED and rejected: it floats the feet ~5cm / flattens them (the
        # natural straight leg can't reach the grounded foot — a knee↔foot geometry conflict, see §P13).
        Rloc = exp_so3(pose.reshape(-1, 3)).reshape(T, N_JOINTS, 3, 3)
        if T >= 3:
            accel = Rloc[2:] - 2 * Rloc[1:-1] + Rloc[:-2]
            smooth = ((accel ** 2).sum((-1, -2)) * sw).mean()
        elif T >= 2:                                            # T==2: fall back to velocity
            dR = Rloc[1:] - Rloc[:-1]
            smooth = ((dR ** 2).sum((-1, -2)) * sw).mean()
        else:
            smooth = torch.zeros((), dtype=dt)
        # §P14-era retune: lam_contact was 20 (§P13). At 20 the ground term OVER-grounds
        # the toe and FLATTENS the foot — the ankle drops ~30mm below the dataset ankle and
        # the foot pitch collapses (dataset 22.8° → solved ~21°→flat). The ankle-low residual
        # the §P14 continuation blamed on leg PROPORTION was actually THIS: at lam_contact=8
        # the foot pitches naturally (23.5°), the actor ankle gap 30→1mm (reactor 4→1.5mm),
        # and in-contact skate IMPROVES (2.70→1.15 mm/frame), for +2mm penetration. The leg
        # retarget (bone_scale) is orthogonal — it fixes bone LENGTHS, not foot pitch.
        zf = jp[:, _FOOT_GROUND, 2]                             # BODY foot heights
        ground_pen = (torch.relu(floor - zf) ** 2 * cm).mean()
        if T > 1:
            skate = ((jp[1:, _FOOT_GROUND, :2] - jp[:-1, _FOOT_GROUND, :2]) ** 2
                     ).sum(-1)
            skate = (skate * cm[1:]).mean()
        else:
            skate = torch.zeros((), dtype=dt)
        delta_prior = (dw * (delta ** 2).sum(-1)).sum()
        loss = (data + anchor + lam_smooth * smooth
                + lam_contact * (ground_pen + skate) + delta_prior)
        # §P17 joint limits (anatomical ROM hinge) + passive damping. Damping is the
        # ACCELERATION form (per-joint anatomical weight on the SAME chordal 2nd-order
        # term as `smooth`), NOT a 1st-order velocity penalty: §P13 (smpl_fit:564-579)
        # proved a velocity penalty drains real motion, and G17c/d confirmed the literal
        # `phys_avbd` velocity-damping form both drains the arms AND fails to cut jitter.
        # The anatomical `passive_damping` (HIGH spine/proximal, LOW distal) thus modulates
        # how hard each joint is jitter-smoothed — the §P17 plan's §P13-consistent fallback.
        if lam_limit > 0:
            loss = loss + lam_limit * _limit_residual(pose, _lim_axes, _lim_ax, _lim_lo, _lim_hi)
        if lam_damp > 0 and T >= 3:
            loss = loss + lam_damp * ((accel ** 2).sum((-1, -2)) * damp_w[None]).mean()
        loss.backward()
        opt.step()
    return (pose.detach(), delta.detach()) if return_offsets else pose.detach()


def fit_markers(obs_markers, marker_bone, smpl_model, Q, betas_init=None,
                bone_scale=None, n_shape_frames: int = 12,
                iters_shape: int = 500, iters_pose: int = 500, lr: float = 0.02,
                lam_smooth: float = 0.10, lam_delta: float = 0.02,
                lam_beta: float = 1.0, lam_contact: float = 0.0,
                lam_anchor: float = 0.5, lam_limit: float = 1.0,
                sigma_sq: float = 0.5, sigma_sq0: float = 0.5):
    """§P23 — true-MoSh surface-marker SMPL solve for a joint-INCONGRUENT reference.

    The reference points are SURFACE markers, NOT SMPL joints (ExPI: hip/knee/heel on
    the skin, a single upper-back marker, no pelvis/spine/ankle). Each marker m is
    rigidly attached to a SMPL bone `marker_bone[m]` with a FREE frame-invariant offset
    delta_m (the surface→joint vector, weakly regularized — NOT the §P13 strong prior):
        v_m(theta,beta,delta) = body_{b(m)} + s·R^world_{b(m)}·delta_m
    where body = trans + s·(FK_proper − FK_root). Two MoSh++ stages:
      Stage I  — on a sequence-SPANNING subset of `n_shape_frames` (diverse poses break
                 the pose/offset degeneracy): solve shape beta + uniform size s +
                 per-marker delta + the subset's theta/trans, minimizing robust
                 ‖v−obs‖² + lam_beta‖beta‖² + lam_delta‖delta‖².
      Stage II — freeze beta,s,delta; solve per-frame theta_t + free root trans_t
                 (ExPI has no pelvis marker to ground to), with 2nd-order chordal-accel
                 smoothness (+ optional contact). MoSh: Loper SIGGRAPH Asia 2014;
                 MoSh++/AMASS arXiv:1904.03278; ExPI arXiv:2105.08825.

    obs_markers [T,M,3] proper-frame markers; marker_bone [M] long SMPL joint indices.
    Returns (pose [T,22,3], trans [T,3], beta [n_betas], s float, delta [M,3])."""
    dt = smpl_model.dtype
    obs = torch.as_tensor(obs_markers, dtype=dt)
    T, M = obs.shape[0], obs.shape[1]
    mb = torch.as_tensor(marker_bone, dtype=torch.long)
    QM = torch.as_tensor(Q, dtype=dt) @ _RX_NEG90.to(dt)         # native → proper
    nb = smpl_model.n_betas
    # root proxy = mean of the hip-attached markers (fallback: marker centroid) for trans init.
    hipm = torch.tensor([k for k in range(M) if mb[k].item() in (1, 2)], dtype=torch.long)
    root0 = obs[:, hipm].mean(1) if hipm.numel() else obs.mean(1)   # [T,3]
    sw = torch.ones(N_JOINTS, dtype=dt)
    sw[_TORSO] = 3.0; sw[_LEGS] = 3.0; sw[_FEET] = 3.0
    sw[0] = 1.0          # root carries REAL global rotation (not jitter) — don't over-smooth;
                         # the lateral hip markers (attached to the pelvis) lag if it is (§P23)
    # No marker hits the spine1/2/3, neck or collars, so those joints are FREE to curl into an
    # implausible torso (the §P23 pot-belly/hunch) that still satisfies the sparse markers.
    # Anchor THEM to rest (straight spine) — strong where unconstrained, zero on the
    # marker-pinned limbs/head/root. Plus §P17 anatomical joint limits keep every joint in ROM.
    aw = torch.zeros(N_JOINTS, dtype=dt)
    aw[[3, 6, 9, 12, 13, 14]] = lam_anchor
    if lam_limit > 0:
        _lim_axes, _lim_ax, _lim_lo, _lim_hi = smpl_joint_limits(smpl_model, None, bone_scale)

    def markers_of(pose, trans, s, beta, delta):
        jnat, Rnat = _fk(smpl_model, pose, beta, bone_scale=bone_scale)
        jp = torch.einsum("mn,tjn->tjm", QM, jnat)
        Reff = torch.einsum("mn,tjnk->tjmk", QM, Rnat)
        body = trans[:, None, :] + s * (jp - jp[:, :1])
        off = torch.einsum("tmkl,ml->tmk", Reff[:, mb], delta)
        return body[:, mb] + s * off, body

    # ExPI/2C are CLEAN mocap (no marker outliers), so the GMOF runs at a LARGE σ
    # (≈L2 over the whole working range) — a redescending small-σ robustifier dead-zones
    # any marker beyond ~1.5σ, freezing whichever lateral hip/head marker is still far
    # when σ shrinks (measured §P23: 250 mm stuck hips at σ²=0.0025; L2 → 1 mm). `_sig`
    # keeps a coarse→fine hook (sigma_sq0→sigma_sq) for a future noisy source; default
    # sigma_sq0==sigma_sq ⇒ constant large σ, gradient everywhere from zero-init.
    def _sig(i, n):
        return sigma_sq0 * (sigma_sq / sigma_sq0) ** (i / max(n - 1, 1))

    def data_term(v, sig):
        return gmof(((v - obs) ** 2).sum(-1), sig).mean()

    def smooth_term(pose, T_):
        Rloc = exp_so3(pose.reshape(-1, 3)).reshape(T_, N_JOINTS, 3, 3)
        if T_ >= 3:
            accel = Rloc[2:] - 2 * Rloc[1:-1] + Rloc[:-2]
            return ((accel ** 2).sum((-1, -2)) * sw).mean()
        if T_ >= 2:
            dR = Rloc[1:] - Rloc[:-1]
            return ((dR ** 2).sum((-1, -2)) * sw).mean()
        return torch.zeros((), dtype=dt)

    # ── ONE joint solve over ALL frames: shared beta+s+delta, per-frame theta_t+trans_t.
    # delta (the surface→bone offset) is frame-INVARIANT but estimated from the WHOLE
    # sequence — a 12-frame freeze (the MoSh++ efficiency trick) leaves the torso markers
    # (hip/back/head) under-determined (spine/pelvis gauge freedom) and does NOT generalize
    # (§P23: frozen-δ gave 200 mm hip residuals). All frames removes that gap.
    beta = (torch.zeros(nb, dtype=dt) if betas_init is None
            else torch.as_tensor(betas_init, dtype=dt).clone()).requires_grad_(True)
    log_s = torch.zeros((), dtype=dt, requires_grad=True)        # s = exp(log_s) > 0
    delta = torch.zeros(M, 3, dtype=dt, requires_grad=True)
    pose = torch.zeros(T, N_JOINTS, 3, dtype=dt, requires_grad=True)
    trans = root0.clone().requires_grad_(True)
    contact, floor = detect_contacts(obs) if lam_contact > 0 else (None, None)
    cm = contact.to(dt) if contact is not None else None
    n_it = iters_shape + iters_pose
    opt = torch.optim.Adam([beta, log_s, delta, pose, trans], lr=lr)
    for i in range(n_it):
        opt.zero_grad()
        v, body = markers_of(pose, trans, torch.exp(log_s), beta, delta)
        loss = (data_term(v, _sig(i, n_it)) + lam_smooth * smooth_term(pose, T)
                + lam_beta * (beta ** 2).mean() + lam_delta * (delta ** 2).sum()
                + (aw * (pose ** 2).sum(-1)).mean())
        if lam_limit > 0:
            loss = loss + lam_limit * _limit_residual(pose, _lim_axes, _lim_ax, _lim_lo, _lim_hi)
        if lam_contact > 0:
            zf = body[:, _FOOT_GROUND, 2]
            ground_pen = (torch.relu(floor - zf) ** 2 * cm).mean()
            if T > 1:
                skate = ((body[1:, _FOOT_GROUND, :2] - body[:-1, _FOOT_GROUND, :2]) ** 2
                         ).sum(-1)
                skate = (skate * cm[1:]).mean()
            else:
                skate = torch.zeros((), dtype=dt)
            loss = loss + lam_contact * (ground_pen + skate)
        loss.backward(); opt.step()
    beta = beta.detach(); s = float(torch.exp(log_s).detach()); delta = delta.detach()
    return pose.detach(), trans.detach(), beta, s, delta


# ═══════════════════════════════════════════════════════════════════════════
# §P15 — Damped-Gauss-Newton reduced-coordinate pose solver.
#
# Rewrites the §P13 anchor-tower `solve_pose` (above, kept for `--fit unified`).
# Port of pose_estimation/kinematics/phys_kintrack.py::KinematicPoseTracker — a
# damped Gauss-Newton solver over the SMPL reduced coordinates (root SO(3) + 21
# joint SO(3); bone lengths FIXED by the §P13 β-shaped + §P14 bone_scale rest
# joints ⇒ exact lengths by FK construction). It fixes the four structural
# wobble causes the anchor tower could only band-aid (see KOOPMAN_MASTER_PLAN.md
# §P15 R1–R6):
#   R1  robust data term ACTIVE at cm scale (GMOF IRLS, σ²=0.04) — not σ²=0.25.
#   R2  Tikhonov/LM damping + per-step clamp = damped-least-squares cure of the
#       straight-leg IK singularity (phys_kintrack:315,389-393).
#   R3  2nd-order acceleration smoothing on WORLD joint positions (the space the
#       data lives in), not lever-blind local rotations.
#   R4  Gauss-Newton with Lie retraction (phys_kintrack:386-400) — monotone, no
#       Adam random walk; warm-started per frame.
#   R5  fps read per sequence; contact threshold in m/s; targets pre-filtered.
#   R6  NO per-joint anchors / no "static lower body" weighting — plausibility is
#       emergent from conditioning + the constant-velocity prior (MoSh++ §3).
# The §P13/§P14 plumbing (compute_Q, forward_proper, the rest-joint retarget,
# the δ latent markers) is KEPT; this only replaces the per-frame pose optimiser.
# ═══════════════════════════════════════════════════════════════════════════

_GN_DT = torch.float64


def read_fps(bvh_path, default: float = 50.0):
    """fps from a BVH `Frame Time:` line (§P15 R5 — never assume FPS=60)."""
    try:
        with open(bvh_path) as f:
            for line in f:
                if 'Frame Time' in line:
                    ft = float(line.split(':')[1].strip())
                    if ft > 1e-6:
                        return 1.0 / ft
    except Exception:
        pass
    return float(default)


def prefilter_targets(obs_joints, fps: float, window_s: float = 0.15,
                      polyorder: int = 3):
    """Stage-0 non-causal Savitzky-Golay low-pass on the observed 3D targets.

    Recorded data ⇒ offline (non-causal) smoothing is appropriate; SG preserves
    kick/airstep peaks while removing the frame-to-frame jitter that the singular
    leg direction (R2) would otherwise amplify (R1/R5). Window is fps-aware
    (~window_s seconds). Returns a float64 tensor [T,J,3]."""
    obs = np.asarray(obs_joints, dtype=np.float64)
    T = obs.shape[0]
    win = int(round(window_s * fps))
    if win % 2 == 0:
        win += 1
    win = max(5, win)
    if win >= T:
        win = T if (T % 2 == 1) else T - 1
    if win < 5 or win <= polyorder:
        return torch.as_tensor(obs, dtype=_GN_DT)
    po = min(polyorder, win - 1)
    sm = savgol_filter(obs, window_length=win, polyorder=po, axis=0)
    return torch.as_tensor(sm, dtype=_GN_DT)


def _chain_mask(parents, n: int):
    """[n,n] bool: mask[j,a]=True iff joint a is an ancestor-or-self of j (the
    joints whose rotation moves joint j — the support of the geometric Jacobian)."""
    mask = torch.zeros(n, n, dtype=torch.bool)
    for j in range(n):
        a = j
        while a >= 0:
            mask[j, a] = True
            a = int(parents[a])
    return mask


def _fk_mats(R_local, J, parents, n: int):
    """Native FK from per-joint LOCAL rotation MATRICES (matches
    vizsys.smpl._global_transforms, pre-_TRANS): t[j]=t[p]+Rw[p]·(J[j]−J[p]),
    Rw[j]=Rw[p]·R_local[j]. R_local [n,3,3], J [n,3]. Returns (t [n,3], Rw [n,3,3])."""
    t = [None] * n
    Rw = [None] * n
    for j in range(n):
        p = int(parents[j])
        if p < 0:
            t[j] = J[j]
            Rw[j] = R_local[j]
        else:
            Rw[j] = Rw[p] @ R_local[j]
            t[j] = t[p] + Rw[p] @ (J[j] - J[p])
    return torch.stack(t), torch.stack(Rw)


def _rest_joints_native(smpl_model, betas, bone_scale, n: int):
    """β-shaped + §P14 bone_scale-retargeted rest JOINTS (native), sliced to n.
    Mirrors forward_R's joint construction so the GN forward ≡ forward_proper."""
    dt = _GN_DT
    betas = (torch.zeros(smpl_model.n_betas, dtype=dt) if betas is None
             else torch.as_tensor(betas, dtype=dt))
    v_shaped = smpl_model.v_template.to(dt) + torch.einsum(
        "vxb,b->vx", smpl_model.shapedirs.to(dt), betas)
    J = smpl_model.J_regressor.to(dt) @ v_shaped                 # [24,3] full tree
    if bone_scale is not None:
        J = retarget_joints(J, torch.as_tensor(bone_scale, dtype=dt),
                            smpl_model.parents)
    return J[:n]


def solve_pose_gn(obs_joints, init_pose, smpl_model, Q, betas=None, scale=1.0,
                  bone_scale=None, fps: float = 50.0, delta=None,
                  n_iters: int = 14, sigma_sq: float = 0.04,
                  w_smooth: float = 3.0, w_ground: float = 10.0,
                  w_skate: float = 10.0, foot_weight: float = 8.0,
                  tikhonov: float = 1e-3, max_step_rot: float = 0.5,
                  h_thresh: float = 0.05, v_thresh_mps: float = 0.25,
                  prefilter: bool = True, return_extra: bool = False):
    """§P15 damped-Gauss-Newton pose solve. Drop-in replacement for `solve_pose`.

    Solves per-frame SMPL LOCAL pose (root SO(3) + 21 joint SO(3); root TRANSLATION
    pinned to the observed root, as `forward_proper` grounds it) against the
    pre-filtered observed 3D joints, in the PROPER frame. The body is mapped
    native→proper by the SAME `sQM = s·Q·_TRANS` + root-grounding that
    `forward_proper` applies, so the solved θ feeds `forward_proper` unchanged
    (gate G15k). Reduced-coordinate damped GN with analytic geometric Jacobian and
    Lie-tangent retraction — no autodiff, no per-joint anchors.

    obs_joints [T,J,3] proper-frame observed joints (GT skeleton or pred).
    init_pose  [T,J,3] LOCAL axis-angle warm start (BVH retarget / pred body_log).
    betas,scale,bone_scale: the frozen §P13/§P14 shape. delta [J,3]: frozen
    upper-body latent markers (default 0 — lower-body wobble is δ-independent).
    Returns pose [T,J,3] LOCAL axis-angle (and, with return_extra, a diagnostics dict)."""
    dt = _GN_DT
    n = N_JOINTS
    parents = list(SMPL_PARENTS)
    obs = torch.as_tensor(obs_joints, dtype=dt)[:, :n]
    if prefilter:
        obs = prefilter_targets(obs, fps)[:, :n]
    T = obs.shape[0]
    root_target = obs[:, 0]                                      # [T,3] pinned root
    s = float(scale)
    QM = torch.as_tensor(Q, dtype=dt) @ _RX_NEG90.to(dt)        # native→proper (pre-scale)
    sQM = s * QM
    J = _rest_joints_native(smpl_model, betas, bone_scale, n)   # [n,3] native rest
    delta_t = (torch.zeros(n, 3, dtype=dt) if delta is None
               else torch.as_tensor(delta, dtype=dt)[:n])
    chain = _chain_mask(parents, n)                             # [n,n] bool
    Rinit = exp_so3(torch.as_tensor(init_pose, dtype=dt)[:, :n].reshape(-1, 3)
                    ).reshape(T, n, 3, 3)

    # fps-aware foot contact (v_thresh in m/s → m/frame), on the pre-filtered obs.
    contact, floor = detect_contacts(obs, h_thresh=h_thresh,
                                     v_thresh=v_thresh_mps / max(fps, 1e-6))
    cm = contact.to(dt)                                         # [T,F]
    F = len(_FOOT_GROUND)
    wdata = torch.ones(n, dtype=dt)
    wdata[_FEET] = foot_weight
    I66 = torch.eye(3 * n, dtype=dt)

    solved_R = torch.zeros(T, n, 3, 3, dtype=dt)
    proper_hist = torch.zeros(T, n, 3, dtype=dt)
    loss_hist = []

    def forward(Rl, rt):
        t_nat, Rw = _fk_mats(Rl, J, parents, n)
        off = torch.einsum('jmk,jk->jm', Rw, delta_t)          # R_world_j · δ_j
        p_nat = t_nat + off                                    # marker (native, abs)
        proper = rt + torch.einsum('mn,jn->jm', sQM, p_nat - t_nat[0])
        return t_nat, Rw, p_nat, proper

    def jac(p_nat, t_nat, Rw):
        # native geometric Jacobian: block[j,a] = -hat(p_j - t_a) · Rw_a  (ω×r)
        diff = p_nat[:, None, :] - t_nat[None, :, :]           # [n,n,3]
        block = -torch.einsum('jaxy,ayk->jaxk', hat_so3(diff), Rw)   # [n,n,3,3]
        block = block * chain[:, :, None, None]                # zero non-ancestors
        # native→proper, ordered [j, proper-xyz m, dof (a*3+k)] for the GN basis.
        Jp = torch.einsum('mx,jaxk->jmak', sQM, block)         # [n,3,n,3]
        return Jp.reshape(n, 3, 3 * n)                         # [n,3,66]

    for ti in range(T):
        Rl = (solved_R[ti - 1].clone() if ti >= 1 else Rinit[0].clone())
        rt = root_target[ti]
        for _ in range(n_iters):
            t_nat, Rw, p_nat, proper = forward(Rl, rt)
            Jp = jac(p_nat, t_nat, Rw)                          # [n,3,66]
            res = proper - obs[ti]                              # [n,3] data residual
            sq = (res ** 2).sum(-1)                             # [n]
            rw = sigma_sq ** 2 / (sigma_sq + sq) ** 2           # GMOF IRLS weight (R1)
            wd = (wdata * rw)[:, None, None]                    # [n,1,1]
            H = torch.einsum('jmd,jme->de', wd * Jp, Jp)
            g = torch.einsum('jmd,jm->d', wd * Jp, res)
            # 2nd-order WORLD acceleration smoothing (R3): pull markers toward the
            # constant-velocity extrapolation of the two solved neighbours.
            if ti >= 2:
                extrap = 2 * proper_hist[ti - 1] - proper_hist[ti - 2]
                res_s = proper - extrap
                H = H + w_smooth * torch.einsum('jmd,jme->de', Jp, Jp)
                g = g + w_smooth * torch.einsum('jmd,jm->d', Jp, res_s)
            elif ti == 1:
                res_s = proper - proper_hist[0]                 # velocity→0 (mild)
                H = H + 0.5 * w_smooth * torch.einsum('jmd,jme->de', Jp, Jp)
                g = g + 0.5 * w_smooth * torch.einsum('jmd,jm->d', Jp, res_s)
            # contact: floor non-penetration + in-contact anti-skate (foot joints).
            for fi, fj in enumerate(_FOOT_GROUND):
                c = float(cm[ti, fi])
                if c <= 0:
                    continue
                Jz = Jp[fj, 2, :]                               # ∂(foot z)/∂q
                pen = float(proper[fj, 2] - floor)
                if pen < 0:
                    H = H + w_ground * torch.outer(Jz, Jz)
                    g = g + w_ground * Jz * pen
                if ti >= 1:
                    Jxy = Jp[fj, :2, :]                         # [2,66]
                    res_xy = proper[fj, :2] - proper_hist[ti - 1][fj, :2]
                    H = H + w_skate * (Jxy.t() @ Jxy)
                    g = g + w_skate * (Jxy.t() @ res_xy)
            H = H + tikhonov * I66                              # LM/DLS damping (R2)
            dq = torch.linalg.solve(H, -g).reshape(n, 3)
            nr = dq.norm(dim=1)
            mr = float(nr.max())
            if mr > max_step_rot:                               # step clamp (R2/R4)
                dq = dq * (max_step_rot / mr)
            Rl = Rl @ exp_so3(dq)                               # Lie retraction
        t_nat, Rw, p_nat, proper = forward(Rl, rt)
        solved_R[ti] = Rl
        proper_hist[ti] = proper
        loss_hist.append(float(((proper - obs[ti]) ** 2).sum()))

    pose = log_so3(solved_R.reshape(-1, 3, 3)).reshape(T, n, 3)
    if return_extra:
        return pose, {'proper': proper_hist, 'obs_filt': obs, 'floor': floor,
                      'contact': cm, 'loss_hist': loss_hist, 'fps': fps}
    return pose


def solve_pose_gn_global(obs_joints, init_pose, smpl_model, Q, betas=None, scale=1.0,
                         bone_scale=None, fps: float = 50.0, delta=None,
                         n_iters: int = 12, sigma_sq: float = 0.04,
                         w_smooth: float = 3.0, w_smooth_upper=None, w_ground: float = 10.0,
                         foot_weight: float = 8.0, tikhonov: float = 1e-3,
                         max_step_rot: float = 0.5, h_thresh: float = 0.05,
                         v_thresh_mps: float = 0.25, prefilter: bool = True,
                         return_extra: bool = False):
    """§P15.1 GLOBAL/batch damped-Gauss-Newton pose solve — the STABLE smoother.

    Same reduced-coordinate engine, analytic geometric Jacobian, GMOF-robust data
    term, fps-aware ground, Lie retraction, and native→proper plumbing as the
    per-frame `solve_pose_gn` (so `forward_proper` consumes the θ unchanged), BUT
    the 2nd-order WORLD-acceleration smoothness is solved IMPLICITLY over ALL frames
    jointly — one block-penta-diagonal linear solve per GN iteration — instead of
    the per-frame causal pull toward an EXPLICIT `2·pₜ₋₁−pₜ₋₂` extrapolation.

    Why this is stable (§P15.1): the per-frame causal form is a forward-Euler
    recursion that diverges on fast motion once GMOF attenuates the data restoring
    force (measured: fit 29→212→486 mm as w_smooth 0→3→10 on fast LindyHop). Here
    the const-velocity prior is part of one joint normal-equation solve — implicit,
    unconditionally stable (UNIFIED's stability) but GN-fast and exact-bone.

    Structure: with root TRANS pinned (`forward_proper` grounds it), proper[t]
    depends only on q[t] ⇒ E_data is block-diagonal. The penalty
    `Σₜ w·‖pₜ₊₁−2pₜ+pₜ₋₁‖²` linearises to `H_smooth[u,v] = w·(LapᵀLap)[u,v]·AᵤᵀAᵥ`
    (Aₜ = the marker Jacobian of frame t), a block-PENTA-diagonal contribution
    (band 2). Assembled sparse (scipy) and solved by `spsolve`.

    NOTE (v1): cross-frame anti-`skate` is DEFERRED — the world-accel smoothness
    already suppresses in-contact foot xy jitter; ground non-penetration is kept.

    Returns pose [T,J,3] LOCAL axis-angle (matching `solve_pose_gn`)."""
    import scipy.sparse as ssp
    from scipy.sparse.linalg import spsolve
    dt = _GN_DT
    n = N_JOINTS
    D = 3 * n
    parents = list(SMPL_PARENTS)
    obs = torch.as_tensor(obs_joints, dtype=dt)[:, :n]
    if prefilter:
        obs = prefilter_targets(obs, fps)[:, :n]
    T = obs.shape[0]
    root_target = obs[:, 0]
    s = float(scale)
    sQM = s * (torch.as_tensor(Q, dtype=dt) @ _RX_NEG90.to(dt))   # native→proper
    J = _rest_joints_native(smpl_model, betas, bone_scale, n)
    delta_t = (torch.zeros(n, 3, dtype=dt) if delta is None
               else torch.as_tensor(delta, dtype=dt)[:n])
    chain = _chain_mask(parents, n)
    Rl = exp_so3(torch.as_tensor(init_pose, dtype=dt)[:, :n].reshape(-1, 3)
                 ).reshape(T, n, 3, 3).clone()
    contact, floor = detect_contacts(obs, h_thresh=h_thresh,
                                     v_thresh=v_thresh_mps / max(fps, 1e-6))
    cm = contact.to(dt)
    wdata = torch.ones(n, dtype=dt)
    wdata[_FEET] = foot_weight
    eyeD = torch.eye(D, dtype=dt)
    # §P16 per-joint temporal smoothness. UNIFIED bows because its δ marker absorbs the
    # 5→3 spine convention gap (smpl_fit.py:398-405) while the BODY pose tracks the lean;
    # GN has no δ, so a UNIFORM w_smooth over-damps the spine and the head/neck rides UP
    # (no bow, neck-err 150–191 mm — §P16 visual verdict). Give the upper-body convention-
    # mismatch dofs (_DELTA_FREE: spine/neck/collars/head/shoulders) a LOWER smoothness so
    # the rhythmic spine swing is not crushed, while the slow lower body (root/legs/feet) +
    # arms KEEP w_smooth (the §P15.1 wobble-fix). w_smooth_upper is None ⇒ uniform = w_smooth
    # ⇒ bit-identical to the pre-§P16 solver (gate G16f / G15.1 tests).
    _sm_uni = w_smooth_upper is None
    wsm = torch.full((D,), float(w_smooth), dtype=dt)            # [D] per-dof smoothness wt
    if not _sm_uni:
        for jdx in _DELTA_FREE:
            wsm[3 * jdx:3 * jdx + 3] = float(w_smooth_upper)
    Wsm = torch.diag(wsm)                                        # [D,D]

    # constant 2nd-difference operator Lap [T-2,T] and its band K = LapᵀLap [T,T].
    if T >= 3:
        k = np.arange(T - 2)
        rows = np.repeat(k, 3)
        cols = np.concatenate([k, k + 1, k + 2])
        vals = np.concatenate([np.ones(T - 2), -2 * np.ones(T - 2), np.ones(T - 2)])
        Lap = ssp.csr_matrix((vals, (rows, cols)), shape=(T - 2, T))
        Kc = (Lap.T @ Lap).tocsr()
        Kdiag = torch.as_tensor(Kc.diagonal(0), dtype=dt)        # [T]
        Koff1 = torch.as_tensor(Kc.diagonal(1), dtype=dt)        # [T-1]
        Koff2 = torch.as_tensor(Kc.diagonal(2), dtype=dt)        # [T-2]

    def forward_and_jac(Rl):
        proper = torch.zeros(T, n, 3, dtype=dt)
        Jp = torch.zeros(T, n, 3, D, dtype=dt)
        for t in range(T):
            t_nat, Rw = _fk_mats(Rl[t], J, parents, n)
            off = torch.einsum('jmk,jk->jm', Rw, delta_t)
            p_nat = t_nat + off
            proper[t] = root_target[t] + torch.einsum('mn,jn->jm', sQM, p_nat - t_nat[0])
            diff = p_nat[:, None, :] - t_nat[None, :, :]
            block = -torch.einsum('jaxy,ayk->jaxk', hat_so3(diff), Rw)
            block = block * chain[:, :, None, None]
            Jp[t] = torch.einsum('mx,jaxk->jmak', sQM, block).reshape(n, 3, D)
        return proper, Jp

    def forward_only(Rl):
        proper = torch.zeros(T, n, 3, dtype=dt)
        for t in range(T):
            t_nat, Rw = _fk_mats(Rl[t], J, parents, n)
            p_nat = t_nat + torch.einsum('jmk,jk->jm', Rw, delta_t)
            proper[t] = root_target[t] + torch.einsum('mn,jn->jm', sQM, p_nat - t_nat[0])
        return proper

    def energy(Rl, proper):
        """The GN objective: robust GMOF data + global POSE-space 2nd-order smoothness
        + ground. Pose-space (not world-space) because the world-position smoothness is
        ill-conditioned (root-rotation lever); UNIFIED's effective smoothing is likewise
        on the pose params (§P15.1 finding)."""
        res = proper - obs
        sq = (res ** 2).sum(-1)
        e = float((wdata[None] * sigma_sq * sq / (sigma_sq + sq)).sum())   # GMOF ρ(s)
        if T >= 3:
            th = log_so3(Rl.reshape(-1, 3, 3)).reshape(T, D)
            s2 = th[2:] - 2 * th[1:-1] + th[:-2]
            if _sm_uni:
                e += w_smooth * float((s2 ** 2).sum())
            else:
                e += float(((s2 ** 2) * wsm[None]).sum())
        for fi, fj in enumerate(_FOOT_GROUND):
            pen = proper[:, fj, 2] - floor
            below = (cm[:, fi] > 0) & (pen < 0)
            if below.any():
                e += w_ground * float((pen[below] ** 2).sum())
        return e

    def coo(blocks, ru, cu):
        """[K,D,D] blocks at block-rows ru[K], block-cols cu[K] → (rows,cols,vals)."""
        ar = np.arange(D)
        r = (ru[:, None, None] * D + ar[None, :, None])
        c = (cu[:, None, None] * D + ar[None, None, :])
        K = blocks.shape[0]
        return (np.broadcast_to(r, (K, D, D)).ravel(),
                np.broadcast_to(c, (K, D, D)).ravel(),
                blocks.reshape(-1))

    loss_hist = []
    proper = None
    lam = 1e-2                                                   # LM damping (Marquardt, adaptive)
    H0_capture = g0_capture = None                              # iter-0 GN system (gate G15.1d)
    for _ in range(n_iters):
        proper, Jp = forward_and_jac(Rl)
        res = proper - obs                                       # [T,n,3]
        sq = (res ** 2).sum(-1)                                  # [T,n]
        rw = sigma_sq ** 2 / (sigma_sq + sq) ** 2                # GMOF IRLS (R1)
        wd = (wdata[None] * rw)[..., None, None]                 # [T,n,1,1]
        wdJp = wd * Jp
        Hd = torch.einsum('tjmd,tjme->tde', wdJp, Jp)           # [T,D,D] data (diag)
        gd = torch.einsum('tjmd,tjm->td', wdJp, res)            # [T,D]
        # fps-aware ground non-penetration (block-diagonal, in-contact feet).
        for fi, fj in enumerate(_FOOT_GROUND):
            for t in range(T):
                if float(cm[t, fi]) <= 0:
                    continue
                pen = float(proper[t, fj, 2] - floor)
                if pen < 0:
                    Jz = Jp[t, fj, 2, :]
                    Hd[t] = Hd[t] + w_ground * torch.outer(Jz, Jz)
                    gd[t] = gd[t] + w_ground * Jz * pen
        Hd = Hd + tikhonov * eyeD                                # base floor
        g = gd.clone()                                           # [T,D]
        if T >= 3:
            # GLOBAL POSE-space 2nd-order (constant-velocity) smoothness. ∂θ/∂dq ≈ I ⇒
            # the smoothness Jacobian is the CONSTANT 2nd-difference operator (identity
            # per dof) ⇒ H_smooth = w·(LapᵀLap)⊗I, perfectly conditioned (no lever
            # blow-up). This is the well-conditioned analogue of UNIFIED's pose smooth.
            th = log_so3(Rl.reshape(-1, 3, 3)).reshape(T, D)     # current axis-angle pose
            s2 = th[2:] - 2 * th[1:-1] + th[:-2]                 # [T-2,D] pose accel
            bth = torch.zeros(T, D, dtype=dt)                    # b = Lapᵀ s2
            bth[:T - 2] += s2
            bth[1:T - 1] += -2 * s2
            bth[2:] += s2
            g = g + bth * wsm[None, :]                           # per-dof g_smooth (A=I)
            B0 = Hd + Kdiag[:, None, None] * Wsm
            B1 = Koff1[:, None, None] * Wsm
            B2 = Koff2[:, None, None] * Wsm
        else:
            B0 = Hd
        # assemble the sparse global Hessian (penta-band) and solve H·dq = −g.
        rr, cc, vv = [], [], []
        r0, c0, v0 = coo(B0.numpy(), np.arange(T), np.arange(T))
        rr.append(r0); cc.append(c0); vv.append(v0)
        if T >= 3:
            # mirror block (v,u) = (u,v)ᵀ: swapping (rows,cols) already transposes,
            # so the VALUES stay as-is — transposing them too would double-transpose
            # and corrupt the non-symmetric AᵤᵀAᵥ coupling.
            u1 = np.arange(T - 1)
            r1, c1, v1 = coo(B1.numpy(), u1, u1 + 1)
            rr += [r1, c1]; cc += [c1, r1]; vv += [v1, v1]
            u2 = np.arange(T - 2)
            r2, c2, v2 = coo(B2.numpy(), u2, u2 + 2)
            rr += [r2, c2]; cc += [c2, r2]; vv += [v2, v2]
        H = ssp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                           shape=(T * D, T * D)).tocsr()
        if return_extra and H0_capture is None:                # the pre-LM-damping GN system —
            H0_capture = H.copy()                              # off-diag blocks isolate the
            g0_capture = g.reshape(-1).clone()                 # smoothness band (gate G15.1d).
        # adaptive Levenberg-Marquardt with MARQUARDT (diagonal) scaling: damp each dof
        # by λ·diag(H). The world-accel term gives high-lever root/hip dofs huge
        # curvature and tightly-pinned feet small curvature — a uniform step length
        # stalls; diagonal damping lets each dof move at its own scale (the §P15.1
        # slow-convergence cure). Accept if energy drops (λ↓), else damp harder (λ↑).
        diagH = torch.as_tensor(H.diagonal(), dtype=dt)
        gnp = -g.reshape(-1).numpy()
        E0 = energy(Rl, proper)
        accepted = False
        for _ in range(8):
            Hlm = (H + ssp.diags((lam * diagH).numpy())).tocsr()
            dq = torch.as_tensor(spsolve(Hlm, gnp), dtype=dt).reshape(T, n, 3)
            mr = float(dq.norm(dim=2).max())
            if mr > max_step_rot:
                dq = dq * (max_step_rot / mr)
            Rl_try = Rl @ exp_so3(dq.reshape(-1, 3)).reshape(T, n, 3, 3)
            if energy(Rl_try, forward_only(Rl_try)) < E0:
                Rl = Rl_try
                lam = max(lam * 0.5, 1e-8)
                accepted = True
                break
            lam = min(lam * 4.0, 1e8)
        loss_hist.append(E0)
        if not accepted:
            break                                               # converged / stuck

    pose = log_so3(Rl.reshape(-1, 3, 3)).reshape(T, n, 3)
    if return_extra:
        return pose, {'proper': proper, 'obs_filt': obs, 'floor': floor,
                      'contact': cm, 'loss_hist': loss_hist, 'fps': fps,
                      'H0': H0_capture, 'g0': g0_capture}
    return pose


def _unwrap_axis_angle(pose):
    """Make a [T,J,3] axis-angle sequence temporally CONTINUOUS.

    A rotation R is invariant to θ→θ±2π along its own axis, and the per-frame
    BVH→SMPL / matrix-log retarget can flip across that ±π branch — e.g. ch1's root
    jumping ~2π/frame when its orientation crosses the singularity, which teleports
    the whole body (§P18 diagnosis: ch1 MAX jitter 2477→46 mm/f² after unwrap). For
    each joint, pick the equivalent rep `(θ+2πk)·n̂` closest to the previous frame.
    A no-op where there is no wraparound."""
    out = torch.as_tensor(pose).clone()
    two_pi = 2.0 * torch.pi
    for t in range(1, out.shape[0]):
        v = out[t]; prev = out[t - 1]
        n = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        best = v.clone(); bestd = (v - prev).norm(dim=-1)
        for k in (-1, 1):
            cand = v + two_pi * k * n
            d = (cand - prev).norm(dim=-1)
            b = d < bestd
            best = torch.where(b[..., None], cand, best); bestd = torch.where(b, d, bestd)
        out[t] = best
    return out


def refine_pose_vposer(obs_joints, init_pose, smpl_model, Q, betas=None, scale=1.0,
                       bone_scale=None, root_target=None, lam_vp: float = 0.1,
                       lam_smooth: float = 8.0, n_iters: int = 120, lr: float = 0.02,
                       sigma_sq: float = 0.04):
    """§P18 — VPoser natural-pose refinement of a fitted pose (kills the unnatural
    hip rotation/sway the data-only GN solve leaves; §P16's "rotatery hip", REFUTED).

    Adam on the LOCAL axis-angle `pose`, warm-started from `init_pose` (the GN/unified
    solve), minimising
        GMOF(forward_proper(pose) − obs) + lam_vp·VPoser-prior(pose 1..21)
                                         + lam_smooth·‖2nd-order pose accel‖².
    The VPoser prior pulls the body (esp. hips) onto the natural-pose manifold; the
    data + smoothness terms hold the fit + the §P15.1 lower-body smoothness.

    Verified (scratch_p18_vposer.py shot_031): VPoser-implausibility 5.30→~0.16 at
    flat fit-err, hip rot-accel −30%. Returns pose [T,J,3] LOCAL axis-angle."""
    from skel2smpl.vposer import vposer_prior_loss
    dt = smpl_model.dtype
    obs = torch.as_tensor(obs_joints, dtype=dt)[:, :N_JOINTS]
    rt = None if root_target is None else torch.as_tensor(root_target, dtype=dt)
    # §P18: unwrap the init so a ±2π retarget flip (ch1's teleporting root) doesn't
    # seed the fit with a discontinuity the smoothness term then fights.
    init = _unwrap_axis_angle(torch.as_tensor(init_pose, dtype=dt)[:, :N_JOINTS])
    pose = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([pose], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        jp = forward_proper(smpl_model, pose, Q, betas=betas, scale=scale,
                            root_target=rt, bone_scale=bone_scale)
        data = gmof((jp - obs).pow(2).sum(-1), sigma_sq).mean()
        vp = vposer_prior_loss(pose[:, 1:22])                       # body joints 1..21
        if pose.shape[0] >= 3:
            accel = pose[2:] - 2 * pose[1:-1] + pose[:-2]
            smooth = (accel ** 2).mean()
        else:
            smooth = torch.zeros((), dtype=dt)
        (data + lam_vp * vp + lam_smooth * smooth).backward()
        opt.step()
    return pose.detach()
