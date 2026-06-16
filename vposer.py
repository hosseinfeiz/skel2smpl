"""Vendored, self-contained VPoser v2 natural-pose prior.

No dependency on the `pose_estimation` / `human_body_prior` packages. Contains
(1) the VPoser v2 VAE architecture (num_neurons=512, latentD=32), vendored from
`pose_estimation/kinematics/vposer.py`; (2) the matrot→axis-angle conversion the
decoder needs (`matrot2aa` and its quaternion helpers), vendored VERBATIM from
`pose_estimation/utils.py` so the decode convention is identical by construction
(the §P18 acceptance check: rest-pose prior ≈ 0.009, random pose ≈ 2.0); and
(3) a minimal weight loader replacing `model_load.load_model`/`exprdir2model` —
no OmegaConf, no glob-of-globs.

Weights are located via the `VPOSER_PATH` env var, else the bundled
`skel2smpl/data/vposer_v02`. The directory must hold `*.yaml` (model_params) and
`snapshots/*.ckpt` (the trained snapshot, keys prefixed `vp_model.`).
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = os.environ.get('VPOSER_PATH', str(_MODULE_DIR / 'data' / 'vposer_v02'))

_VP = None
_VP_DEVICE = None


# ── rotation conversions (vendored from pose_estimation/utils.py) ────────────────
def quaternion_to_angle_axis(quaternion):
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )
    k_pos = two_theta / (sin_theta + 1e-8)
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] = q1 * k
    angle_axis[..., 1] = q2 * k
    angle_axis[..., 2] = q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    rmat_t = rotation_matrix.transpose(1, 2)
    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([
        rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
        t0,
        rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
        rmat_t[:, 2, 0] + rmat_t[:, 0, 2]
    ], dim=-1)
    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
        rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
        t1,
        rmat_t[:, 1, 2] + rmat_t[:, 2, 1]
    ], dim=-1)
    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([
        rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        t2
    ], dim=-1)
    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([
        t3,
        rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
        rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
        rmat_t[:, 0, 1] - rmat_t[:, 1, 0]
    ], dim=-1)
    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~mask_d0_d1)
    mask_c2 = (~mask_d2) * mask_d0_nd1
    mask_c3 = (~mask_d2) * (~mask_d0_nd1)
    mask_c0 = mask_c0.unsqueeze(-1).type_as(q0)
    mask_c1 = mask_c1.unsqueeze(-1).type_as(q1)
    mask_c2 = mask_c2.unsqueeze(-1).type_as(q2)
    mask_c3 = mask_c3.unsqueeze(-1).type_as(q3)
    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    t0_rep = t0.repeat(4, 1).t()
    t1_rep = t1.repeat(4, 1).t()
    t2_rep = t2.repeat(4, 1).t()
    t3_rep = t3.repeat(4, 1).t()
    denom = (t0_rep * mask_c0 + t1_rep * mask_c1 +
             t2_rep * mask_c2 + t3_rep * mask_c3).sqrt()
    q = q / (denom + 1e-8)
    q = 0.5 * q
    return q


def rotation_matrix_to_angle_axis(rotation_matrix):
    q = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(q)


def matrot2aa(pose_matrot):
    b = pose_matrot.size(0)
    pad_col = torch.zeros(b, 3, 1, device=pose_matrot.device, dtype=pose_matrot.dtype)
    homogen_matrot = torch.cat([pose_matrot, pad_col], dim=2)
    return rotation_matrix_to_angle_axis(homogen_matrot)


# ── VPoser v2 architecture (vendored from pose_estimation/kinematics/vposer.py) ──
class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()
        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


class VPoser(nn.Module):
    def __init__(self, num_neurons: int = 512, latentD: int = 32):
        super(VPoser, self).__init__()
        self.latentD = latentD
        self.num_joints = 21
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def encode(self, pose_body):
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        bs = Zin.shape[0]
        prec = self.decoder_net(Zin)
        return {
            'pose_body': matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }

    def forward(self, pose_body):
        q_z = self.encode(pose_body)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
        return decode_results


def _model_params_from_yaml(expr_dir: str):
    """(num_neurons, latentD) from the experiment's *.yaml; defaults to the V02
    architecture (512, 32) if no yaml / keys are present."""
    num_neurons, latentD = 512, 32
    try:
        import yaml
        ymls = sorted(Path(expr_dir).glob('*.yaml'))
        if ymls:
            with open(ymls[0]) as f:
                cfg = yaml.safe_load(f) or {}
            mp = cfg.get('model_params', {}) or {}
            num_neurons = int(mp.get('num_neurons', num_neurons))
            latentD = int(mp.get('latentD', latentD))
    except Exception:
        pass
    return num_neurons, latentD


def load_vposer(expr_dir: str = None, device: str = 'cpu'):
    """Load (and cache) the frozen vendored VPoser. `expr_dir` defaults to
    `$VPOSER_PATH` else the bundled `skel2smpl/data/vposer_v02`. Raises a clear
    error if the weights are not reachable."""
    global _VP, _VP_DEVICE
    expr_dir = expr_dir or _DEFAULT_WEIGHTS
    if _VP is not None and _VP_DEVICE == device:
        return _VP
    snap_dir = os.path.join(expr_dir, 'snapshots')
    ckpts = sorted(__import__('glob').glob(os.path.join(snap_dir, '*.ckpt')),
                   key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(
            f"skel2smpl VPoser prior: no snapshot *.ckpt under {snap_dir}. Set "
            f"VPOSER_PATH to a vposer_v02 dir (with snapshots/*.ckpt and *.yaml).")
    num_neurons, latentD = _model_params_from_yaml(expr_dir)
    vp = VPoser(num_neurons=num_neurons, latentD=latentD)
    # Trusted local checkpoint (Lightning .ckpt; top level may carry non-tensor
    # hyper-params, so weights_only=False is required to reach ['state_dict']).
    sd = torch.load(ckpts[-1], map_location='cpu', weights_only=False)['state_dict']
    sd = {k.replace('vp_model.', '') if k.startswith('vp_model.') else k: v
          for k, v in sd.items()}
    keep = set(vp.state_dict().keys())
    sd = {k: v for k, v in sd.items() if k in keep}
    vp.load_state_dict(sd, strict=False)
    vp = vp.to(device).eval()
    for p in vp.parameters():
        p.requires_grad_(False)
    _VP, _VP_DEVICE = vp, device
    return _VP


def vposer_prior_loss(pose_body: torch.Tensor) -> torch.Tensor:
    """Reconstruction-form VPoser prior.

    pose_body: [T, 21, 3] LOCAL axis-angle of the SMPL body joints 1..21.
    Returns ‖pb − VPoser.decode(VPoser.encode(pb).mean)‖² — off-manifold
    (unnatural) poses reconstruct poorly and are penalised. Returned in
    pose_body's dtype.
    """
    vp = load_vposer(device=pose_body.device.type)
    pb = pose_body.float()
    rec = vp.decode(vp.encode(pb).mean)['pose_body']
    return (pb - rec).pow(2).mean().to(pose_body.dtype)
