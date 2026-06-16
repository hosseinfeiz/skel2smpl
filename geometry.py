"""SO(3)/SE(3) Lie-group primitives for the skeleton→SMPL fit.

Vendored VERBATIM from koopman `models/se3_geometry.py` (the proven, π-safe
implementations). Only the subset the fitter + adapters need is kept:
SO(3) hat/vee/exp/log (+ axis-angle alias), the closest-SO(3) projection, and the
core SE(3) exp/log/compose/inverse/project. The koopman-specific machinery
(dual quaternions, BCH, noise, BodyModel, pipeline config) does NOT belong here.
"""

from dataclasses import dataclass

import torch


@dataclass
class Tol:
    """Numerical tolerance constants (tuned for float64 precision)."""
    small_angle: float = 1e-4
    near_pi_margin: float = 1e-2  # wider margin for near-π diagonal extraction
    small_angle_alt: float = 1e-8
    eps: float = 1e-12
    eps_fine: float = 1e-15
    eps_coarse: float = 1e-10
    eps_safe: float = 1e-14
    eps_ultra: float = 1e-14
    eps_min: float = 1e-16
    clamp_min: float = 1e-12
    clamp_norm: float = 1e-14
    clamp_denom: float = 1e-15


tol = Tol()


def hat_so3(omega: torch.Tensor) -> torch.Tensor:
    """Vectorized skew-symmetric hat map: R^3 -> so(3)."""
    batch_shape = omega.shape[:-1]
    zeros = torch.zeros(batch_shape, dtype=omega.dtype, device=omega.device)
    wx, wy, wz = omega.unbind(-1)
    return torch.stack([
        torch.stack([zeros, -wz, wy], dim=-1),
        torch.stack([wz, zeros, -wx], dim=-1),
        torch.stack([-wy, wx, zeros], dim=-1)
    ], dim=-2)


def vee_so3(Omega: torch.Tensor) -> torch.Tensor:
    """Vectorized vee map: so(3) -> R^3."""
    return torch.stack([Omega[..., 2, 1], Omega[..., 0, 2], Omega[..., 1, 0]], dim=-1)


def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    """Vectorized SO(3) exponential map with Taylor expansion for small angles."""
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq + tol.eps_min * tol.eps_min)
    theta_pow4 = theta_sq * theta_sq
    small = (theta < tol.small_angle).squeeze(-1)

    sin_t = torch.where(small[..., None], 1.0 - theta_sq / 6.0 + theta_pow4 / 120.0, torch.sin(theta) / theta.clamp(min=tol.eps_safe))
    cos_t = torch.where(small[..., None], 0.5 - theta_sq / 24.0 + theta_pow4 / 720.0, (1.0 - torch.cos(theta)) / theta_sq.clamp(min=tol.eps_safe))

    K = hat_so3(omega)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(*omega.shape[:-1], 3, 3)
    return I + sin_t[..., None] * K + cos_t[..., None] * (K @ K)


def log_so3(R: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm on SO(3) with π-safe branch handling using diagonal extraction."""
    input_is_batched = R.dim() > 2
    if not input_is_batched:
        R = R.unsqueeze(0)

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    small_angle = theta < tol.small_angle
    near_pi = theta > (torch.pi - tol.near_pi_margin)

    theta_safe = torch.where(small_angle, torch.ones_like(theta), theta)
    sin_theta = torch.sin(theta_safe)
    coeff = torch.where(
        small_angle, 0.5 + theta * theta / 12 + 7.0 * theta.pow(4) / 720,
        theta / (2 * sin_theta)
    )
    omega_hat = coeff[..., None, None] * (R - R.transpose(-1, -2))
    omega = vee_so3(omega_hat)

    if near_pi.any():
        batch_shape = R.shape[:-2]
        omega_near_pi = torch.zeros(batch_shape + (3,), dtype=R.dtype, device=R.device)

        near_pi_indices = near_pi.nonzero(as_tuple=False)

        for idx_tensor in near_pi_indices:
            idx_tuple = tuple(idx_tensor.tolist())
            R_single = R[idx_tuple]

            trace_single = R_single[0, 0] + R_single[1, 1] + R_single[2, 2]
            cos_theta_unclamped = (trace_single - 1) / 2
            theta_unclamped = torch.acos(torch.clamp(cos_theta_unclamped, -1.0, 1.0)).item()

            diag = torch.diag(R_single)
            diag_plus_one = diag + 1.0

            k = torch.argmax(diag_plus_one).item()

            axis = torch.zeros(3, dtype=R.dtype, device=R.device)
            axis[k] = torch.sqrt(torch.clamp(diag_plus_one[k] / 2.0, min=0.0))

            if k == 0:
                if axis[0].abs() > tol.eps:
                    axis[1] = (R_single[0, 1] + R_single[1, 0]) / (4.0 * axis[0])
                    axis[2] = (R_single[0, 2] + R_single[2, 0]) / (4.0 * axis[0])
            elif k == 1:
                if axis[1].abs() > tol.eps:
                    axis[0] = (R_single[0, 1] + R_single[1, 0]) / (4.0 * axis[1])
                    axis[2] = (R_single[1, 2] + R_single[2, 1]) / (4.0 * axis[1])
            else:
                if axis[2].abs() > tol.eps:
                    axis[0] = (R_single[0, 2] + R_single[2, 0]) / (4.0 * axis[2])
                    axis[1] = (R_single[1, 2] + R_single[2, 1]) / (4.0 * axis[2])

            axis = axis / axis.norm().clamp(min=tol.eps_min)

            omega_candidate = axis * theta_unclamped
            R_test = exp_so3(omega_candidate)
            error_pos = (R_test - R_single).norm()

            omega_candidate_neg = -axis * theta_unclamped
            R_test_neg = exp_so3(omega_candidate_neg)
            error_neg = (R_test_neg - R_single).norm()

            if error_neg < error_pos:
                omega_candidate = omega_candidate_neg

            omega_near_pi[idx_tuple] = omega_candidate

        omega = torch.where(near_pi[..., None], omega_near_pi, omega)

    if not input_is_batched:
        omega = omega.squeeze(0)

    return omega


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle via SO(3) logarithm."""
    return log_so3(R)


def clamp_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """Project a matrix to SO(3) using SVD to ensure orthogonality and det=1.

    Robust version that handles ill-conditioned matrices gracefully.
    """
    original_shape = R.shape
    original_dtype = R.dtype
    device = R.device

    # SVD doesn't support half precision - cast to float64 for maximum stability
    if R.dtype != torch.float64:
        R = R.double()

    # Flatten batch dimensions for processing
    if R.dim() > 3:
        batch_shape = R.shape[:-2]
        R = R.reshape(-1, 3, 3)
    else:
        batch_shape = None

    B = R.shape[0] if R.dim() == 3 else 1
    if R.dim() == 2:
        R = R.unsqueeze(0)

    # Pre-condition: clamp values to prevent extreme matrices
    R = torch.clamp(R, min=-1e6, max=1e6)

    # Replace NaN/Inf with identity
    nan_mask = ~torch.isfinite(R).all(dim=-1).all(dim=-1)
    if nan_mask.any():
        identity = torch.eye(3, device=device, dtype=R.dtype).unsqueeze(0)
        R = torch.where(nan_mask.unsqueeze(-1).unsqueeze(-1), identity.expand(B, -1, -1), R)

    R_corrected = torch.zeros_like(R)

    # Process in chunks to catch individual failures
    try:
        # Try batch SVD first (faster)
        # Closest SO(3) projection: fix U before multiplying by Vt
        # This gives the closest rotation in Frobenius norm (Umeyama 1991)
        U, S, Vt = torch.linalg.svd(R)
        det = torch.det(U @ Vt)
        sign = torch.where(det < 0, -1.0, 1.0).to(R.dtype)
        U_fix = U.clone()
        U_fix[..., :, -1] = U_fix[..., :, -1] * sign.unsqueeze(-1)
        R_corrected = U_fix @ Vt

    except RuntimeError:
        # Fallback: process element by element with Gram-Schmidt
        for i in range(B):
            try:
                U, S, Vt = torch.linalg.svd(R[i:i+1])
                det = torch.det(U @ Vt)
                sign = torch.where(det < 0, -1.0, 1.0).to(R.dtype)
                U_fix = U.clone()
                U_fix[..., :, -1] = U_fix[..., :, -1] * sign
                R_corrected[i] = (U_fix @ Vt)[0]
            except RuntimeError:
                # Last resort: use modified Gram-Schmidt orthogonalization
                R_i = R[i].clone()
                # Gram-Schmidt
                u0 = R_i[:, 0]
                u0 = u0 / u0.norm().clamp(min=1e-15)

                u1 = R_i[:, 1] - (R_i[:, 1] @ u0) * u0
                u1 = u1 / u1.norm().clamp(min=1e-15)

                u2 = torch.cross(u0, u1)  # Ensure orthogonality and det=+1

                R_gs = torch.stack([u0, u1, u2], dim=-1)

                # Check if valid, else use identity
                if torch.isfinite(R_gs).all():
                    R_corrected[i] = R_gs
                else:
                    R_corrected[i] = torch.eye(3, device=device, dtype=R.dtype)

    # Reshape back
    if batch_shape is not None:
        R_corrected = R_corrected.reshape(*batch_shape, 3, 3)
    elif R_corrected.shape[0] == 1 and len(original_shape) == 2:
        R_corrected = R_corrected.squeeze(0)

    # Cast back to original dtype
    if original_dtype != R_corrected.dtype:
        R_corrected = R_corrected.to(original_dtype)

    return R_corrected


def project_so3(R: torch.Tensor) -> torch.Tensor:
    """Project matrix to SO(3) via SVD with determinant correction."""
    return clamp_rotation_matrix(R)


def hat_se3(xi: torch.Tensor) -> torch.Tensor:
    """Vectorized hat map for se(3) twists [omega; v] -> 4x4 matrix."""
    omega, v = xi[..., :3], xi[..., 3:]
    Xi = torch.zeros(xi.shape[:-1] + (4, 4), dtype=xi.dtype, device=xi.device)
    Xi[..., :3, :3] = hat_so3(omega)
    Xi[..., :3, 3] = v
    return Xi


def vee_se3(Xi: torch.Tensor) -> torch.Tensor:
    """Vectorized vee map: se(3) 4x4 -> twist [omega; v]."""
    return torch.cat([vee_so3(Xi[..., :3, :3]), Xi[..., :3, 3]], dim=-1)


def project_se3(T: torch.Tensor) -> torch.Tensor:
    """Project 4x4 matrix to SE(3)."""
    T_proj = torch.zeros_like(T)
    T_proj[..., :3, :3] = project_so3(T[..., :3, :3])
    T_proj[..., :3, 3] = T[..., :3, 3]
    T_proj[..., 3, 3] = 1.0
    return T_proj


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Vectorized SE(3) exponential map: exponential coordinates → SE(3) matrix.

    T = exp(ξ) where ξ = [ω; v]; translation t = J(ω)·v couples rotation and
    translation via the left Jacobian.
    """
    w, v = xi[..., :3], xi[..., 3:]
    theta_sq = (w * w).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq + tol.eps_min * tol.eps_min)
    theta_cu = theta_sq * theta
    small = (theta < tol.small_angle).squeeze(-1)

    R = exp_so3(w)
    K = hat_so3(w)
    K_sq = K @ K
    I = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(*v.shape[:-1], 3, 3)

    A = torch.where(small[..., None], 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0, (1.0 - torch.cos(theta)) / theta_sq.clamp(min=tol.eps_safe))
    B = torch.where(small[..., None], 1.0 / 6.0 - theta_sq / 120.0 + theta_sq * theta_sq / 5040.0, (theta - torch.sin(theta)) / theta_cu.clamp(min=tol.eps_safe))

    J = I + A[..., None] * K + B[..., None] * K_sq
    t = torch.einsum('...ij,...j->...i', J, v)

    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Vectorized SE(3) matrix logarithm: SE(3) matrix → 6D twist [ω; v]."""
    R, t = T[..., :3, :3], T[..., :3, 3]
    w = log_so3(R)

    theta_sq = (w * w).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq + tol.eps_min * tol.eps_min)
    theta_pow4 = theta_sq * theta_sq
    half_theta = theta / 2.0
    small = (theta < tol.small_angle).squeeze(-1)
    near_pi = (theta > (torch.pi - tol.near_pi_margin)).squeeze(-1)

    K = hat_so3(w)
    K_sq = K @ K
    I = torch.eye(3, dtype=T.dtype, device=T.device).expand(*t.shape[:-1], 3, 3)

    half_theta_safe = torch.clamp(half_theta, min=tol.eps_safe,
                                   max=(torch.pi - tol.near_pi_margin) / 2.0 - 1e-6)
    tan_half = torch.tan(half_theta_safe)

    general_B_inv = 1.0 / (theta_sq + tol.eps_safe) - 1.0 / (tan_half * 2.0 * theta + tol.eps_safe)
    small_B_inv = 1.0/12.0 + theta_sq / 720.0 + theta_pow4 / 30240.0
    near_pi_B_inv = 1.0 / (theta_sq + tol.eps_safe)

    B_inv = torch.where(
        small[..., None],
        small_B_inv,
        torch.where(near_pi[..., None], near_pi_B_inv, general_B_inv)
    )

    J_inv = I - 0.5 * K + B_inv[..., None] * K_sq
    v = torch.einsum('...ij,...j->...i', J_inv, t)
    return torch.cat([w, v], dim=-1)


def se3_compose(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """Vectorized SE(3) composition T1 @ T2."""
    target_dtype = T1.dtype
    T2 = T2.to(dtype=target_dtype)

    R1, t1 = T1[..., :3, :3], T1[..., :3, 3]
    R2, t2 = T2[..., :3, :3], T2[..., :3, 3]

    out_shape = torch.broadcast_shapes(T1.shape[:-2], T2.shape[:-2]) + (4, 4)
    T = torch.zeros(out_shape, dtype=target_dtype, device=T1.device)
    T[..., :3, :3] = R1 @ R2
    T[..., :3, 3] = torch.einsum('...ij,...j->...i', R1, t2) + t1
    T[..., 3, 3] = 1.0
    return T


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Vectorized SE(3) inverse."""
    R, t = T[..., :3, :3], T[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.einsum('...ij,...j->...i', R_inv, t)

    Tout = torch.zeros_like(T)
    Tout[..., :3, :3] = R_inv
    Tout[..., :3, 3] = t_inv
    Tout[..., 3, 3] = 1.0
    return Tout
