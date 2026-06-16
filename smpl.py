import os
import pickle
import numpy as np
import torch

_TRANS = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# Model file via env SMPL_MODEL_PATH else the bundled neutral body.
_DEFAULT_PKL = os.environ.get("SMPL_MODEL_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "smpl", "SMPL_NEUTRAL.pkl"))


def _dense(x):
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def axis_angle_to_matrix(aa):
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = torch.zeros_like(kx)
    K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return I + s * K + (1.0 - c) * (K @ K)


def _quat_mul(a, b):
    """Hamilton product of quaternions (w,x,y,z), broadcasting on the batch dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _quat_conj(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _mat_to_quat(R):
    """[...,3,3] proper rotation → unit quaternion [...,4] (w,x,y,z), numerically
    stable 4-case method (divide by the largest component)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22
    c0 = t > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = (~c0) & (~c1) & (~c2)
    s0 = torch.sqrt((t + 1.0).clamp_min(1e-12)) * 2
    q0 = torch.stack([0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0], -1)
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) * 2
    q1 = torch.stack([(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], -1)
    s2 = torch.sqrt((1 - m00 + m11 - m22).clamp_min(1e-12)) * 2
    q2 = torch.stack([(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], -1)
    s3 = torch.sqrt((1 - m00 - m11 + m22).clamp_min(1e-12)) * 2
    q3 = torch.stack([(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], -1)
    q = torch.where(c0[..., None], q0, q1)
    q = torch.where(c2[..., None], q2, q)
    q = torch.where(c3[..., None], q3, q)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def retarget_joints(J, bone_scale, parents):
    """§P14 hierarchical per-bone rest-joint retarget (root→leaf over `parents`).

    J [B,3] rest joints, bone_scale [B] per-bone length ratio r (root and any
    non-retargeted bone = 1), parents [B] (parents[0] = -1). Each child joint carries
    its parent's cumulative retarget plus its own local scale:
        J'[c] = J'[p] + r[c]·(J[c] − J[p]).
    r ≡ 1 → J' = J exactly. Returns J' [B,3]."""
    r = torch.as_tensor(bone_scale, dtype=J.dtype)
    Jr = J.clone()
    for c in range(1, J.shape[0]):
        p = int(parents[c])
        Jr[c] = Jr[p] + r[c] * (J[c] - J[p])
    return Jr


def retarget_bones(J, v, W, bone_scale, parents):
    """§P14 per-bone bind-pose bone-length retarget (TRANSLATION form).

    Retargets the rest JOINTS to the target per-bone lengths (`retarget_joints`) and
    rides the rest MESH along by the LBS-weighted joint displacement:
        v'[i] = v[i] + Σ_b W[i,b]·(J'[b] − J[b]).
    The standard skinned-mesh bone-length edit: it lengthens each bone while PRESERVING
    its cross-section girth (a pure length change — consistent with the §P9 girth
    invariant; the rejected isotropic form `Σ_b W[i,b]·(J'[b]+r[b](v[i]−J[b]))` would
    inflate girth by r and distort interior edges — see §P14 BUILD DECISION). W is a
    partition of unity → no seam. r ≡ 1 → (J,v) unchanged exactly.

    J [B,3] rest joints, v [V,3] rest vertices, W [V,B] LBS weights, bone_scale [B],
    parents [B]. Returns (J' [B,3], v' [V,3])."""
    Jr = retarget_joints(J, bone_scale, parents)
    v_ret = v + torch.einsum("vb,bx->vx", W, Jr - J)
    return Jr, v_ret


class SMPL:
    def __init__(self, pkl_path=None, dtype=torch.float32):
        with open(pkl_path or _DEFAULT_PKL, "rb") as f:
            m = pickle.load(f, encoding="latin1")
        self.dtype = dtype
        self.faces = _dense(m["f"]).astype(np.int32)
        self.v_template = torch.tensor(_dense(m["v_template"]), dtype=dtype)
        self.shapedirs = torch.tensor(_dense(m["shapedirs"]), dtype=dtype)
        self.posedirs = torch.tensor(_dense(m["posedirs"]), dtype=dtype)
        self.J_regressor = torch.tensor(_dense(m["J_regressor"]), dtype=dtype)
        self.weights = torch.tensor(_dense(m["weights"]), dtype=dtype)
        self.parents = _dense(m["kintree_table"])[0].astype(np.int64)
        self.parents[0] = -1
        self.n_betas = self.shapedirs.shape[-1]

    def forward(self, aa, trans=None, betas=None, skinning="lbs", bone_scale=None):
        aa = torch.as_tensor(aa, dtype=self.dtype)
        if aa.shape[1] < 24:
            pad = torch.zeros(aa.shape[0], 24 - aa.shape[1], 3, dtype=self.dtype)
            aa = torch.cat([aa, pad], dim=1)
        return self.forward_R(axis_angle_to_matrix(aa), trans, betas, skinning=skinning,
                              bone_scale=bone_scale)

    def forward_R(self, R, trans=None, betas=None, skinning="lbs", bone_scale=None):
        R = torch.as_tensor(R, dtype=self.dtype)
        T = R.shape[0]
        if R.shape[1] < 24:
            I = torch.eye(3, dtype=self.dtype).expand(T, 24 - R.shape[1], 3, 3)
            R = torch.cat([R, I], dim=1)
        if betas is None:
            betas = torch.zeros(self.n_betas, dtype=self.dtype)
        v_shaped = self.v_template + torch.einsum("vxb,b->vx", self.shapedirs, betas.to(self.dtype))
        J = self.J_regressor @ v_shaped
        if bone_scale is not None:                          # §P14 per-bone bind-pose retarget
            J, v_shaped = retarget_bones(
                J, v_shaped, self.weights,
                torch.as_tensor(bone_scale, dtype=self.dtype), self.parents)
        I3 = torch.eye(3, dtype=self.dtype)
        pose_feat = (R[:, 1:] - I3).reshape(T, -1)
        v_posed = v_shaped + torch.einsum("vxp,tp->tvx", self.posedirs, pose_feat)
        G = self._global_transforms(R, J)
        joints = G[:, :, :3, 3]
        J_pad = torch.cat([J, torch.zeros(24, 1, dtype=self.dtype)], dim=1)
        offset = torch.einsum("tjmn,jn->tjm", G[:, :, :3, :], J_pad)
        G_rel = G.clone()
        G_rel[:, :, :3, 3] = G[:, :, :3, 3] - offset
        if skinning == "dqs":
            # Dual-quaternion skinning (Kavan et al. 2008) — blends the per-bone rigid
            # transforms as unit dual quaternions, avoiding the LBS "candy-wrapper"
            # volume collapse at large joint rotations/twist (elbows/wrists/shoulders).
            verts = self._dqs(G_rel, v_posed.expand(T, -1, -1))
        else:
            Tlbs = torch.einsum("vj,tjmn->tvmn", self.weights, G_rel)
            vh = torch.cat([v_posed.expand(T, -1, -1),
                            torch.ones(T, v_posed.shape[1], 1, dtype=self.dtype)], dim=-1)
            verts = torch.einsum("tvmn,tvn->tvm", Tlbs, vh)[..., :3]
        if trans is not None:
            tr = torch.as_tensor(trans, dtype=self.dtype)
            verts = verts + tr[:, None, :]
            joints = joints + tr[:, None, :]
        M = _TRANS.to(self.dtype)
        verts = torch.einsum("mn,tvn->tvm", M, verts)
        joints = torch.einsum("mn,tjn->tjm", M, joints)
        return verts, joints

    def _dqs(self, G_rel, v_posed):
        """Dual-quaternion skinning. G_rel [T,24,4,4] per-bone rigid transforms,
        v_posed [T,V,3]. Returns verts [T,V,3]. Antipodality is resolved per vertex
        against its max-weight bone (Kavan's pivot); looped over frames to bound memory."""
        Tn, V = G_rel.shape[0], v_posed.shape[1]
        W = self.weights                                     # [V,24]
        pivot = W.argmax(dim=1)                              # [V]
        out = torch.empty(Tn, V, 3, dtype=self.dtype)
        zero = torch.zeros(24, 1, dtype=self.dtype)
        for t in range(Tn):
            qr = _mat_to_quat(G_rel[t, :, :3, :3])           # [24,4] bone rotations
            qt = torch.cat([zero, G_rel[t, :, :3, 3]], -1)   # (0, tx,ty,tz) per bone
            qd = 0.5 * _quat_mul(qt, qr)                     # [24,4] dual part
            sgn = torch.sign((qr.unsqueeze(0) * qr[pivot].unsqueeze(1)).sum(-1))  # [V,24]
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            ws = W * sgn                                     # [V,24] signed weights
            br = ws @ qr                                     # [V,4] blended real
            bd = ws @ qd                                     # [V,4] blended dual
            nr = br.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            br, bd = br / nr, bd / nr
            v = v_posed[t]                                   # [V,3]
            xyz = br[:, 1:]
            tt = 2.0 * torch.cross(xyz, v, dim=-1)
            vrot = v + br[:, :1] * tt + torch.cross(xyz, tt, dim=-1)   # rotate by br
            trv = _quat_mul(2.0 * bd, _quat_conj(br))[:, 1:]           # translation
            out[t] = vrot + trv
        return out

    def _global_transforms(self, R, J):
        T = R.shape[0]
        rel = J.clone()
        rel[1:] = J[1:] - J[self.parents[1:]]
        mats = torch.zeros(T, 24, 4, 4, dtype=self.dtype)
        mats[:, :, :3, :3] = R
        mats[:, :, :3, 3] = rel
        mats[:, :, 3, 3] = 1.0
        G = [mats[:, 0]]
        for j in range(1, 24):
            G.append(G[self.parents[j]] @ mats[:, j])
        return torch.stack(G, dim=1)


def smpl_vertex_normals(verts, faces):
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    nrm = np.zeros_like(v)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(nrm, f[:, k], fn)
    n = np.linalg.norm(nrm, axis=1, keepdims=True)
    return (nrm / np.clip(n, 1e-8, None)).astype(np.float32)


def body_mesh(verts, faces, color):
    v = np.ascontiguousarray(verts, np.float32)
    return {"verts": v, "faces": np.asarray(faces, np.int32),
            "normals": smpl_vertex_normals(v, faces), "color": tuple(color)}
