"""§P24 — convert a 2C / ExPI / LindyHop / Ninjutsu sequence to boxing-format SMPL XML.

Offline tool. Fit both performers of one sequence and write the EXACT boxing schema
  <mocap shape_0=.. shape_1=..>
    <keyframe key="000000">
      <key personID="0" poses="tx ty tz | gx gy gz | body_pose(69)" triangulated=".."/>
so the four datasets load exactly like boxing's optimization/smpl.xml via the UNCHANGED
`smpl_xml_payload` (plain `SMPL.forward(aa, trans, β)`). After this skel2smpl is not needed
at runtime.

β estimation (user-directed): the EasyMocap `optimizeShape` (a faithful port,
`skel2smpl.fit.optimize_shape` ≡ pose_estimation/kinematics/optimization.py:785-836), adapted
per dataset via a marker-space kintree (`kintree_from_marker_bone`) so only OBSERVED direct
bones are length-matched. β is FROZEN and the pose is solved with it
(`fit_markers(fit_beta=False, bone_scale=None)`). ExPI uses the §P23.v surface-vertex path
instead (β identifiable from the skin markers). Pure plain SMPL — no bone_scale, no scale.

Frame contract (why a plain `SMPL.forward` reproduces the fit): `forward_R` returns
`M·(FK+trans)` (M=_TRANS=Rx−90); `forward_proper` post-rotates by Q=RXP=Rx+90, and Q·M=I. So:
  global_orient = pose[:,0];  body_pose = pose[:,1:22] (+2 zero hand joints);
  trans_box = trans − FK_root (FK_root = forward_proper(.., root_target=None)[:,0]);  β = β.
The boxing loader then gives forward(aa_box,trans_box,β) = M·(forward_proper body).
"""
import argparse
import os

import numpy as np
import torch
import xml.etree.ElementTree as ET

from skel2smpl.smpl import SMPL, _TRANS
from skel2smpl.fit import (fit_markers, forward_proper, optimize_shape,
                           kintree_from_marker_bone)
from skel2smpl.geometry import exp_so3, rotation_matrix_to_axis_angle
from skel2smpl.constants import _RX_NEG90, N_JOINTS
from skel2smpl import datasets as D

RXP = _RX_NEG90.t().double()                 # SMPL native (Y-up) → proper; Q·_TRANS = I
# The fit runs in the mocap Y-up frame (human up = +Y); the boxing loader applies a fixed
# M=Rx(−90) and the viewer is Y-up, so an unrotated export renders head-down. Boxing's own
# bodies are Y-up (head+Y) through the loader. Bake Rx(+90) so M·(FRAME_FIX·FK) is Y-up head+Y,
# matching boxing exactly (Rx90·(+Y)=+Z, then loader M·(+Z)=+Y). §P24.
FRAME_FIX = torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)   # Rx(+90)
LAM_S_PIN = 1e4                              # pin s≈1 (boxing has no scale field)
BETA_REG = 0.3                               # optimizeShape L2-β prior (β-only carries size)
LAM_BETA_VERT = 1e-3                         # ExPI vertex-attach: β identifiable


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # solvers run here


def _smpl():
    if not hasattr(_smpl, "_m"):
        _smpl._m = SMPL()
    return _smpl._m


def _smpl_gpu():
    """SMPL with buffers on DEV — used only for the heavy solvers (fit_markers /
    optimize_shape). Built under `torch.device(DEV)` so every buffer lands on the GPU
    (no SMPL edit needed). Falls back to a CPU model when CUDA is absent."""
    if not hasattr(_smpl_gpu, "_m"):
        with torch.device(DEV):
            _smpl_gpu._m = SMPL()
    return _smpl_gpu._m


# ───────────────────────── per-dataset performer markers ─────────────────────────
def load_performers(dataset, seq, nf):
    """Return [(markers[T,M,3] Y-up, marker_bone[M], marker_vert|None), …] for the two
    performers of `seq`, reusing the skel2smpl.datasets marker loaders."""
    if dataset == "2c":
        m0, mb = D._markers_2c(seq, nf)
        m1, _ = D._markers_2c(seq.replace("_C0.bvh", "_C1.bvh"), nf)
        return [(m0, mb, None), (m1, mb, None)]
    if dataset == "expi":
        ev = torch.tensor(D.EXPI_VERT, dtype=torch.long)
        m0, mb = D._markers_expi(seq, nf, follower=False)
        m1, _ = D._markers_expi(seq, nf, follower=True)
        return [(m0, mb, ev), (m1, mb, ev)]
    if dataset in ("lindyhop", "ninjutsu"):
        c0, c1 = D.remocap_csvs(seq)
        out = []
        for c in (c0, c1):
            obs, w = D._obs_remocap(c, nf)
            obsv = torch.where(w > 0.5)[0]
            out.append((obs[:, obsv], obsv.clone().long(), None))
        return out
    raise ValueError(f"unknown dataset {dataset!r}")


# ───────────────────────── plain-SMPL fit + boxing params ─────────────────────────
def fit_to_boxing(markers, marker_bone, marker_vert):
    """Plain-SMPL fit → (aa_box[T,24,3], trans_box[T,3], beta, err_mm). bone_scale=None, s≈1."""
    smpl = _smpl()                                           # CPU: forward_proper / err / I/O
    smpl_g = _smpl_gpu()                                     # GPU: the heavy solvers
    mb = torch.as_tensor(marker_bone, dtype=torch.long)
    mk = markers.to(DEV)                                     # markers on the solver device
    mbd = mb.to(DEV)
    # The two solvers run under `torch.device(DEV)` so every factory tensor lands on the GPU
    # (fit.py has no device pins; fit_markers self-moves obs/mb/Q to the model device). lam_vp=0:
    # §P24 plain-SMPL doesn't use the VPoser prior (already on-manifold) — also avoids its device path.
    with torch.device(DEV):
        if marker_vert is not None:                          # ExPI: β free via surface verts
            mvd = marker_vert.to(DEV)
            pose, trans, beta, s, _ = fit_markers(
                mk, mbd, smpl_g, RXP, bone_scale=None, marker_vert=mvd,
                lam_beta=LAM_BETA_VERT, lam_s=LAM_S_PIN, lam_vp=0.0)
        else:                                                # joint datasets: optimizeShape β, frozen
            # §P24: faithful EasyMocap optimizeShape, adapted per dataset via the marker-space
            # kintree (from marker_bone → only OBSERVED direct bones, no phantom segment).
            kintree = kintree_from_marker_bone(mb)
            beta0 = optimize_shape(mk, kintree, mbd, smpl_g, lam_reg=BETA_REG)
            pose, trans, beta, s, _ = fit_markers(
                mk, mbd, smpl_g, RXP, bone_scale=None, betas_init=beta0, fit_beta=False,
                lam_s=LAM_S_PIN, lam_vp=0.0)
    pose, trans, beta = pose.cpu(), trans.cpu(), beta.cpu()  # back to CPU for the I/O below
    T = pose.shape[0]
    jp = forward_proper(smpl, pose, RXP, betas=beta, scale=1.0, root_target=trans, want_verts=False)
    err = (jp[:, mb] - markers.to(jp)).norm(dim=-1).mean().item() * 1000.0 if marker_vert is None \
        else _vert_err(smpl, pose, beta, trans, marker_vert, markers)
    # bake the 180° X-flip into global_orient + trans so the boxing loader (M=Rx−90) is Z-up upright
    pose = pose.clone()
    ff = FRAME_FIX.to(pose.dtype)
    pose[:, 0] = rotation_matrix_to_axis_angle(ff @ exp_so3(pose[:, 0]))
    trans = (ff @ trans.to(pose.dtype).unsqueeze(-1)).squeeze(-1)
    j0 = forward_proper(smpl, pose, RXP, betas=beta)[:, 0]                       # FK_root of fixed pose
    trans_box = (trans.to(j0) - j0).float()
    aa_box = torch.cat([pose.float(), torch.zeros(T, 24 - N_JOINTS, 3)], dim=1)
    return aa_box, trans_box, beta.float(), err, float(s)


def _vert_err(smpl, pose, beta, trans, marker_vert, obs):
    _, vp = forward_proper(smpl, pose, RXP, betas=beta, scale=1.0, root_target=trans, want_verts=True)
    return (vp[:, marker_vert.to(torch.long)] - obs.to(vp)).norm(dim=-1).mean().item() * 1000.0


# ───────────────────────── boxing XML I/O ─────────────────────────
def _fmt(arr):
    return " ".join(f"{float(v):.6f}" for v in np.asarray(arr, np.float64).reshape(-1))


def write_boxing_xml(out_path, performers, gt_markers):
    """performers: [(aa_box[T,24,3], trans_box[T,3], beta), …]; gt_markers: [obs[T,M,3], …]."""
    T = performers[0][0].shape[0]
    attrs = " ".join(f'shape_{i}="{_fmt(p[2])}"' for i, p in enumerate(performers))
    lines = [f"<mocap {attrs}>"]
    for t in range(T):
        lines.append(f'<keyframe key="{t:06d}">')
        for pid, ((aa, tr, _), gt) in enumerate(zip(performers, gt_markers)):
            poses = np.concatenate([np.asarray(tr[t]).reshape(-1), np.asarray(aa[t]).reshape(-1)])
            lines.append(f'<key personID="{pid}" poses="{_fmt(poses)}" '
                         f'triangulated="{_fmt(gt[t])}" />')
        lines.append("</keyframe>")
    lines.append("</mocap>")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _load_boxing_body(xml_path):
    """Reconstruct verts via the boxing recipe using skel2smpl.smpl (byte-identical to
    vizsys.smpl, which `smpl_xml_payload` uses) — the G24a self-check."""
    smpl = _smpl()
    root = ET.parse(xml_path).getroot()
    per = {}
    for kf in root.findall("keyframe"):
        for k in kf.findall("key"):
            per.setdefault(int(k.get("personID")), []).append(
                [float(x) for x in k.get("poses").split()])
    out = []
    for pid in sorted(per):
        poses = np.asarray(per[pid], np.float32)
        aa = np.concatenate([poses[:, 3:6][:, None, :], poses[:, 6:75].reshape(-1, 23, 3)], 1)
        beta = torch.tensor([float(x) for x in root.get(f"shape_{pid}").split()], dtype=torch.float32)
        verts, _ = smpl.forward(torch.from_numpy(aa), torch.from_numpy(poses[:, 0:3]), beta)
        out.append(verts)
    return out


# ───────────────────────── orchestration ─────────────────────────
def convert(dataset, seq, out, nf, verify=True):
    smpl = _smpl()
    M = _TRANS.to(torch.float32)
    params, gts, refs = [], [], []
    for (markers, marker_bone, marker_vert) in load_performers(dataset, seq, nf):
        aa_box, trans_box, beta, err, s = fit_to_boxing(markers, marker_bone, marker_vert)
        params.append((aa_box, trans_box, beta))
        gts.append(np.einsum("mn,tjn->tjm", FRAME_FIX.numpy(), np.asarray(markers, np.float64)))
        print(f"    performer {len(params) - 1}: marker err {err:.1f} mm  ‖β‖={beta.norm():.2f}  s={s:.4f}")
        if verify:
            pose = aa_box[:, :N_JOINTS]
            root_t = trans_box + forward_proper(smpl, pose, RXP, betas=beta)[:, 0].float()
            _, vp = forward_proper(smpl, pose, RXP, betas=beta, scale=1.0, root_target=root_t,
                                   want_verts=True)
            refs.append(torch.einsum("mn,tvn->tvm", M, vp.float()))
    write_boxing_xml(out, params, gts)
    if verify:
        box = _load_boxing_body(out)
        d = max(float((b - r).abs().max()) for b, r in zip(box, refs))
        print(f"  G24a round-trip: max |boxing − forward_proper| = {d * 1000:.4f} mm "
              f"({'PASS' if d < 1e-3 else 'FAIL'})")
    print(f"  → {out}  ({params[0][0].shape[0]} frames, {len(params)} performers)")
    return out


# ───────────────────────── sequence enumeration (pilot/--all) ─────────────────────────
def enumerate_seqs(dataset, data_root):
    """Sorted [(seq_path, out_path), …] for a dataset, mirroring vr/viz's first-seq resolution."""
    out = []
    if dataset == "2c":
        for base, _d, files in os.walk(os.path.join(data_root, "Datasets", "2C")):
            for fn in sorted(files):
                if fn.endswith("_C0.bvh"):
                    seq = os.path.join(base, fn)
                    out.append((seq, seq[:-len("_C0.bvh")] + "_smpl.xml"))
    elif dataset == "expi":
        for base, _d, files in os.walk(os.path.join(data_root, "Datasets", "ExPI")):
            if "mocap_cleaned.tsv" in files:
                out.append((os.path.join(base, "mocap_cleaned.tsv"), os.path.join(base, "smpl.xml")))
    elif dataset in ("lindyhop", "ninjutsu"):
        root = os.path.join(data_root, "Ninjutsu" if dataset == "ninjutsu" else "LindyHop")
        for base, _d, files in os.walk(root):
            if "0_worldpos.csv" in files or os.path.exists(os.path.join(base, "0", "motion_worldpos.csv")):
                out.append((base, os.path.join(base, "smpl.xml")))
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description="convert a skeleton/marker dataset to boxing-format SMPL XML")
    p.add_argument("--dataset", required=True, choices=["2c", "expi", "lindyhop", "ninjutsu"])
    p.add_argument("--seq", default=None, help="explicit sequence path (overrides --data-root resolution)")
    p.add_argument("--out", default=None, help="output xml (default: next to the source)")
    p.add_argument("--data-root", default="/media/insq/ins/data")
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--all", action="store_true", help="convert every sequence of the dataset")
    a = p.parse_args()

    if a.seq:
        out = a.out or (a.seq[:-len("_C0.bvh")] + "_smpl.xml" if a.dataset == "2c"
                        else os.path.join(os.path.dirname(a.seq) if a.dataset == "expi" else a.seq, "smpl.xml"))
        convert(a.dataset, a.seq, out, a.max_frames)
        return
    seqs = enumerate_seqs(a.dataset, a.data_root)
    if not seqs:
        raise FileNotFoundError(f"no {a.dataset} sequences under {a.data_root}")
    if not a.all:
        seqs = seqs[:1]
    print(f"[{a.dataset}] converting {len(seqs)} sequence(s)")
    for i, (seq, out) in enumerate(seqs):
        print(f"[{a.dataset} {i + 1}/{len(seqs)}] {seq}")
        convert(a.dataset, seq, out, a.max_frames)


if __name__ == "__main__":
    main()
