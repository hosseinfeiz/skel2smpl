"""skel2smpl — a standalone, reusable skeleton→SMPL fitting module.

Fits a single shared SMPL shape (β + uniform size s) and per-frame body pose to
observed 3D joints (MoSh++/EasyMocap-style), with an optional VPoser natural-pose
refinement. Self-contained: vendored SO(3)/SE(3) geometry, SMPL skinning, and
VPoser — no `koopman`, `vizsys`, or `pose_estimation` on the path.

Import (sibling-dir form): put the parent of this directory on sys.path, then
`from skel2smpl import SMPL, solve_shape, solve_pose, refine_pose_vposer`.

Data files (gitignored, located via env or bundled under ``data/``):
  SMPL_MODEL_PATH → else data/smpl/SMPL_NEUTRAL.pkl
  VPOSER_PATH     → else data/vposer_v02
"""

from skel2smpl import geometry, constants, adapters, vposer, skeletons, datasets
from skel2smpl.datasets import (
    fit_2c, fit_expi, fit_remocap, fit_dataset, fit_smpl, fit_smpl_markers,
    Fit, DATASETS, C2_MAP, EXPI_MARKERS, EXPI_BONE, EXPI_BONE_SEG, EXPI_EDGES,
    REMO_NAME, SMPL_NAME, remocap_csvs,
)
from skel2smpl.smpl import SMPL
from skel2smpl.fit import (
    solve_shape,
    solve_pose,
    solve_pose_gn,
    solve_pose_gn_global,
    refine_pose_vposer,
    forward_proper as forward,
    _unwrap_axis_angle as unwrap_axis_angle,
    measure_bone_ratios,
    prefilter_targets,
    detect_contacts,
    fit_smpl_betas_limblen,
)

__all__ = [
    "SMPL",
    "solve_shape",
    "solve_pose",
    "solve_pose_gn",
    "solve_pose_gn_global",
    "refine_pose_vposer",
    "forward",
    "unwrap_axis_angle",
    "measure_bone_ratios",
    "prefilter_targets",
    "detect_contacts",
    "fit_smpl_betas_limblen",
    "geometry",
    "constants",
    "adapters",
    "vposer",
    "skeletons",
    "datasets",
    "fit_2c",
    "fit_expi",
    "fit_remocap",
    "fit_dataset",
    "fit_smpl",
    "fit_smpl_markers",
    "Fit",
    "DATASETS",
    "C2_MAP",
    "EXPI_MARKERS",
    "EXPI_BONE",
    "EXPI_BONE_SEG",
    "EXPI_EDGES",
    "REMO_NAME",
    "SMPL_NAME",
    "remocap_csvs",
]
