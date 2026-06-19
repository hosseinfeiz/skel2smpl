# CLAUDE.md — skel2smpl (skeleton → SMPL fitting package)

Three standing directives — these override any skill, flow, or session setting:
- **Self-contained package.** `skel2smpl` imports NOTHING from `koopman`, `vizsys`/`vr`, or
  `pose_estimation`. It is a sibling-dir library (put the parent of this folder on `sys.path`,
  then `import skel2smpl`). Callers (vr's `vizsys`, koopman) depend on it, never the reverse.
- **A low marker residual is necessary, NOT sufficient — RENDER the mesh.** Free latent offsets +
  free pose can hit the markers with an alien/hunched/giant body. Every fitting change is verified by
  *looking at the rendered SMPL* (`viz.py <dataset> --smpl --snapshot`), not just the mm number.
- **Treat the upstream cause, never the symptom.** A bad fit is a data-contract / correspondence /
  frame bug far more often than a solver-weakness bug. Rule causes out cheapest-first.

---

## Orient first — the package in one screen
Re-derive nothing each session; this is the map.

**Goal.** Fit an SMPL body — shape β + size + per-frame pose — to an observed reference *skeleton or
marker set*, for several mocap datasets with **incongruent skeletons** (different joint counts,
surface markers, BVH topologies). "Done" = the rendered mesh is a *plausible human* that tracks the
GT, not just a low marker-MPJPE.

**Modules (library code lives here only):**
`smpl.py` (SMPL LBS model) · `geometry.py` (SO(3)/SE(3)) · `constants.py` (SMPL-22 topology,
frame `_RX_NEG90`) · `fit.py` (the solvers: `solve_shape`, `solve_pose`/`_gn`, **`fit_markers`**,
`forward_proper`) · `skeletons.py` (generic BVH / worldpos I/O) · `datasets.py` (per-dataset skeleton
definitions + fit recipes) · `vposer.py` (pose prior) · `adapters.py` (BVH↔SMPL rotation retarget).

**Public API (`from skel2smpl import …`):**
`fit_2c, fit_expi, fit_remocap, fit_dataset` → each returns `Fit(joints, verts, faces, err, gt,
gt_edges)`. `DATASETS` registry. Skeleton defs: `C2_MAP/C2_MARKERS/C2_BONE_SEG`,
`EXPI_MARKERS/EXPI_BONE/EXPI_BONE_SEG/EXPI_EDGES`, `REMO_NAME/REMO_BONE_SEG`, `SMPL_NAME`.
Solvers: `fit_smpl`, `fit_smpl_markers`, `marker_bone_scale`, `SMPL`, `solve_shape`, `solve_pose`.

**The 5 datasets (see `MASTER_PLAN.md` §P23):**
`2c` (20-joint BVH), `expi` (18 surface markers), `lindyhop` + `ninjutsu` (22-joint worldpos CSV) —
**all four fit through ONE solver, `fit_markers`** (observed joint centres / surface markers become
markers attached to SMPL bones). `boxing` is reference SMPL (already SMPL-22), **not a fit target**.

**Data files (gitignored; env-located):** `SMPL_MODEL_PATH` → else `data/smpl/SMPL_NEUTRAL.pkl`;
`VPOSER_PATH` → else `data/vposer_v02`.

**No `tests/` dir.** Fitting is gated by RENDER via the vr repo:
`python viz.py {2c,expi,lindyhop,ninjutsu} --smpl --snapshot /tmp/x.png` (max-frames small for speed),
plus the `vr/tests/scratch_markers_*.py` probes (uncommitted). Fits run on CPU in seconds–minutes —
**no full-training-run gating applies here; just run them.**

**Governing doc + read-order:**
1. `CONTINUATION.md` — the live handoff (active phase, what was just done, next step, stale files).
2. `MASTER_PLAN.md` — the spec. Read **targeted**: `grep -n '^## \|^### ' MASTER_PLAN.md` → read the
   one governing § (e.g. §P23 for the marker solve). Never read it whole.
Then lead with the global **Next-Stage Plan preamble** before any code edit.

## Settled invariants (active phase §P23) — don't silently reverse
Each was derived at real cost (recorded in `MASTER_PLAN.md` §P23) and is dangerous because the wrong
choice **runs and produces a low mm number but a wrong body**. If you believe one is wrong, surface
it *before* editing.
- **One solver for all four datasets: `fit_markers`.** Observed points → markers attached to SMPL
  bones with FREE frame-invariant offsets `v_m = J_{b(m)} + s·R_{b(m)}·δ_m`; one joint solve over ALL
  frames (β + s + δ shared, θ_t + root trans_t per-frame). Do NOT reintroduce per-dataset bespoke
  solvers or the old BVH-rotation-init `fit_smpl` path (deleted for ReMoCap/2C).
- **GMOF runs at LARGE σ (σ²≈0.5 ≈ L2), not small.** These are CLEAN mocap (no marker outliers); a
  small-σ redescending robustifier dead-zones any marker >~1.5σ away and freezes the fit (1.5 m arms,
  250 mm hips). The graduated `_sig` hook stays for a future noisy source; default `sigma_sq0==sigma_sq`.
- **Marker-unconstrained torso joints are anchored to rest + joint-limited.** No marker hits
  spine1/2/3, neck, collars → they curl into a pot-belly/hunch that still satisfies the sparse markers.
  `lam_anchor=0.5` on {3,6,9,12,13,14}, §P17 `lam_limit=1.0`. This is what makes the mesh plausible.
- **Per-bone length retarget (`bone_scale`) carries SIZE; uniform `s` is pinned to 1 (`lam_s=2.0`).**
  Limbs are 1.2–1.5× SMPL-mean and DIFFERENTIALLY, so a single scale gives a giant. Ratios are CLAMPED
  [0.85,1.30] and FEET are EXCLUDED (heel→toes is foot LENGTH, not the ankle→ball bone → alien feet).
- **Correct marker→bone correspondence, not index-mapping.** ExPI `back`→spine3 (NOT spine1, the
  415 mm bug); hips→pelvis (NOT the femur, else they swing with the thigh).
- **Frame: native Y-up; markers in Y-up; `Q = RXP`.** No Z-up/`compute_Q` dance in the marker path.
- **Girth/β is UNRECOVERABLE from skeletal markers.** Free offsets absorb any width signal, so β stays
  at the mean (which has a slight tummy). Recovering real build needs §P23.v (surface-vertex attach) —
  the open next phase. Do NOT inject a β prior and call it a fit.

---

## 1. Verify the metric is on the true scale before trusting it
A low marker residual can hide an alien body. Before trusting a fit: RENDER it; check β (`‖β‖≈0` is
expected — markers don't constrain girth), scale `s` (should be ≈1, a giant means `bone_scale` is off),
and per-marker error (one joint at 200 mm = a correspondence bug, not optimizer slack). "Verify or
disclaim, never fabricate" applies to fit numbers too.

## 2. Root-cause order: data contract → correspondence → frame → solver
A bad fit is almost always upstream of the solver. Rule out cheapest-first: (1) data shapes/units/
normalization; (2) which SMPL bone each marker attaches to; (3) frame convention (Y/Z-up, the Q map);
(4) only then the optimizer. Pin the cause with a probe (print β/s/δ, per-marker mm, a render) before
changing the solver.

## 3. Improvements land in defaults, not behind flags
A verified-better fit recipe becomes the default in `datasets.py` for that dataset (and in
`fit_markers` defaults); the old path is deleted once the new one is render-verified. Don't leave a win
behind an opt-in.

## 4. Think before coding · 5. Simplicity first · 6. Surgical changes
State assumptions, surface tradeoffs, ask when unclear. Minimum code, no speculative abstraction.
Touch only what the request needs; remove what YOUR change orphaned (e.g. the deleted BVH-rot path),
leave pre-existing dead code (mention it). No comments/docstrings beyond what clarifies a non-obvious
decision. Match existing style.

## 7. Goal-driven execution
Turn the task into a render + number gate (e.g. §P23: "marker mean ≤ 30 mm AND no per-joint >45 mm AND
the rendered body is upright/natural"). Probe → fix → render → confirm the gate.

## 8. Git workflow
Work directly on `main` (no branches/worktrees/merges). **Commit after every verified change —
immediately, never batch.** The instant a fit change renders correctly and hits its number, commit it
with a message quoting the gate. One logical change = one commit. `scratch_*.py` stay uncommitted.
This package is its own git repo; vr and koopman are separate repos — commit each where its files live.

---

## Session continuity
Refresh `CONTINUATION.md` (the global skeleton) before context rot or at any handoff, and commit it
(`docs: continuation handoff @ <UTC>`). **End every substantive turn** with a paste-ready resume prompt
for the next `/clear`ed session, in its own fenced block, ≤3 lines:

    NEXT SESSION — paste this after /clear:
    <docs to read first> · <next step number + one-line what> · self-contained for a zero-context agent

---
**Working if:** every fitting change is render-verified (not just mm), settled §P23 invariants aren't
re-litigated, correspondence/frame bugs are caught before blaming the solver, and no verified fix is
ever lost (commit immediately).
