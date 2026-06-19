# SESSION_CONTINUATION — 2026-06-19 14:24 UTC — all 4 fittable datasets on ONE marker solver; girth (§P23.v) is the open next phase

> **The headline.** `skel2smpl` is now a clean standalone package: the per-dataset skeleton
> definitions + fit recipes all live here (`datasets.py`, public API in `__init__.py`), the generic
> BVH/worldpos I/O is here (`skeletons.py`), and **all four fittable datasets — 2C, ExPI, LindyHop,
> Ninjutsu — fit through ONE solver, `fit_markers`** (true-MoSh: graduated-σ-≈L2, free per-marker
> offset, torso rest-anchor + joint limits, per-bone retarget, free root). Render-verified plausible
> bodies (2C 14/17 mm, ExPI ~20 mm, ReMoCap ~18 mm). The one remaining wrongness — body GIRTH (the
> mean SMPL tummy) — is **unrecoverable from skeletal markers** and is the next phase, **§P23.v
> surface-vertex attach** (designed in `MASTER_PLAN.md`, not yet implemented).

## Active phase
- Phase ID: **P23** (ExPI/2C/ReMoCap marker solve) — DONE; **P23.v** (surface-vertex attach) — DESIGNED, NOT STARTED.
- Phase name: "MoSh surface-marker solve for joint-incongruent reference skeletons."
- Status: **in-progress** — fitting works (all 4 datasets render plausibly); girth-recovery (§P23.v) open.

## What was just done (2026-06-19 b — marker-aware anchors, commit `15a55a0`)
- **Two `fit_markers` anchor bugs fixed** (user: "ninjutsu mesh rotating around root … feet should be
  straight"). (1) Torso rest-anchor was hardcoded to {3,6,9,12,13,14}; ReMoCap observes markers there, so
  anchoring locked the spine straight and the body rotated RIGIDLY about the root. Now anchor a torso joint
  only when NO marker attaches to it. (2) Feet {7,8,10,11} now ALWAYS rest-anchored → flat human feet (was
  ankle 37° plantarflex/twist chasing the surface toe marker → ballet-pointe "alien" feet).
- **Render-verified all 4 datasets**: ninjutsu 16/13, lindyhop 12/12 (↑ from ~18), 2c 12/11 (↑ from 14/17),
  expi 27/23 (= baseline). Feet flat; ninjutsu spine3 articulates 23° (was 0). Edit is 2 lines in
  `fit.py::fit_markers` (the `aw` set). MASTER_PLAN §P23 amended ("MARKER-AWARE ANCHOR AMENDMENT").
- **Caveat:** shot_001 (the render gate) is a near-static stance, so I confirmed the *mechanism* (torso freed,
  spine articulates, feet flat) + numbers, not a sharp-turn animation end-to-end. A turning shot orbit-render
  would be the final visual proof of "no rigid root spin."
- **VPoser is NOT in `fit_markers`** (user asked). Naturalness is hand-tuned limits+anchors+smoothness only;
  `vposer.py`/`lam_vp` belongs to the old §P18 `solve_pose` path, unused by the marker solve. Wiring VPoser
  in as a soft prior would replace the per-joint anchors with a learned natural-pose prior (open option).

## What was done earlier (prior session)
- **Plan split:** moved the SMPL-fitting phases (§P9–P19, §P23) out of `koopman/KOOPMAN_MASTER_PLAN.md`
  into this package's **`MASTER_PLAN.md`** (koopman keeps P8 + operator/AE/reactor; pointer stubs left).
- **Code move:** `skeletons.py` (was `vr/vizsys/skeletons.py`) + new **`datasets.py`** (all loaders +
  fit recipes) now live here; `vr/vizsys/smpl_fit.py` is thin viz-payload glue importing this package.
- **Public API:** `datasets.py` exposes per-dataset skeleton defs + `fit_2c/fit_expi/fit_remocap/
  fit_dataset` (→ `Fit`), `DATASETS` registry, `marker_bone_scale`; all re-exported from `__init__.py`.
- **2C → marker solve:** `fit_2c` now treats the 20 BVH joints as markers (`fit_markers`); 25→14/17 mm.
- **ReMoCap → marker solve:** `fit_remocap` treats the 22 worldpos joints as markers; ~18 mm. Deleted
  the BVH-rotation-init path (`_fit_remocap`, `_world_to_local_aa`, `adapters`/`geometry` imports).
- **Shape fixes (earlier in session):** large-σ GMOF (fixed dead-zone arms), torso anchor + joint limits
  (fixed hunch/pot-belly), `bone_scale` + `s→1` prior (fixed giant), feet excluded + ratios clamped
  (fixed alien feet).
- Commits (this repo): `1e3f3b9` ReMoCap; `d7de131` 2C; `d5067b9` public API; `e963b5e` skeletons+datasets;
  `84ce1ae` MASTER_PLAN; plus §P23 shape-fix chain (`68bc52b`, `e5540c3`, `a53dccc`). vr + koopman repos
  have matching companion commits.

## Test results
- No `tests/` dir. Gated by RENDER: `viz.py {2c,expi,lindyhop,ninjutsu} --smpl` all produce plausible
  upright bodies (verified this session via `--snapshot`). Numbers: 2C 14/17 mm, ExPI 14.5 mm-mean over
  16 clips (20 mm with shape-reg), ReMoCap ~18 mm.
- NOT-YET-RUN: no automated regression suite exists — consider adding `tests/` with a marker-fit
  smoke per dataset (assert err ≤ gate + finite verts).

## Open user decisions (blockers)
- **§P23.v girth recovery (user chose "surface-vertex attach").** Body build/belly is unrecoverable
  with free joint-offsets (β stays ≈0). Implement the surface-vertex attach to make β identifiable —
  the only principled fix. See `MASTER_PLAN.md` §P23.v for the full design.

## Next concrete steps
1. (Optional) add `tests/` — one marker-fit smoke per dataset (err ≤ §P23 gate, finite mesh) →
   verify: `pytest` green.
2. **§P23.v surface-vertex attach** — define the 18 ExPI marker→SMPL-vertex landmarks; rewrite the
   virtual-marker onto LBS verts + small normal residual; make β identifiable (`lam_reg≈0.3`, drop
   joint-δ). Gate G23f: `‖β‖∈[0.5,2.5]`, shoulder/hip width within ~10% of marker widths, render shows
   a leaner build. → verify: `viz.py expi --smpl` render + the width numbers.

## Files modified this session (now committed)
- `MASTER_PLAN.md` — fitting phases extracted from koopman; §P23 marker-solve + shape-reg + 2C/ReMoCap.
- `datasets.py` — NEW; all skeleton defs + `fit_*` recipes + registry.
- `skeletons.py` — NEW (moved from vr/vizsys); generic BVH/worldpos I/O.
- `__init__.py` — exports the public dataset API.
- `CLAUDE.md`, `CONTINUATION.md` — NEW (this turn).

## Stale files in this session's cache (next session MUST re-Read)
- `datasets.py` — the live fit recipes; re-read before any fit change.
- `fit.py::fit_markers` — the shared solver (graduated σ, anchor, limits, bone_scale); re-read before tuning.
- `MASTER_PLAN.md` §P23 / §P23.v — the governing spec + the open next-phase design.

## How to resume
1. Read `MASTER_PLAN.md` §P23 (targeted, per the grep rule) + §P23.v.
2. Read THIS file.
3. Re-read `datasets.py` and `fit.py::fit_markers` before editing.
4. Resume at "Next concrete steps" #2 (§P23.v) unless the user redirects.
