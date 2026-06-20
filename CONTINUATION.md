# SESSION_CONTINUATION — 2026-06-19 — ACTIVE: §P24 convert all 4 datasets → boxing-format SMPL XML (skel2smpl runtime-free)

## §P24 Phase-3 cutover (ACTIVE, 2026-06-20) — viz loads processed XML, removing skel2smpl runtime use
- **DONE (vizsys/smpl_fit.py, top-repo commit `be3d25c`):** `smpl_fit_payload` now LOADS the precomputed
  boxing XML (flat `vr/data/<Name>/<rel>.xml`, or next-to-source fallback) and rebuilds both SMPL bodies
  via `vizsys.smpl`; GT drawn from the XML `triangulated` (M=Rx(-90) applied to match the loader frame).
  NO skel2smpl import, NO runtime fit. VERIFIED: `vr/viz expi --smpl --snapshot` renders 2 bodies + GT
  overlay on the checkerboard. Resolver OK for expi/lindyhop/ninjutsu (203 flat XMLs).
- **pose_estimation:** already loads XML via `viz.py::smpl_xml_payload`; does NOT import skel2smpl → nothing to do.
- **OPEN — 2C:** never converted to `vr/data/2C/` (only the 3 datasets were). 80/95 stale next-to-source
  `*_smpl.xml` exist (mixed, from the interrupted render batch). To finish: run the converter for 2c into
  vr/data/2C/ (add '2c' to scratch_convert_xml ROOTS or use --all next-to-source). ~20 min GPU.
- **OPEN — koopman (CONFLICT, needs user decision):** `koopman/viz.py` fits SMPL to FOUR seqs — GT
  actor/reactor AND **predicted** actor/reactor. Predictions have NO precomputed XML (generated at viz
  time) → cannot be replaced by loading XML. Koopman also uses skel2smpl as a MATH lib (se3_exp/log in
  klib/motion_bridge/preprocess_bridges, adapters). So skel2smpl can't be removed from koopman without
  (a) keeping runtime fit for predictions and (b) porting the math. Options: leave koopman on skel2smpl
  (library), OR load XML only for koopman's GT bodies + keep fit for preds. SURFACED to user.
- **OPEN — vizsys/skeletons.py:** re-exports skel2smpl BVH/worldpos I/O for the raw-GT payloads
  (two_c/expi/remocap_payload in vr/viz). Removing = port the loaders into vizsys.

## §P24 (ACTIVE, 2026-06-19 PM) — optimizeShape-per-dataset + GPU landed
- **DONE (commits `3c289e0`, `b6421cf`):** β now from a FAITHFUL `optimize_shape` port (EasyMocap
  optimizeShape:785-836) adapted per dataset via `kintree_from_marker_bone` (marker→SMPL direct bones,
  only observed → no phantom segment); keeps §P9 `spine_weight=0.1`. Solvers run on **GPU** (`_smpl_gpu`
  + `with torch.device(DEV)`; fit.py device-correct at 3 const sites; `lam_vp=0`). 500f ≈ 5–7 s compute
  (~15 s incl CUDA init), 45 MB VRAM. `fit_smpl_betas_limblen` kept (koopman tests).
- **PILOTS (GPU, all G24a<0.003mm PASS, render upright/grounded):** 2C 42/50 ‖β‖1.97 · LindyHop 31/41
  ‖β‖2.0 · Ninjutsu 46/38 ‖β‖1.9 · ExPI 56/44 ‖β‖1.4. Pilot XMLs in /tmp (NOT data/).
- **AWAITING USER (G24c):** accept plain-SMPL → run full `--all` (~596 seqs, fast on GPU, writes
  smpl.xml next to each source under data/), OR bone_scale off-ramp. Then P3 cutover `vizsys/smpl_fit.py`.
- **Stale (re-Read):** `fit.py::optimize_shape`/`fit_markers`/`smpl_joint_limits` (device edits),
  `to_boxing_xml.py::fit_to_boxing`/`_smpl_gpu`.

## §P24 (earlier, 2026-06-19 AM) — boxing-XML conversion
- **Goal (user):** "create SMPL format for all datasets; then we don't need skel2smpl anywhere — just like
  boxing for all pipelines." Convert 2C/ExPI/LindyHop/Ninjutsu → EXACT boxing `smpl.xml` (β+75-pose+trans),
  loaded by the UNCHANGED `pose_estimation/viz.py::smpl_xml_payload`. skel2smpl becomes an offline tool.
- **DONE this session (skel2smpl commits `023f859` plan, `c25455b` tool):**
  - `to_boxing_xml.py` — converts a sequence (both performers) → boxing XML. Frame Q·M=I exact
    (G24a round-trip 0.0024 mm PASS). β via `fit_smpl_betas_limblen` (optimizeShape, β-only) FROZEN +
    plain pose fit (`fit_markers(fit_beta=False, bone_scale=None, lam_s=1e4)`). ExPI → vertex-β path.
  - `fit.py` — new `fit_beta` freeze flag on `fit_markers` (additive; koopman default unchanged).
  - §P24 in MASTER_PLAN (pure-plain-SMPL decision + bone_scale off-ramp fallback + gates G24a–d).
  - MEASURED ninjutsu shot_001 200f: err 47/36 mm, ‖β‖ 1.9/1.6 (sane), s=1.0; renders via the UNCHANGED
    boxing loader as clean, UPRIGHT, grounded bodies, GT skeleton co-located.
  - **FRAME FIX (the "not z up" bug):** fit is mocap Y-up; boxing loader applies M=Rx−90 + Y-up viewer;
    boxing's own bodies are Y-up (head+Y) through the loader. Bake `FRAME_FIX=Rx(+90)` into the exported
    global_orient+trans so M·(FRAME_FIX·FK) is Y-up head+Y, matching boxing. Verified up-axis=Y (cos 0.99).
- **TRADEOFF (measured):** plain β can't carry differential limb length from joints — upper arms ~21–28%
  short vs the bone_scale fit (16 mm). Off-ramp (optional `bone_scale_N` attr, 3-line loader change) is the
  documented fallback if the render review rejects the plain limbs.
- **AWAITING USER:** render-review decision — accept plain-SMPL & run the FULL conversion (~596 seqs), OR
  take the bone_scale off-ramp, OR see a GT-overlay render first. Pilot XMLs NOT yet written to data/ (only
  /tmp). Phases left: P2 full `--all` conversion (gated), P3 cutover `vizsys/smpl_fit.py` → load XML.
- **Stale files (re-Read before edit):** `skel2smpl/to_boxing_xml.py`, `skel2smpl/fit.py::fit_markers`
  (fit_beta), `vizsys/smpl_fit.py` (Phase-3 cutover target, NOT yet touched), `pose_estimation/viz.py`
  (loader bone_scale change was added then REVERTED — currently upstream/plain).

---

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

## What was just done (2026-06-19 e — §P23.v ExPI vertex-attach, commit `051bd9b`)
- **Body SHAPE recovered for ExPI** (user: "expi/2C big belly + narrow limbs; all 4 natural shape").
  Root cause: β≡0 (SMPL neutral mean has a belly) + bone_scale stretches limb LENGTH without width. β stuck
  at 0 because the joint+free-offset δ absorbs all width. FIX (user chose vertex-attach): `_fk_verts`
  (sparse-vertex LBS, validated bit-exact vs forward_R), `fit_markers(marker_vert=…)` path (no δ, β freed),
  `EXPI_VERT` (SSM/AMASS vertex IDs; lateral-hip 1228/4712 + acromion 1861/5322 give width), `fit_expi`
  `lam_beta=1e-3` (recalibrated from 0.3 — m²-scale data term crushed β). RESULT (render): belly GONE, lean
  build, ‖β‖=1.21. TRADEOFF: residual 27→62 mm (torso markers back 123/hip 98/shoulder 86-95 carry slack).
- **VPoser** wired into `fit_markers` (commit `87c66d7`, default `lam_vp=2e-4`) — but the fit was already
  on-manifold (implaus 0.035 vs §P18's 5.30); VPoser barely helps and slightly raises jitter; smoothness is
  the jitter lever and is already near-optimal at 0.10. Honest: not the jitter fix the prompt assumed.
- **NEXT (open):** (1) refine ExPI back/hip vertex correspondence to cut the 62 mm residual; (2) **2C/ReMoCap
  still have the belly** — joint-centre markers can't observe girth, so β stays 0; they need a TEMPLATE β
  (the option the user did NOT pick) — surface to the user before adding. (3) optional scalar normal residual
  r_m for vertex markers. See MASTER_PLAN §P23.v.

## What was done (2026-06-19 c — root-orientation init, commit `4ebbfb3`)
- **Spinning-root + "elbow not fitted" FIXED** (user: ninjutsu still rotating root). Root cause: the marker
  objective is gauge-degenerate — a body turn can be absorbed by the ROOT or by twisting spine/hips/shoulders.
  From zero-init the optimizer chose the twist minimum on turning/acrobatic frames → erratic spinning root +
  contorted torso (shot_007 reactor: root flailed 9–134°, mean 59.5 mm, peak 136 mm). The actor tracked fine,
  so static shot_001 looked OK — bug only bites on real rotation.
- **FIX** `fit.py::_root_init_kabsch` + `_kabsch_so3`: seed `pose[:,0]` per-frame by rigid Procrustes/Kabsch of
  the torso markers onto the SMPL rest torso joints (`_ROOT_CORE={0,1,2,3,6,9,12,15,16,17}`). MoSh/SMPLify
  global-orient init. Render-verified: reactor 59.5→18.8 mm (peak 28.7), frame-130 contortion → natural lunge
  (17/49→17/18 mm). No regression: 2C 12/11, ExPI 27/24. MASTER_PLAN §P23 "ROOT-ORIENTATION INIT AMENDMENT".
- *Not re-rendered:* LindyHop (same recipe, mild motion, root-init only helps) — low risk, worth a glance next.

## What was done (2026-06-19 b — marker-aware anchors, commit `15a55a0`)
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
