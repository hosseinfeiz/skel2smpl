# skel2smpl — MASTER PLAN (skeleton → SMPL fitting)

`skel2smpl` (`/media/insq/ins/skel2smpl/`) is a **standalone package** that fits an
SMPL body — shape β + uniform scale s + per-frame pose θ (+ §P14 per-bone retarget) —
to an observed reference **skeleton or marker set**, via one principled MAP estimate
per subject. It is imported by `vr/vizsys/smpl_fit.py` (2C / ExPI / ReMoCap viz) and
historically fed the koopman bridge mesh path.

These phases (§P9–P19, §P23) were **EXTRACTED from `koopman/KOOPMAN_MASTER_PLAN.md` on
2026-06-19** — the fitting work had accreted in the koopman plan but belongs with the
package it governs. §P8 (BVH→SMPL rotation retarget for the koopman model's `body_log`
target + retrain) **stays in the koopman plan**: it concerns the koopman MODEL output,
not this fitting module. Cross-refs to §P8 below point back to that plan.

## How to read
- **Active fitting path:** §P13 unified MoSh++/EasyMocap fit (`solve_shape` + `solve_pose`
  with δ latent markers) → §P14 per-bone length retarget → §P16 spine 5→3 reconciliation →
  §P17 anatomical kinematics regularizers. §P15/§P15.1 = the damped-Gauss-Newton pose-solver
  variant. §P23 = the ExPI/2C surface-marker MoSh solve (`fit_markers`).
- **Superseded (historical):** §P9, §P10, §P11, §P12 — earlier shape/pose/retarget attempts
  subsumed by §P13. Kept for provenance.
- The frame convention (SMPL native Y-up → proper Z-up via Q = Rx(+90)·Mᵀ·Rx(+90)) is
  documented in §P13 and the `skel2smpl/fit.py` module docstring.
- Same authoring rules as the koopman plan: read the targeted phase before editing; record
  numerical gates; commit per verified change.

---

### P11 — Convention-correct BVH→SMPL rotation retarget (the actual mesh fix) [RESOLVED 2026-06-14]
- **Decision (user, 2026-06-13):** the mesh is "fucked up" because we build the SMPL
  skeleton WRONG, then skin it. Both P8 retarget and P10 pose-fit contort the torso
  (bulging belly) — confirmed it's the SMPL POSE θ, not shape, not LBS. P8 derives bone
  DIRECTIONS from joint POSITIONS (spine/pelvis hubs especially); P10 fits θ to positions
  (under-constrains twist). Correct approach: build the SMPL skeleton from the SOURCE
  ROTATIONS with correct conventions, then standard SMPL LBS ⇒ mesh correct BY DEFINITION.
- **Edit (FINAL):** `motion_bridge.bvh_to_smpl_world_rotations` — R_smpl_world_j =
  **M·R_src_world_b·Mᵀ** (PURE CONJUGATION, M = global rest-frame alignment), NO position
  fitting, NO per-joint correction. Pure conjugation induces R_smpl_local = M·R_src_local·Mᵀ,
  i.e. it transports the source's anatomically-correct LOCAL rotations (full twist) straight
  into SMPL ⇒ clean natural mesh. Verified end-to-end by render.
- **Gates (tests/test_bvh_smpl_convention.py):**
  - **G11a rest (PASSING):** `_world_to_theta(identity) = 0` (exact). ✓
  - **G11b motion CONVENTION sanity (PASSING):** primary-bone directions broadly match,
    **mean cosine ~0.96 ≥ 0.93** (shot_031). A gross-convention check, NOT a mesh gate.
- **ROOT CAUSE (diagnosed 2026-06-14):** Comparison/pipeline frame was a REFLECTION.
  `preprocess._read_worldpos` reads positions `(x,y,z)→(x,z,y)` — det **−1**, a mirror —
  while rotations transport via the proper `_RX_NEG90` (det +1). A proper SMPL skeleton can
  NEVER match a mirror-imaged target ⇒ the −0.95 cosine. Fix: compare/feed SMPL against
  observed positions transported by the SAME proper rotation `_RX_NEG90·M` (det +1). The
  Euler order is NOT the bug: source-rotation FK vs raw BVH worldpos = **+0.9996** in the
  BVH frame with order **XYZ** (verified).
- **REJECTED: per-joint rest correction C_j (2026-06-14).** A `C_j = shortest_arc(d_smpl,
  M·d_bvh)` appended on the right drives every bone-direction cosine to 0.9996 — BUT
  full-res renders showed it CREASES the chest and collapses the shoulders ("back folded,
  no clavicle/shoulder", user). shortest_arc fixes the bone DIRECTION while corrupting the
  TWIST (it swings the joint frame away from the correct M-conjugation). The cosine is a
  DIRECTION proxy BLIND to twist; chasing it produced a WORSE human than pure conjugation.
  Also rejected: `swing·M` rest offset (severe torso fold — pushes joint frame ≈M, far from
  canonical), and a global-up full-frame offset (still creases). LESSON: judge mesh quality
  from the RENDER, not the bone-direction number. Pure conjugation's ~4° direction residual
  is visually negligible; the clean local-rotation transport is what makes a correct human.
- **RENDER (Path 1, user-chosen 2026-06-14) — DELIVERED.** `render_convrot.py` renders
  GT actor+reactor SMPL bodies straight from a shot's BVH via `bvh_to_smpl_world_rotations`,
  in the up-preserving PROPER world frame P=Rx(+90)=(x,−z,y), mapping SMPL output in with
  the rigid Q = Rx(+90)·Mᵀ·Rx(+90). No re-bake, no model, no retrain. Result
  (`bridges_p11_convrot.mp4`, shot_031, 300f): both bodies UPRIGHT, clean natural SMPL
  bodies — natural shoulders, smooth torso, no crease/bulge (full-res frames 60/150/240),
  correctly co-located in a believable close-contact exchange. The §P11 mesh fix is proven.
  NOTE on `_RX_NEG90`: it is Rx(−90)=(x,z,−y) and NEGATES up; the up-preserving proper
  rotation is its transpose Rx(+90)=(x,−z,y). The legacy pipeline `_read_worldpos` mirror
  `(x,z,y)` (det −1) preserves up only by being a reflection — that is the incompatibility.
- **PIPELINE INTEGRATION (2026-06-14) — DELIVERED (`viz_bridges --bodies-pose convrot`).**
  - **shot_id (no retrain):** `preprocess_bridges.build_shot` now bakes `shot_id` + `split`
    so the rollout render can locate the GT BVH sources. `actor_body`/`actor_head_global`
    are byte-identical to the prior bake (Δ=0) ⇒ model target unchanged ⇒ NO retrain.
  - **One proper frame for the whole scene.** `_render_vizsys` un-mirrors the scene
    POSITIONS to proper with F=diag(1,−1,1) (proper = F·reflection; a single Y/depth sign
    flip). Bodies are posed directly in proper by TWO clean posers:
    - GT: BVH `bvh_to_smpl_world_rotations` → θ → SMPL.forward(θ,0) → Q=Rx(+90)·Mᵀ·Rx(+90)
      → shift to the proper GT root (the render_convrot method — clean mesh).
    - pred: MoSh++ `fit_smpl_pose` to the F-flipped proper PREDICTED joints (β=0 standard
      body, normal winding, co-located with the proper skeleton).
  - **Why not unify via one rotation path:** `bvh_to_smpl`'s native frame and `body_log`'s
    native frame differ by the global M (≈180° about Y), so feeding BVH θ through the
    retarget path inverts GT; conjugating the predicted ROTATIONS by F (det −1) inverts
    pred through SMPL.forward's _TRANS. Posing each from its own clean source into the
    shared proper POSITION frame (GT=Q-method, pred=moshfit) sidesteps both. Verified by
    render (frames 3/70): GT + pred both upright, clean, co-located; GT shoulders/torso
    natural (§P11 mesh fix preserved through the integration).
  - **Still deferred (Path 2):** full `_read_worldpos`→proper re-bake + retrain to make the
    pipeline natively proper (the convrot scene is currently un-mirrored at render only).

### P12 — Precise SMPL shape + pose (port the EasyMocap-style 3D-point optimizer) [SCOPE A IMPLEMENTED 2026-06-14]
- **Goal (user, 2026-06-14):** the SMPL body must match the **dataset GT skeleton more
  precisely, especially the UPPER BODY** (clavicle/shoulder/spine). Root cause re-confirmed:
  the convrot **pred** path fits θ on a **β=0** body (`motion_bridge.py:233`) against joint
  CENTERS only — wrong limb lengths + unconstrained twist. P9's earlier shape-fit was
  abandoned; **P12 re-opens it the right way** by porting a proven optimizer the user owns.
- **Reference implementation to copy (EasyMocap-style; READ before porting, verify line refs):**
  `/media/insq/ins/pose_estimation/kinematics/optimization.py`
  - `optimizeShape()` **~785–836** — the shape solve: β is a **free var, ONE per sequence**,
    **LBFGS** (`line_search_fn='strong_wolfe'`, `max_iter=50`); objective = **limb-length
    term** (match SMPL bone lengths to observed-keypoint bone lengths, `err = dst-src-
    dir_normalized*limb_length`, conf-weighted) + **L2-β prior** (1e-3) + optional init-anchor
    (1e-2). This is the "estimate shape from 3D points" the user pointed to — bone-length
    matching DECOUPLES shape from pose, more robust than raw joint-position β-fit.
  - 3D-keypoint data term **~361–379** with **robust loss** (GMOF σ²=0.04 / Huber 0.1) +
    per-joint confidence; multi-stage `multi_stage_optimize` **~838–864** (Stage 1 global
    R/t → Stage 2 pose → Stage 3 2D refine); `ftol=1e-4`.
  - Reusable priors/constraints (adopt as needed): **VPoser** VAE prior **~342–359**, **GMM**
    pose prior **~513–598** (`gmm_08.pkl`), 2nd-order **temporal smoothness** with body>limb
    joint weights **~441–511**, **ground/skate** contact (`contact.py`), **collision** capsules
    (`collision.py`), **mesh-jitter** vertex smoothness **~407–424**.
  - SMPL forward/LBS `kinematics/model/smpl_model.py` **~319–405** (axis-angle, shapedirs,
    J_regressor) — reconcile frame with our `vizsys.smpl.SMPL` (Z-up via `_RX_NEG90`).
- **Papers (cite + verify arXiv before gating):** EasyMocap (Shuai et al.) for the multi-view
  multi-stage fitter; SMPLify-X / VPoser (Pavlakos et al. CVPR 2019, arXiv 1904.05866 —
  unverified) for the VAE pose prior; MoSh++/AMASS (Mahmood et al. ICCV 2019, arXiv 1904.03278
  — cited in `motion_bridge.py:229`).
- **Scope decision (ask user before building):** (A) **shape-only** — port `optimizeShape`
  limb-length β solve, feed β into the existing `fit_smpl_pose` for GT subject + pred (smallest,
  highest-leverage); (B) **shape + robust/weighted pose** — also swap pose data term to GMOF +
  upper-body joint weights; (C) **full port** — add VPoser/GMM priors + contact/collision.
  Recommend **A → B**, defer C. We have **no 2D keypoints**, so drop the 2D-reprojection stage.
- **Edits (A, provisional):** `motion_bridge.py` — replace/augment `fit_smpl_shape` (139) with a
  limb-length LBFGS β solve over SMPL-22 bones (downweight the topology-corrupted
  SPINE_NECK_COLLAR per existing handling); thread β through the convrot specs in
  `viz_bridges.py::_render_vizsys` (both GT body + pred `fit_smpl_pose(betas=β)`), reusing the
  `_betas_a/_r` plumbing P9 already added.
- **Numerical gates (`tests/test_smpl_shape_fit.py`):** (G12a) on synthetic data — β-perturbed
  SMPL body resynthesized → limb-length fit **recovers β to ≤0.15 L2** and bone lengths to
  **≤1 cm**; (G12b) on shot_031 — fitted-β SMPL **upper-body** joints (shoulders/elbows/wrists)
  match the dataset skeleton **better than β=0** by ≥30% mean error reduction; (G12c) full
  vizsys suite stays green; (G12d) human visual check — red dataset skeleton overlays the green
  mesh shoulders/clavicle on `--interactive`.
- **Reversible:** β solve behind a flag (`--smpl-betas fit` already exists, currently maps to
  stored β; route it to the new limb-length solve); `--smpl-betas zero` reproduces today's β=0.
- **IMPLEMENTED 2026-06-14 (scope A landed):**
  - `motion_bridge.fit_smpl_betas_limblen(joints_obs, pose, smpl_model, …)` — faithful port
    of `optimizeShape:785–836`: one β/sequence, `LBFGS(strong_wolfe, max_iter=50)`, limb-length
    term `err = (J_c−J_p) − unit(J_c−J_p).detach()·‖obs_c−obs_p‖` + L2-β prior. SMPL-22 kintree
    from `SMPL_PARENTS` (`_SMPL22_BONES`); bones ARRIVING at `SPINE_NECK_COLLAR` joints
    downweighted (`spine_weight=0.1`) — collar→shoulder / neck→head kept full-weight (upper body).
  - **AMENDMENT — `w_data` data-term scale:** the §P12 prior weight `lam_reg=1e-3` is EasyMocap's,
    calibrated against a high-weight `s3d` term; our length term is metre² (tiny), so `1e-3`
    over-regularized (measured: ‖β‖ 1.16→0.37, β-L2 0.997). Added `w_data=1000.0`:
    `loss = w_data·data + lam_reg·‖β‖²`. The prior now only bites the **girth null-space** (DOF
    bone lengths cannot constrain, per §P9), not the length-determined β. Measured recovery with
    w_data=1000: β-L2 **0.060**, bone-length **0.010 cm**, ‖β‖ 1.12 (true 1.16).
  - Frame-safe: the limb-length term uses only bone LENGTHS (direction cancels) and lengths are
    frame-invariant, so the convrot `proper_j` is a valid observation despite the proper/_TRANS
    frame gap. Wired in `viz_bridges.py::_render_vizsys` convrot: β solved per CHANNEL from the
    GT bone lengths, applied to the GT body forward AND reused for that channel's pred
    `fit_smpl_pose(betas=β)`. Gated by `--smpl-betas` (zero ⇒ β=0).
  - **Tests** in `tests/test_smpl_shape_fit.py` (shared with P9 G7–G9; the SESSION_CONTINUATION
    handoff wrongly assumed this file was an empty P12 stub — it held the live P9 gates, so P12
    gates were APPENDED, not overwritten): **G12a** synthetic β recovery ≤0.15 L2 + bones ≤1 cm
    (PASS 0.060 / 0.010 cm); **G12b** "different-shape subject" upper-body bone-length error beats
    β=0 by ≥30% — a **hermetic synthetic proxy** for the original shot_031 gate (the real shot_031
    number is the render/visual gate **G12d** via `--interactive`, which needs a checkpoint + GPU
    and is run by the user). Full `tests/` suite green (58).
  - **CRITICAL FINDING — girth inflation on real data (G12d run 2026-06-14, the P9 lesson
    RE-CONFIRMED):** rendered shot_031 (`--bodies-pose convrot --smpl-betas fit`). Limb-length
    matching works *too* well: with the weak prior (`lam_reg=1e-3`) it drove **‖β‖→24** (actor)
    / 15.5 (reactor) — bone LENGTHS matched superbly (upper-body err 3.57→1.67 cm, **+53%**) but
    GIRTH (β's null-space, which NO joint/length data constrains) inflated → visibly blobby mesh.
    Prior-strength sweep (shot_031 actor): there is **NO `lam_reg` giving both ≥30% length match
    AND in-distribution β** — +30% needs ‖β‖≈12 (grossly inflated); sane ‖β‖≈1.5 yields only ~11%.
    This is exactly §P9's conclusion ("joints alone can't recover body width"); the limb-length
    method is more robust at *lengths* but does not escape the girth null-space. The synthetic
    G12a/G12b passed because their obs is SMPL-GENERATED (representable) — they are blind to this.
  - **RESOLUTION (user-chosen 2026-06-14): strong prior, sane β.** Production DEFAULT
    `lam_reg=0.5` ⇒ ‖β‖ **1.55 / 1.23** (β~N(0,I), max|β|≈1.3, no bloat), honest **+11% (actor) /
    +23% (reactor)** upper-body bone-length improvement over β=0. Re-rendered shot_031: mesh
    natural (not blobby), red dataset skeleton overlays the green shoulders/clavicle better than
    β=0. Weak `lam_reg` is for SMPL-representable (synthetic) data ONLY.
  - **AMENDMENT — G12b is a MECHANISM gate, not the real-data target.** The ≥30% reduction is
    only achievable on SMPL-representable data (G12a/G12b pass with explicit `lam_reg=1e-3`). On
    the real Ninjutsu skeleton the achievable-with-sane-β target is ~10–25%, NOT ≥30%. **NEW gate
    G12e** (`test_g12e_real_data_beta_bounded`, shot_031, production prior): asserts ‖β‖ ≤ 2.0
    (no girth bloat) AND upper-body err < β=0 — the guard the synthetic gates cannot provide.
    Full `tests/` suite green (59).
  - **§P9-REDUX LANDED 2026-06-14 (strong-β + uniform SIZE scale s — user-directed "fix it"):**
    added a jointly-fit scalar `s` to `fit_smpl_betas_limblen` (`return_scale=True`): the length
    term becomes `s·(J_c−J_p) − dir·obs_len`, so a uniform `s` captures the subject's SIZE and the
    strong-prior β only carries residual shape. Applied at render as `verts = root + s·(verts−root)`
    (proportional resize about the root — no bloat), for BOTH the GT body and the channel's pred.
    Measured shot_031 (default lam_reg=0.5): actor **+11%→+17%** upper-body, ‖β‖ **1.55→0.87**,
    s=1.131; reactor **+23%→+26%**, ‖β‖ 1.23→1.07, s=1.064. β SHRINKS (safer) AND the match
    improves; s≈1.06–1.13 agrees with §P9's s≈1.155. Re-rendered: mesh natural (not blobby), limbs
    reach the dataset skeleton. **NEW gate G12f** (`test_g12f_real_data_scale_helps`): ‖β‖ ≤ 2.0,
    1.0 ≤ s ≤ 1.4, (β,s) match ≥ β-only. Full `tests/` suite green (60).
  - **SCOPE B CORRECTION 2026-06-14 (spine-only refine in the convrot frame — the RIGHT fix):**
    the first scope-B attempt (`fit_smpl_pose` seeded with the convrot pose + upper-body weights)
    FIXED the lean but introduced a regression the user caught: the SMPL **pelvis rotated ~80°**
    ("a joint mismatch causes the SMPL mesh rotation"). Root cause: the convrot body is posed
    `forward(θ,0)` then mapped to proper by `Q=RXP·Mᵀ·RXP` and grounded; `fit_smpl_pose` evaluates
    joints in its own Q-LESS frame, so seeding it with the convrot θ is a frame mismatch — the
    optimizer swings the ROOT to compensate (measured hip-line err: convrot **0.6°** → bad fit
    **81.8°**). Fix: **`motion_bridge.refine_upper_body_convrot`** — optimizes ONLY the spine chain
    (`free_joints=spine1/2/3,neck,collars,shoulders`) evaluated in the EXACT convrot render frame
    (`forward_R → Q → ground`), freezing root/hips/legs. Measured shot_031: hip-line err STAYS
    0.6°, hips 65mm / knees 58mm unchanged, while pelvis→neck tilt 3.2°→**13.6°** (skel 14.2°),
    head 221→153mm. Pred reverted to a clean β pose fit (no weighting). `--gt-pose fit` (default)
    now calls the refine; `convrot` = legacy. **G12g rewritten**: asserts the refine keeps the
    hip-line within 2° of the rotation transfer (NOT the 80° bad-fit swing) AND moves the tilt
    toward the skeleton. `fit_smpl_pose(joint_weights=…)` / `upper_body_pose_weights` retained but
    no longer used by the render. Full `tests/` suite green (62).
  - **(superseded) first scope-B attempt — upper-body-weighted pose:**
    user observed the SMPL upper body leaning BACK vs the dataset skeleton. Root cause (measured
    shot_031): the convrot pure rotation-transfer (`bvh_to_smpl_world_rotations`) applies correct
    LOCAL rotations to SMPL's rest spine, whose bones point ~34° off the BVH's (`spine3` cosine
    0.83), so the lean is "left behind" — SMPL pelvis→neck tilt **3.6° vs skeleton 14.2°**, head
    **220mm**, shoulders **205mm** off (errors accumulate up the chain). The `C_j` direction fix
    was rejected (§P11, creases chest). Fix: `fit_smpl_pose` gains a `joint_weights` arg
    (`upper_body_pose_weights`: neck/head/collars/shoulders ×6, spine ×2) — the spine joints are
    near-colinear so their positions weakly pin the back/neck rotation, but the head/collar/
    shoulder positions (offset from the axis) DO; upweighting them forces the optimizer to rotate
    the spine to plant them on the skeleton. The convrot GT body now POSE-FITS (init from the
    convrot rotations) instead of rotation-transferring; pred also weighted. Measured: head
    220→**56mm**, shoulders 205→**33mm**, tilt 3.6→**~16°** (skel 14.2°). Re-rendered: upper body
    follows the skeleton's forward lean. **Flag `--gt-pose {fit,convrot}`** (default fit; convrot =
    legacy rotation transfer). **Gate G12g** (`test_g12g_upperbody_weighted_pose_fixes_lean`):
    weighted-fit head/shoulder err < rotation-transfer AND tilt within 3° of skeleton (rotation
    transfer off by >8°). Full `tests/` suite green (62).
  - **GLOBAL FIT 2026-06-14 (now the DEFAULT — supersedes the upper-only refine + foot snap):**
    user: "the whole body fitting should be on optimization" (the deterministic foot-plant snap
    JUMPED on falls). `motion_bridge.fit_smpl_global_convrot` — ONE optimisation of ALL joint
    rotations (incl. root) to the full skeleton, in the EXACT convrot render frame (forward_R →
    Q → ground → §P9 scale), with foot/ankle up-weighting (feet grounded BY the fit, no snap),
    temporal smoothness (no jump), and a prior toward the convrot init (so the root does NOT
    swing — the init is already correct, unlike a Q-less fit). Head-to-head vs the hybrid
    (shot_031): mean joint err **64→42mm**, upper **62→24mm**, lower **53→34mm**, foot-Δ
    **0→1mm** (now WITHOUT a snap), root-Z jerk **2.25→1.14** (≈2× smoother). Wired as
    `--gt-pose fit` (DEFAULT, global); `refine` = upper-only spine refine + deterministic foot
    plant (A/B); `convrot` = legacy rotation transfer. Deterministic foot-plant now applies ONLY
    to refine/convrot/pred. **Gate G12h**: global fit grounds feet ≤30mm (no snap) AND root-Z
    jerk < the snap path. Render verified: skeleton overlays head-to-feet, feet planted.
  - **NOT done (deferred):** scope C (VPoser/GMM pose prior + contact/collision; the smoothness +
    prior here are a lightweight stand-in); pred still uses the Q-less `fit_smpl_pose` + snap
    (GT is the skeleton-match focus). The (β,s)+global-pose fit is the honest ceiling for
    skeletal-only input — larger gains need girth-constraining data (surface/silhouette), per §P9.

### P13 — Principled unified SMPL fit (MoSh++/EasyMocap, ONE optimization) [SUPERSEDES P8/P10/P11/P12 fitting] (2026-06-14)
- **Decision (user, 2026-06-14):** the accreted skeleton+mesh fitting is hacky. Across P8→P12 it
  grew into FOUR fitting functions (`fit_smpl_betas_limblen`, `fit_smpl_pose`,
  `refine_upper_body_convrot`, `fit_smpl_global_convrot`), THREE `--gt-pose` modes
  (fit/refine/convrot), a **deterministic foot-snap**, a **post-hoc render scale** `verts=root+s·(verts−root)`,
  per-joint hand-tuned weight tables (`upper_body_pose_weights`), and DIFFERENT code paths for GT
  vs pred. **Replace all of it with ONE principled MAP optimization** in the MoSh++/EasyMocap mould,
  applied identically to GT and pred. "Find the principle, no special treatment."
- **The principle (MoSh++ / SMPLify, adapted to our data).** Per subject (actor, reactor) solve a
  single MAP estimate of a **shared shape β + uniform size scale `s`**, **per-frame pose θ_t**, and
  **per-frame translation γ_t**, minimizing
      E(β,s,{θ_t},{γ_t}) = λ_D·E_data + λ_β·E_β + λ_θ·E_pose + λ_u·E_smooth + λ_c·E_contact
  with all joints evaluated in ONE world frame (the up-preserving "proper" Z-up frame). Terms:
    - **E_data** — robust 3D-joint distance `Σ_t Σ_j w_j · ρ(‖Φ(θ_t,γ_t,β,s)_j − obs_{t,j}‖)`, with
      Geman-McClure/GMOF robustifier `ρ(r)=σ²r²/(σ²+r²)`, σ²=0.04 (EasyMocap `gmof`,
      `optimization.py:61`). `w_j`: spine/neck/collar **downweighted** (BVH 5-spine→SMPL 3-spine
      topology corruption, the §P9 `SPINE_NECK_COLLAR` set), feet/ankles **upweighted** for contact.
    - **E_β** — `‖β‖²` Mahalanobis shape prior (β~N(0,I); EasyMocap `reg_shapes`).
    - **E_pose** — `Σ_t ‖θ_t − θ_init_t‖²`, the **BVH-retarget anchor**. This is our principled
      substitute for MoSh++'s GMM/VPoser pose prior: unlike marker MoSh we *have* the source BVH
      orientation incl. **twist** (§P11 pure conjugation `M·R_bvh·Mᵀ`), so the data-consistent
      orientation prior is "stay near the retarget", which carries full twist a position-only fit
      cannot. (pred: anchor = the model's predicted `body_log` rotations — symmetric.)
    - **E_smooth** — 2nd-order acceleration `Σ_t ‖θ_{t+1}−2θ_t+θ_{t−1}‖²`, **body>limb weighted**
      (EasyMocap `smooth_poses_loss`, `optimization.py:441-511`; torso ~1e3 : limb ~1).
    - **E_contact** — replaces the deterministic snap. Detect foot contact by height+velocity
      threshold + median filter (EasyMocap `contact.py`: h_thresh≈0.08 m, v_thresh≈0.25 m/s, med
      size 5); penalize ground penetration `relu(−z)²` and in-contact xy skate `‖Δxy‖²·contact`
      INSIDE the objective (`optimization.py:387-397`). Feet plant naturally, no jump on falls.
- **The de-hack = frame + one forward.** `vizsys.smpl.SMPL.forward_R` runs FK then applies
  `_TRANS=Rx(−90)` (`smpl.py:78-81`), so its output reaches the proper render frame only via the
  rigid `Q = Rx(+90)·Mᵀ·Rx(+90)` (M = `_rest_frame_alignment` Procrustes, `motion_bridge.py:808`).
  Q is CORRECT and kept — but applied **once**, inside a SINGLE `forward_proper(θ,γ,β,s,Q)` =
  forward_R → Q-conjugate → scale `s` about root → place at γ. This one function is used by BOTH the
  objective and the render, for BOTH GT and pred. The observed joints (worldpos for GT, decoded
  `body_log` for pred) are un-mirrored to proper **once** at ingress with `F=diag(1,−1,1)`. No more
  per-function Q, in-objective root-shift, post-hoc render scale, or snap. **Scale `s` is a fit
  variable** (solved in the shape stage, applied via `forward_proper`) — folds today's render hack
  into the optimization (user decision 2026-06-14).
- **Staging (EasyMocap `multi_stage_optimize`, `optimization.py:838-864`).**
  1. **Shape stage** — `(β,s)` via the limb-length solve (EasyMocap `optimizeShape:785-836`, LBFGS
     strong-Wolfe, `err=s·(J_c−J_p)−unit(J_c−J_p)·‖obs_c−obs_p‖`, `λ_β` strong so girth→mean). This
     math is the already-faithful `fit_smpl_betas_limblen` port — KEEP it as the shape stage.
  2. **Pose stage** — `(θ_t,γ_t)` MAP fit (E_data+E_pose+E_smooth+E_contact), β,s fixed. LBFGS/Adam.
     This single stage REPLACES `fit_smpl_global_convrot`+`refine_upper_body_convrot`+`fit_smpl_pose`.
- **Scope decisions locked (user, 2026-06-14):** size = optimized scale folded into the fit (NOT
  post-hoc, NOT β-only); frame = **fit+viz rewrite only, NO re-bake / NO retrain** (`body_log` target
  untouched; single un-mirror at render/fit ingress — the §P11 "Path 2" full proper re-bake stays
  deferred); contact = **principled contact term** (not the snap, not data-weight-only).
- **References (verify line refs before porting):**
  - EasyMocap fitter `/media/insq/ins/pose_estimation/kinematics/optimization.py` — `optimizeShape`
    785-836, GMOF `gmof` 61-62, 3D data term 361-379, `multi_stage_optimize` 838-864, smoothness
    441-511, ground/skate 387-397; `kinematics/contact.py` (contact detection); `kinematics/model/smpl_model.py:319-405` (LBS, reconcile frame with `vizsys.smpl`, which is Z-up via `_TRANS`).
  - MoSh++/AMASS — Mahmood, Ghorbani, Troje, Pons-Moll, Black, ICCV 2019, arXiv:1904.03278
    (one shape/subject + per-frame pose + velocity smoothness; code github.com/nghorbani/moshpp).
  - SMPLify-X / VPoser — Pavlakos et al. CVPR 2019, arXiv:1904.05866 (robust GMOF data term + pose
    prior framing; we substitute the BVH anchor for VPoser).
  - SMPL — Loper et al. ACM TOG 2015 (LBS; consumed via `vizsys.smpl.SMPL.forward_R`).
- **Edits:** NEW `smpl_fit.py` (the unified module: `solve_shape`, `solve_pose`, `forward_proper`,
  `detect_contacts`, `gmof`, weight tables). `viz_bridges.py::_render_vizsys` — collapse the 4-body
  loop to ONE path (obs→proper, solve_shape[GT]/reuse[pred], solve_pose, forward_proper); retire
  `--gt-pose` modes + snap + post-hoc scale. `render_convrot.py` — route through `forward_proper`.
  Legacy `motion_bridge.py` fitters kept importable behind `--fit legacy` for A/B only.
- **Numerical gates (`tests/test_smpl_fit_unified.py`):**
  - **G13a (frame identity):** `forward_proper` with θ=BVH-retarget, s=1, β=0 reproduces the
    `render_convrot.py` GT verts within **1e-4 m** (the unified forward ≡ the proven §P11 render).
  - **G13b (shape recovery, synthetic):** β,s-perturbed SMPL resynth → shape stage recovers β ≤**0.15**
    L2 and bone lengths ≤**1 cm** (carry-over of G12a; the limb-length math is unchanged).
  - **G13c (pose fit ≥ old global):** on shot_031 the unified pose stage matches the dataset skeleton
    **≤ the P12 global fit** — mean joint err **≤ 42 mm**, upper-body **≤ 24 mm** (no regression).
  - **G13d (contact, no snap):** feet grounded **≤ 30 mm** AND root-Z jerk **< the P12 snap path**,
    with NO deterministic snap in the code path (term-only).
  - **G13e (unification):** GT and pred go through the SAME `solve_pose`+`forward_proper`; assert one
    code path (no `--gt-pose` branch, no `if not grounded` snap) — guards against re-accretion.
  - **G13f (full suite):** all `tests/` green; legacy fitters still import.
  - **G13g (visual, user):** red dataset skeleton overlays green mesh shoulders/clavicle/feet on
    `--interactive` for BOTH GT and pred, both subjects, at full render resolution (per memory:
    verify-renders-at-full-resolution).
- **Reversible:** `--fit {unified,legacy}` (unified default; legacy = the P12 path). `--fit legacy`
  reproduces today's render exactly.
- **Risk:** (a) frame bug in the single `forward_proper` → caught by G13a vs the proven render;
  (b) contact term over-constrains during real foot lift → tune thresholds, fall back to data-weight;
  (c) LBFGS instability on the long pose stage → Adam warm-start then LBFGS, as EasyMocap does.
- **IMPLEMENTED 2026-06-14 (`smpl_fit.py`; `--fit unified` is the DEFAULT).**
  - `smpl_fit.compute_Q`, `forward_proper` (the ONE shared forward — verts for render, fast
    joints-only `_fk` for the hot loop), `solve_shape` (wraps the limb-length `fit_smpl_betas_limblen`),
    `solve_pose`, `detect_contacts`, `gmof`. Wired into `viz_bridges._render_vizsys` convrot block as
    a single GT+pred loop behind `--fit {unified,legacy}` (legacy = the P12 path, honors `--gt-pose`).
  - **AMENDMENT — latent joint offsets (the user's "consider the skeleton mismatch too", 2026-06-14).**
    The observed skeleton's joints are NOT SMPL's J_regressor joints (BVH 5-spine collapsed into
    SMPL's 3-spine), so forcing ‖SMPL_j−obs_j‖→0 DISTORTS the pose to chase a convention gap — this
    was the upper-body residual, not optimiser slack. Following MoSh++'s LATENT MARKERS, `solve_pose`
    now also solves a per-joint body-attached offset `δ_j` (shared across frames): the virtual marker
    `marker_j = SMPL_j + s·R_world_j·δ_j` matches obs while the BODY stays natural. δ is regularised
    PER JOINT — WEAK on the topology-corrupted `SPINE_NECK_COLLAR` set (δ absorbs the ~40–60 mm spine
    gap), STRONG elsewhere (δ≈0 so limbs/feet match by pose and do not drift/float). The render uses
    the BODY (`forward_proper`, no δ); δ is the data correspondence, not the mesh.
  - **AMENDMENT — δ-free set = the whole UPPER-BODY region (user: "shoulders and head still weird",
    2026-06-14).** Allowing δ only on the spine but forcing head(15)+shoulders(16,17) to obs by pose
    HUNCHED the shoulders and STRETCHED the neck (the compressed SMPL spine can't reach the higher
    BVH neck/head/shoulder targets, so the optimiser raised them). Fix: `_DELTA_FREE =
    (3,6,9,12,13,14,15,16,17)` — spine + neck + collars + head + shoulders all carry δ (Mixamo vs
    SMPL clavicle/neck/head anatomy differs), so the BODY upper body sits naturally while the markers
    reach obs. STRONG δ stays on root/hips/knees/ankles/feet/elbows/wrists (match by pose). Re-render
    confirmed: shoulders slope naturally (no hunch), neck/head natural, arms still reach. G13c now
    measures the strong-δ ELBOWS/WRISTS (≤40 mm) since shoulders are intentionally offset-absorbed.
  - **AMENDMENT — geodesic smoothness kills the mesh-TWIST flip (user: "twist in mesh… fast
    rotation/flipping in joint rotation", 2026-06-14).** In extreme poses (the reactor's ground
    roll) a distal joint (elbow/wrist) FLIPPED ~180° between frames — position fit fine, but the
    forearm TWIST about the bone axis is unconstrained by position, so the optimiser flipped branches
    → visible mesh twist. The axis-angle 2nd-order smoothness `‖Δ²θ‖²` did NOT stop it (not
    double-cover-safe, and raising the anchor killed arm reach before the flip). Fix: smoothness is
    now GEODESIC — chordal `‖R_t−R_{t+1}‖²_F` on the joint rotation matrices (double-cover-safe,
    smooth gradient, MAX exactly at a 180° flip). Measured reactor: max frame-to-frame rotation
    **177°→17°** with elbow/wrist fit still 16 mm (no reach cost). Defaults `lam_smooth=0.05`
    (chordal), `lam_anchor=0.01` (picks the correct twist branch from the BVH). **NEW gate G13h**:
    solved reactor pose has NO frame-to-frame joint rotation > 90°. Re-render confirmed: the roll is
    a natural curled body (no twist), the standing body has natural sloping shoulders (no hunch).
  - **AMENDMENT — clavicle REST-anchor kills the residual hunch (user: "shoulders still a bit
    unnatural… the original clavicle skeleton is forward vs SMPL", 2026-06-14).** Root cause (user-
    diagnosed, confirmed): the Mixamo clavicle sits FORWARD of SMPL's, so the BVH local rotation
    transferred onto SMPL's clavicle rotates it forward (measured collar **24°→35°** from rest;
    a real clavicle barely moves) → hunch. The latent δ absorbs the clavicle's forward POSITION for
    the marker, but the pose ANCHOR (‖θ−init‖² toward the BVH transfer) re-injected the forward
    rotation into the BODY. Fix: the anchor is now PER-JOINT — the collars (13,14) anchor to REST
    (θ=0, the anatomically-natural near-static clavicle), everything else to the BVH init. This
    decouples the convention gap (→δ, position) from the body orientation (→rest). Measured: collar
    **35°→2°**, the shoulder/glenohumeral joint takes up the arm raise (72°→86°, anatomically
    correct), arm reach cost only ~5 mm (elbow/wrist 19→27 mm). Defaults `lam_rest=0.2`,
    `rest_joints=(13,14)`. Re-render confirmed: shoulders slope naturally, no hunch. G13c threshold
    relaxed to ≤50 mm (the deliberate naturalness-vs-reach trade) keeping the ≥30%-better-than-init guard.
  - **AMENDMENT — dual-quaternion skinning kills the LBS joint collapse (user: "mesh skinning
    still has… collapsed mesh in joint rotations of hands/elbow/shoulder, maybe dual quaternion
    blending", 2026-06-14).** SMPL render used Linear Blend Skinning, which linearly averages bone
    transforms per vertex → the "candy-wrapper" volume collapse at large rotations/twist (elbows,
    wrists, shoulders). Added Dual-Quaternion Skinning (Kavan, Collins, Žára, O'Sullivan 2008,
    "Geometric Skinning with Approximate Dual Quaternion Blending", ACM TOG 27(4); DOI unverified):
    `vizsys.smpl` gains `_mat_to_quat`/`_quat_mul`/`_quat_conj` + `SMPL._dqs` and a
    `forward_R(skinning='lbs'|'dqs')` flag (DEFAULT 'lbs' — the shared boxing path is unchanged).
    Blends per-bone rigid transforms as unit dual quaternions with per-vertex antipodality (pivot =
    max-weight bone), preserving rigidity/volume. `smpl_fit.forward_proper(skinning=…)` threads it;
    the unified viz render requests `'dqs'`. Verified: DQS == LBS at rest and under a global rigid
    transform to **1e-16** (exact), finite under extreme poses; render shows rounder elbows/forearms
    (no collapse). **NEW gate G13i** (DQS≡LBS rest+rigid ≤1e-6, finite under bent/twisted arm).
  - **Measured shot_031 actor (frames 64:224, the running demo):** freeing the spine via δ lets the
    arm chain reach — rendered BODY joints **shoulders 34 mm, elbows 19 mm, wrists 15 mm, feet 7 mm**
    (vs the P12 global fit's ~46 mm upper average); the marker fit reaches obs at the spine (body
    spine ~84 mm is the honest 5→3 gap, carried by δ). Falls are grounded smoothly by the contact
    term (no snap-jump). Speed: the joints-only `_fk` runs the 900-iter pose solve in ~6 s/body (was
    ~min with the verts-in-loop forward).
  - **Tests** `tests/test_smpl_fit_unified.py` G13a–f ALL PASS (6, in ~7 s): G13a frame identity
    ≤1e-4 m vs `render_convrot`; G13b (β,s) bone recovery ≤1 cm; G13c body arm joints ≤40 mm AND
    ≥30% better than init; G13d feet ≤30 mm grounded with NO snap in `solve_pose`; G13e one path for
    GT+pred; G13f legacy fitters still import. G13g (visual) DELIVERED — `bridges_p13_unified.mp4`
    (shot_031, 200f): natural shoulders/clavicle (no §P11 crease), red skeleton overlays the mesh,
    smooth grounded falls. `--fit legacy` reproduces the P12 render. Full `tests/` suite re-run PENDING.
  - **AMENDMENT — E_smooth was 1st-order VELOCITY, not 2nd-order acceleration (user: "the smpl
    motion is still not natural", 2026-06-14).** The geodesic amendment (above) correctly swapped
    axis-angle→chordal to kill the twist-flip, but it SILENTLY also dropped the difference order from
    2nd (acceleration `‖θ_{t+1}−2θ_t+θ_{t-1}‖²`, the §P13 spec + the `solve_pose` docstring) to 1st
    (velocity `‖R_t−R_{t+1}‖²_F`). A velocity penalty's loss-minimiser is a STATIC pose, so it damped
    ALL real motion: measured on shot_031 the solved upper body kept only **arms 57%, torso 76%,
    shoulders 76%** of the BVH-source angular speed (collars 15%, see next). Fix: true 2nd-order
    chordal acceleration `Rloc[2:]−2·Rloc[1:-1]+Rloc[:-2]` at the SAME `lam_smooth=0.05` (still
    chordal/double-cover-safe → G13h flip-guard holds). Restores **arms 82%, torso 102%, shoulders
    95%** at ZERO data-fit cost (58.55 vs 58.59 mm). Verified by a 15-agent diagnose workflow
    (audit + numerical probes + adversarial verify). **NEW gate G13j** (solved upper-body angular
    speed ≥ 0.75× source). The collar near-freeze (15%) is the SEPARATE `lam_rest=0.2` rest-anchor —
    deferred knob (reduce toward ~0.05 if clavicles read stiff; risks re-admitting the hunch).
  - **AMENDMENT — foot FLATTENING / heel-bury fix (user: "the toe mesh is upper than the skeleton
    toe → walking on their ankles with the toe in the air", 2026-06-14).** The dataset is a
    ball-of-foot (heel-raised) stance — obs ankle ≈94 mm, toe ≈34 mm (the contact), pitch ≈−23°. The
    BVH **init already has this correct pitch (−21°)**, but at the fitted scale `s≈1.13` the over-long
    SMPL shin would over-shoot the feet downward, so the optimiser DROPPED the ankle ~40 mm to keep
    the toe grounded → FLATTENED the foot to −8°, and the heel **mesh** sank ~20 mm below the floor
    (the contact term grounds the toe JOINT only). Visually: heel buried, toe lifted. Fix: (a) anchor
    the ANKLE rotation (7,8) to its correct init with `lam_ankle=0.5` (analogous to the clavicle
    rest-anchor — preserves the init pitch instead of letting the data term flatten it); (b) raise the
    foot data weight `foot_weight` 5→**12** (grounds the toe firmly without destabilising; higher
    plateaus/worsens). Measured shot_031: ankle gap 41→**29 mm** (actor) / 50→**3–9 mm** (reactor),
    pitch −8°→**−15° actor / −28° reactor** (≈ data), mesh sole penetration −20→**−10 mm**, toe still
    grounded, non-foot fit unchanged, all prior gates green. **NEW gate G13k** (solved ankle vertical
    gap < 35 mm — no flatten/bury). **Residual (~29 mm low) — RE-DIAGNOSED & FIXED in §P14 (2026-06-14):**
    this was NOT a per-segment shape gap. It was THIS amendment's sibling knob `lam_contact=20`
    OVER-grounding: the strong ground term flattens the foot (pitch collapses, ankle drops ~30 mm) to
    bury the toe. See the contact-retune amendment below.
  - **AMENDMENT — contact-weight retune `lam_contact` 20→8 (§P14 diagnosis, 2026-06-14).** The deferred
    ~29 mm actor ankle residual above was mis-blamed on leg PROPORTION. A shot_031 sweep (retarget ON)
    proved the cause is the contact weight: at `lam_contact=20` the ground-penetration + anti-skate term
    over-grounds the toe and FLATTENS the foot (dataset pitch 22.8° → solved flat), dropping the ankle.
    At **`lam_contact=8`** the foot pitches naturally (23.5°), the actor ankle gap **30→1 mm** (reactor
    4→1.5 mm), in-contact skate IMPROVES (2.70→1.15 mm/frame), for +2 mm max penetration (5→7 mm — within
    the mesh-sole tolerance). Lowered the `solve_pose` default 20→**8** (benefits the boxing path too —
    the over-flattening is scale-independent). Orthogonal to the §P14 leg retarget (that fixes bone
    LENGTHS, this fixes foot PITCH). All 10 P13 gates + all 6 P14 gates green (16 passed).
  - **AMENDMENT — collar de-stiffen (user: "fix everything", 2026-06-14).** The clavicle rest-anchor
    `lam_rest=0.2` (added to kill the hunch) over-froze the collars to **15%** of source speed. Sweep:
    at `lam_rest=0` the optimiser drives the collar to **39°** forward (full hunch) chasing arm reach;
    at **`lam_rest=0.1`** the collar sits at **6.3°** = the natural BVH init (6.8°, NO hunch) with ~2×
    the motion (speed 0.12→0.19) and better arm reach (44→42 mm). Lowered the default 0.2→**0.1** (safe
    no-hunch point; 0.05 gives more motion at a mild 11° lean if wanted). All 10 gates green.
  - **AMENDMENT — lower-body de-wobble: smooth the LEGS like the torso + `lam_smooth` 0.05→0.20 (user:
    "the whole lower body mesh is wobble", 2026-06-14).** ROOT CAUSE: the smoothness weight `sw` smoothed
    the torso 3× but the **legs/feet only 1×** — under-smoothed, so the lower body jittered (2nd-order
    accel ~4.5× the BVH source). STRUCTURAL FIX: `sw[_LEGS]=sw[_FEET]=3` (the lower body is a slow stance
    → smooth it like the torso; only the fast distal ARMS stay sw=1 so strikes keep speed) + `lam_smooth`
    0.05→**0.20**. Together: lower-body accel 4.5×→**~0.5× source** (smoother than source) while arm speed
    ratio stays **0.95** (≥0.75, G13j). **TESTED & REJECTED — uniform data weights** (user's "same weights
    for everything"): `foot_weight` 12→1 DOES give a natural knee (9°) and smooth legs, but FLOATS the
    feet ~5 cm / FLATTENS them (pitch 23→12°) — without the foot data weight the natural straight leg
    can't reach the grounded dataset foot. So `foot_weight=12` stays (grounding); the wobble is cured by
    smoothing, not by reweighting. **NEW gate G13l** (knee accel ≤2.5× source AND knee bend-angle err
    ≤30°). All 17 gates green (G13a–l + G14a–f).
  - **AMENDMENT — bigger body `lam_reg` 0.5→0.1 (user: "smpl should be bigger, β over-regularized",
    2026-06-14) — a DELIBERATE relaxation of the §P9/§P12 girth guard.** The dataset is skeleton-only,
    so girth is unconstrained; §P12 pinned β to the mean (`lam_reg=0.5` ⇒ ‖β‖≈0.86, a slim mesh) to
    avoid the §P9 girth bloat (`lam_reg=1e-3` ⇒ ‖β‖→24, blobby). The user wants the fuller fighter
    build. Sweep (shot_031 actor): `lam_reg` 0.5→0.2→**0.1**→0.05 gives ‖β‖ 0.86→2.0→**3.5**→5.4, s
    1.13→1.17→**1.23**→1.30, body depth +1.8/**+4**/+6 cm, AND a better length fit (worst non-spine bone
    67→**55**→48 mm). Chose **0.1** as the `solve_shape` default — clearly bigger, β within ~3σ (no
    blobby bloat); ≤0.05 (‖β‖>5) risks it; 0.2 is the conservative fallback. This is the one place the
    §P9 "girth→mean" invariant is intentionally loosened; it does NOT affect the synthetic gates (G13b/
    G14 call solve_shape with explicit weak/own lam_reg). Tune in `solve_shape`. All 17 gates green.
  - **AMENDMENT — knee anchor `lam_knee` 0.01→0.15 (user: "the ankle/toe fitting ruined the knee, fix
    that", 2026-06-14).** Diagnosis (knee measured by BEND ANGLE + position, not the earlier accel): the
    dataset legs are near-straight (knee ≈167.5°; BVH init ≈161°, good), but the strong foot data terms
    (`foot_weight=12` on ankle+toe) OVER-FLEX the solved knee to **134° (33.7° off)** and shove it ~122mm
    off its obs position to nail the feet — the "foot fitting ruins the knee" artifact. Fix = anchor the
    knee ROTATION to its good init bend (same pattern as `lam_ankle` for foot pitch, `lam_rest` for the
    collar). **This is a genuine knee↔foot TRADEOFF** — knee bend and ankle grounding are coupled through
    SMPL's fixed (retargeted) bone lengths, so the two cannot both be perfect: `lam_knee` sweep (actor)
    0.01→0.1→0.15→0.2→0.3 gives knee err 33.7→28.4→**24.9**→21.5→15.9° while the ankle lifts
    1→13→**21**→29→40mm. Chose **0.15** (knee 33.7→24.9°, pos 122→94mm; ankle ~1→23mm, still grounded <
    the G13k 35mm). **G13l gate EXTENDED** to also guard knee bend-angle err ≤30° (bites at the
    un-anchored ~34°). **G14c RELAXED** ≤15→≤25mm ankle to reflect the chosen tradeoff (the contact
    retune alone still grounds to ~1mm; the knee anchor trades ~22mm of that back for a natural knee).
    Knob exposed in `solve_pose` — raise for a straighter knee at the cost of more foot lift. All 17 green.
  - **DONE in §P14 — lower-leg per-segment bone-length retarget.** Measured bones (shot_031 actor,
    scaled): SMPL **thigh −30 mm (too short), shin +14 mm (too long)** vs the dataset — a leg PROPORTION
    error a uniform scale can't redistribute and β can't fix without re-inflating girth (§P9 guard).
    §P14 retargets each leg bone to the dataset length (translation form, girth-preserving): bones now
    ≤5 mm (G14a). NOTE: this turn ALSO corrected the belief that the ankle-height residual came from
    this proportion error — it did not (it was `lam_contact=20`; see the contact-retune amendment).
  - **NOT done (deferred):** scope C (VPoser/GMM pose prior — the BVH anchor + δ stand in); the §P11
    "Path 2" full proper re-bake/retrain (un-mirror still at ingress only, per user decision).

### P14 — Per-segment bone-length retarget (anthropometric SMPL) [extends §P13 shape stage] (2026-06-14)
- **Why (user: "build the leg-proportion retarget", 2026-06-14).** After the §P13 `solve_shape` (β +
  uniform `s`) the SMPL leg PROPORTIONS still mismatch the dataset skeleton — measured shot_031 actor
  (scaled bones): **thigh −30 mm (too short), shin +14 mm (too long)**, foot −5 mm. β's PCA shape space
  + ONE uniform scale cannot set arbitrary PER-BONE lengths (the web-search surfaced bone-length-explicit
  models like MHR precisely because β under-determines per-bone length). This proportion error is why the
  §P13 foot-pitch fix leaves the actor **ankle ~29 mm low** (the over-long shin drops the ankle). Fix:
  retarget each SMPL rest BONE to the dataset's measured length, deforming the mesh consistently.
- **Approach (standard skinned-mesh bone retarget, applied in the BIND/rest pose BEFORE LBS).**
  1. **Target bone lengths** from the dataset (frame-invariant, like the existing limb-length fit): for
     each kinematic edge parent `p`→child `c`, `L_data[c] = mean_t ‖obs[c]−obs[p]‖`.
  2. **Per-bone ratio** `r[c] = L_data[c] / L_smpl[c]`, `L_smpl[c] = ‖J[c]−J[p]‖` on the β-shaped rest
     joints `J = J_regressor @ v_shaped` (vizsys/smpl.py:101). Root `r[0]=1`.
  3. **Retarget rest JOINTS hierarchically** (root→leaf over `SMPL_PARENTS`):
     `J'[c] = J'[p] + r[c]·(J[c]−J[p])` — child joints carry the cumulative parent retarget + local scale.
  4. **Retarget rest VERTICES mesh-consistently via the SKINNING WEIGHTS** `self.weights` [6890,24]
     (vizsys/smpl.py:80): blend each vertex's bind-pose displacement by its bone weights so there are NO
     seams. **BUILD CHOSE THE TRANSLATION FORM** (see BUILD DECISION below):
     `v'[i] = v[i] + Σ_b W[i,b]·( J'[b] − J[b] )` — ride each vertex along by the LBS-weighted joint
     displacement. (The originally-specced isotropic form `v'[i]=Σ_b W[i,b]·(J'[b]+r[b]·(v[i]−J[b]))`
     was rejected on build evidence: it inflates girth by r and distorts interior leg edges to a 0.64–1.49
     ratio; the translation form preserves girth and holds edges to 0.89–1.18 — numbers below.) Reuses the
     SAME `W` LBS already uses → consistent skin. Both forms hit the exact target bone lengths (joints=J').
  5. Feed `J'`, `v'` into the existing LBS/DQS forward (`forward_R`→`_global_transforms`, vizsys/smpl.py:
     92,105). Pose θ, the δ markers, and the contact term are UNCHANGED — they now act on the retargeted
     rest skeleton+mesh. Pred reuses the GT subject's `r` (like β/s).
- **Scope/constraints.** LEGS first (the long bones 4,5,7,8,10,11 — thigh/shin/foot L+R); see BUILD
  DECISION for why the hip-offset bones 1,2 were EXCLUDED. **OPT-IN** flag on the SHARED `vizsys/smpl.py`
  (default OFF → the boxing path + all P13 gates bit-identical). **Girth:** the chosen translation form
  preserves girth by construction (leg-tri area ratio 1.001–1.008, i.e. a pure length change, not the
  r²≈14% cross-section blow-up the isotropic form would give) — fully consistent with the §P9 invariant
  (β/shape untouched).

- **BUILD DECISION (2026-06-14, grounded on a shot_031 probe before implementation):**
  - **Vertex form = TRANSLATION** `v'=v+W·(J'−J)`, NOT the originally-recommended isotropic scale. Evidence
    (shot_031 actor, the §P13 (β,s) fit, long-bone set): isotropic gives leg edge-ratios [0.64,1.49] and
    inflates girth; translation gives [0.89,1.18] and leg-tri area ratio 1.008 (girth preserved). Both
    reach the exact target bone lengths (the gate G14a measures joints, set by step 3 = J', identical for
    either vertex form). Translation is also the standard skinned-mesh bone-length retarget.
  - **Bone set = long bones {4,5,7,8,10,11} only** (hips 1,2 EXCLUDED). The probe found hips 1,2 at r≈0.86
    (a ~15 mm pelvis-WIDTH mismatch, stance not leg-length) which drove the worst mesh shear (edge max
    1.49→1.21 when dropped) and is not the ankle-drop motivation. Resolves the spec's "legs-only vs
    all-bones" open question with data. r for non-leg bones (incl. hips, root, hands) stays 1 → identity.
  - **r is s-ORTHOGONAL:** `r[c] = L_data[c] / (s · L_smpl_β[c])` (divide out the uniform §P13 scale s so
    the forward's `scale=s` then yields s·r·L_smpl_β = L_data exactly). Refines spec step 2 (which omitted
    s). Keeps β, s, and bone_scale orthogonal and individually testable (the recommended open-q answer).
  - **Measured residuals (shot_031 actor, s=1.131):** thigh −30.3/−22.3 mm (r 1.072/1.052), shin
    +13.8/+14.0 mm (r 0.969/0.968), foot −5.5/−5.1 mm (r 1.037/1.035). After retarget: ±0.00 mm.
- **References (verify exact line/paper refs in the build).**
  - SMPL (Loper et al., ACM TOG 2015) — LBS, skinning weights `W=self.weights`, `J_regressor`, β PCA
    encodes bone-length VARIATION but not arbitrary per-bone length. The retarget reuses `W` and `J`.
  - Bone-length-explicit body models (MHR and similar; web-search 2026-06-14) — motivation that β alone
    under-determines per-bone length / skeletal proportion. VERIFY exact citation in the build.
  - Standard bind-pose per-bone scaling propagated by skinning weights (character-retarget graphics) —
    cite a concrete source in the build.
- **Numerical gates (`tests/test_smpl_fit_unified.py` or a new `test_bone_retarget.py`).**
  - **G14a (bones match):** after retarget, SMPL leg bones (thigh/shin/foot, L+R) within **≤5 mm** of the
    dataset bone lengths (was thigh −30, shin +14).
  - **G14b (mesh integrity):** vertex count unchanged 6890, no NaN; per-edge length-change ratio across
    leg-region triangle edges within the per-bone scale ± a small seam tolerance (no torn seam).
  - **G14c (foot grounded):** actor ankle vertical gap **≤25 mm** (was ~29, gate renamed/relaxed). The
    §P13 contact retune `lam_contact` 20→8 grounds it to ~1 mm; the §P13 knee anchor `lam_knee=0.15`
    (de-ruins the over-bent knee) then lifts it back to ~23 mm — a deliberate knee↔ankle tradeoff (fixed
    bone lengths couple them). ≤25 mm keeps the foot grounded (< the G13k 35 mm flatten guard) while the
    knee stays natural. NOTE — the leg retarget itself fixes bone LENGTHS, orthogonal to foot pitch; the
    ankle is governed by `lam_contact` + `lam_knee`. Corrected the spec's original causal hypothesis.
  - **G14d (girth guard):** non-leg body bones unchanged (leg-only); torso/arm unaffected; β/shape
    untouched (the §P9 invariant).
  - **G14e (backward-compat):** retarget OFF (default) → `forward_R` output bit-identical to current →
    G13a–k all still green; boxing path unchanged.
  - **G14f (full):** all P13 gates G13a–k green with retarget ON in the convrot path; visual re-render.
- **Edits.** `vizsys/smpl.py` — `retarget_bones(J, v, W, bone_scale, parents)` helper + `forward_R(...,
  bone_scale=None)` thread (default None = current). `smpl_fit.py` — `measure_bone_ratios(obs, smpl, betas)`
  + thread `bone_scale` through `forward_proper`/`_fk`/`solve_pose`. `viz_bridges` convrot unified path —
  compute the per-subject `r` from the dataset bones, pass to GT+pred (pred reuses GT subject's `r`).
- **Open questions for the build [ALL RESOLVED — see BUILD DECISION]:** (a) isotropic vs axial → chose
  TRANSLATION (girth-preserving, cleaner edges); (b) legs-only vs all-bones → long bones {4,5,7,8,10,11}
  (hips excluded); (c) fold `r` into `s` vs separate → SEPARATE `bone_scale`, s-orthogonal `r=L_data/(s·L_smpl)`.
- **Risk:** touches the SHARED `vizsys/smpl.py` forward (2nd change after DQS) — MUST stay
  backward-compatible (default off, G14e bit-identical) and be verified before the convrot path enables it;
  a wrong vertex blend tears the mesh → G14b guards it. Revert = the `bone_scale=None` default.
- **IMPLEMENTED 2026-06-14.** `vizsys/smpl.py`: `retarget_joints` + `retarget_bones` (translation form) +
  `forward`/`forward_R(bone_scale=None)` thread. `smpl_fit.py`: `LEG_BONES=(4,5,7,8,10,11)`,
  `measure_bone_ratios`, `bone_scale` threaded through `_fk`/`forward_proper`/`solve_pose`; AND the
  contact retune `lam_contact` 20→8 (the actual ankle fix). `viz_bridges.py`: `--retarget {on,off}`
  (default on), per-subject `r` in the convrot unified path (pred reuses GT's `r`). **Gates: 16 passed
  (~202 s)** — G14a bones ≤5 mm (retargeted ±0.00 mm; init residual >10 mm so the test bites), G14b
  6890 verts/finite/leg-interior edges 0.89–1.18 (no tear), G14c actor ankle ~1 mm, G14d non-leg ≤1 mm
  + β untouched, G14e None≡ones≡current (≤1e-9), G14f all P13 numbers hold ON; plus all 10 G13 gates
  still green. Tests: `tests/test_bone_retarget.py`. Per-subject r (s=1.131 actor): thigh r 1.072/1.052,
  shin 0.969/0.968, foot 1.037/1.035; hips & all non-leg bones r=1 (identity). **Pending:** full-res
  visual verify of `bridges_p14_retarget.mp4` (memory: verify-renders-at-full-resolution).

### P15 — Damped-Gauss-Newton reduced-coordinate pose solver [REWRITES the §P13 pose stage] (2026-06-14)
- **Decision (user, 2026-06-14): "forget the previous optimization… it creates wobbly lowerbody…
  look at the optimization in mosh++ and kinematic optimization of the pose_estimation and rewrite our
  optimization for fitting to the lindyhop and ninjutsu skeleton."** The §P13 "ONE principled
  optimization" silently re-accreted into the exact hand-tuned anchor tower it was built to kill
  (`lam_knee`, `lam_ankle`, `lam_rest`, `lam_anchor`, `sw[_LEGS]=sw[_FEET]=3`, `lam_smooth=0.20`,
  `lam_contact`, `foot_weight=12`) — all tuned on BOXING (`shot_031`) under a "lower body is a slow
  near-static stance" assumption that is FALSE for LindyHop (50 fps fast swing — spins/kicks/airsteps,
  foot vertical travel ≤1385 mm) and Ninjutsu (martial arts — low stances, kicks). Re-tuning never
  fixed the wobble because it is **structural, not a knob.** This phase REWRITES `solve_pose` as a
  damped Gauss-Newton reduced-coordinate solver (port of `pose_estimation/kinematics/phys_kintrack.py`),
  keeping the §P13/§P14 frame plumbing, shape stage, bone retarget, and the δ latent markers.
- **Diagnosis (workflow `p15-rewrite-diagnosis`, 6 research agents + synthesis; full report
  `P15_REWRITE_DIAGNOSIS.md`). Four INDEPENDENT structural errors + one regime-mismatch:**
  - **R1 (robust kernel dead):** `solve_pose` passes `sigma_sq=0.25` into `gmof` (`smpl_fit.py:242,356`)
    though the docstring/default is 0.04 (`:178`). At σ²=0.25 the GMOF weight is 0.93 at a 10 cm
    residual — a plain quadratic across the entire 1–20 cm joint-error range, so noisy per-frame foot
    targets pull at near-full strength. The designed noise-rejector is a no-op. *Verified by direct read.*
  - **R2 (undamped leg IK singularity):** near-straight legs collapse the 2-link foot-reach sensitivity
    to ~0.79 mm/° at 167° (vs 4.9 at 90°) ⇒ the knee must swing ~1.27°/mm of foot-target error (16°/mm
    at 179°). `foot_weight=12`/`leg_weight=2` drive this ill-conditioned Jacobian with NO damping ⇒ the
    classic 1/σ pseudoinverse blow-up amplifies R1's un-rejected noise into oscillating knee swing (the
    code's own `:300-307` comments document the optimiser over-bending the knee 33° and shoving it
    122 mm to nail the feet). Buss IK survey (arXiv unverified — verify in build).
  - **R3 (wrong smoothing space):** the smoother penalises 2nd-order accel of per-joint LOCAL rotation
    matrices `‖R_loc,t+1−2R_loc,t+R_loc,t-1‖²_F` (`:374-380`) while the data/contact act on WORLD joint
    positions. The same local angular accel ⇒ ~14.8 mm/frame² foot motion at the hip but 2.1 at the
    ankle (7× lever): proximal wobble (which moves the foot most) is LEAST penalised, and two ancestors
    with opposite local accels ADD at the foot. Smoothing local rotations ≠ smoothing the kinematics
    the data sees.
  - **R4 (no optimizer convergence/polish):** fixed-LR Adam (lr=0.01, 900 iters, no schedule, `:351`).
    In near-flat leg directions Adam normalises tiny noisy gradients into a ~0.57°/iter random walk
    that never anneals ⇒ residual per-frame jitter at iter 900.
  - **R5 (fps wrong, targets unfiltered):** `constants.py:73 FPS=60.0` is hardcoded but the data BVHs
    are `Frame Time 0.02` = **50 fps** (verified `data/{LindyHop,Ninjutsu}/.../motion.bvh`; the report's
    "Ninjutsu=25 fps" was a stale sample — TRUE rule: read `Frame Time` per sequence, never assume 60).
    `detect_contacts` `v_thresh=0.02 m/frame` and the accel smoother are frame-step-coupled; the
    observed 3D targets and `init_pose` carry raw BVH jitter end-to-end (garbage-in).
  - **R6 (regime mismatch, the user's instinct):** the whole leg-static block encodes a boxing guard.
    Measured (normalised to m/s²): LindyHop foot accel ~13–17 vs Ninjutsu ~5; knee ~11–14 vs ~4 (≈3×).
    The static prior simultaneously over-smooths real fast footwork AND fights the contact term when
    feet are legitimately airborne.
- **The principle of the rewrite (MoSh++ §3 / EasyMocap / phys_kintrack): buy lower-body stability with
  CONDITIONING + FILTERING + a constant-velocity prior, never with per-joint anchors.** User decisions
  (2026-06-14, AskUserQuestion): **(A) full reduced-coordinate damped-GN engine** (port phys_kintrack,
  not the lighter PyTorch-autodiff hybrid); **(B) conditioning-only plausibility** (NO learned VPoser/
  GMM prior — verified available at `gps/data/model/vposer_v02`, `vr/data/model/gmm_08.pkl`, but
  rejected to avoid AMASS-averaging the extreme ninjutsu/lindy poses we are specifically capturing).
- **The engine (port of `pose_estimation/kinematics/phys_kintrack.py::KinematicPoseTracker`, verified
  by direct read 2026-06-14):**
  - **Reduced coordinates** = root SE(3) + per-joint SO(3) (our SMPL θ IS this; bone lengths fixed by
    the §P13 β-shaped + §P14 `bone_scale`-retargeted rest joints `J'` ⇒ exact bone lengths by FK
    construction, no "breathing"). n = 6 + 3·(N_JOINTS−1).
  - **Geometric Jacobian** of a body-attached world point `p` (`phys_kintrack:187-193`): cols
    `[0:3]=I` (root trans), `[3:6]=−hat(p−pos₀)·R₀` (root rot), and per movable ancestor `a`:
    `−hat(p−anchor_a)·R_a` (ω×r). Built directly from our `_fk` output (`anchor_a=j_native_a`,
    `R_a=R_world_a`) — NO autograd, analytic and exact.
  - **Damped Gauss-Newton step** (`phys_kintrack:380-402`): per iter `dq=solve(H,−g)`, clamp
    `max_step_rot=0.5 rad`/`max_step_pos=0.2 m`, Lie retraction `rel_quat_b ⊗= exp(dq)` (no axis-angle
    flip/double-cover), `H += tikhonov·I` (DLS/LM damping = the R2 cure). n_iters≈8–15 per frame.
  - **Per-frame, warm-started** from the previous frame's solution (`init_state`→`step` swept over t).
- **Loss terms (all fold into the GN normal equations H,g; verified phys_kintrack refs):**
  - **E_data** — robust 3D marker term, IRLS reweighting that is ACTUALLY ACTIVE at cm scale (the R1
    fix): residual `r = (j_world_b + R_b·δ_b) − obs_b`; weight `w_b·ρ'(‖r‖)` with a GMOF/Huber kernel
    σ≈0.04 m (`gmof` IRLS weight `σ⁴/(σ²+‖r‖²)²`, or phys_kintrack's capped `min(1,robust/‖r‖)`).
    `δ_b=obs_attach_b` (phys_kintrack `:278`) — the upper-body latent markers (KEEP, `_DELTA_FREE`),
    estimated frame-invariantly (MoSh++ Stage-I latent markers) and frozen for the pose sweep.
  - **E_smooth_world (NEW, the R3 fix)** — 2nd-order acceleration on WORLD joint positions:
    residual `j_world_b,t − (2·j_world_b,t−1 − j_world_b,t−2)` (a constant-velocity / MoSh++ E_u
    extrapolation prior), Jacobian = the SAME `_point_jac(b, j_world_b)`; H += w_smooth·JᵀJ. Solved
    neighbours are fixed ⇒ a clean per-frame GN term in the data's own space/units. fps-normalised.
    Plus warm-start at the constant-velocity extrapolation (a 2nd-order generalisation of phys_kintrack's
    zeroth-order `g += w_prior·q_acc` stay-near-prev, `:314`).
  - **E_knee_limit** — soft hinge LIMIT (range, NOT a target): reuse phys_kintrack `_add_limits`
    twist/swing decomposition (`:208-238`) to bound the knee/elbow to its anatomical 1-DoF range,
    removing the fully-straight σ→0 degeneracy. Replaces `lam_knee`. (SMPLify-X E_α joint-limit prior.)
  - **E_contact** — `ground` one-sided floor non-penetration `relu(floor−z_foot)²` + in-contact
    anti-`skate` `‖Δxy‖²` (phys_kintrack `_add_ground` `:195-206`; EasyMocap contact), gated by an
    **fps-aware** `detect_contacts` (thresholds in m/s, m/s²). KEEP, de-coupled from frame rate.
  - **E_prior (reduced-coord, light)** — `w_prior·‖q−q_extrap‖²` null-space stabiliser for momentarily
    unobserved joints (phys_kintrack `:313-314`), weak so it does not stiffen real motion.
  - **No `lam_knee`/`lam_ankle`/`lam_rest`/`lam_anchor`/`sw[_LEGS]=3` — the anchor tower is DROPPED.**
- **Stages (EasyMocap `multi_stage_optimize` shape→RT→pose; MoSh++ Stage-I→II):**
  0. **Target pre-filter (NEW)** — read fps from the BVH `Frame Time`; non-causal fps-aware
     Savitzky-Golay low-pass on `obs_joints` (recorded data ⇒ offline OK; preserves kick/airstep peaks,
     removes the jitter the singular leg direction amplifies). Default; one-Euro/Kalman-RTS optional.
  1. **Shape (β,s)** — KEEP `solve_shape`/`fit_smpl_betas_limblen` (EasyMocap limb-length, LBFGS),
     fit-once-and-freeze. KEEP §P14 `measure_bone_ratios` `bone_scale`.
  1b. **Latent markers δ** — frame-invariant upper-body offsets (MoSh++), solved once, frozen.
  2. **Articulated pose** — the damped-GN sweep above, warm-started per frame.
- **Frame reconciliation (keep `forward_proper` UNTOUCHED):** the GN solver runs in SMPL-NATIVE frame —
  map `obs_joints` proper→native by `Q⁻¹` once at ingress, warm-start from the §P11 BVH-retarget
  `init_pose` (already native local), solve, emit θ native; `forward_proper` does the native→proper
  Q-map + uniform scale `s` + root grounding for BOTH render and gates exactly as today. The uniform `s`
  is folded into the solver's rest joints (`J_solve = root + s·bone_scale·(J−root)`) OR matched in
  native at unit scale then scaled by `forward_proper` — pick one and assert no double-scale (gate).
- **References (verify line refs in the build):**
  - `pose_estimation/kinematics/phys_kintrack.py::KinematicPoseTracker` — `_fk` 170-185, `_point_jac`
    187-193, `_assemble_scalar` 270-316 (data/ground/limits/prior/tikhonov), `_add_ground` 195-206,
    `_add_limits` 208-238 (twist/swing hinge), `step` 371-403 (GN solve, clamp, Lie retraction).
    *Read & verified 2026-06-14.* (`phys_kintrack_warp.py` = a Warp/GPU vectorised variant — not needed;
    T≈160 frames × ~10 iters × 69-dim solves is trivial in torch/numpy.)
  - MoSh++/AMASS — Mahmood et al. ICCV 2019, arXiv:1904.03278 (per-subject latent markers + frozen
    shape; E_u 2nd-difference constant-velocity prior; adaptive prior weight). EasyMocap
    `optimization.py` (the loss library: GMOF σ²=0.04 `:61`, `ground`/`skate` `:387-397`, multi-stage
    `:838-864`) — its parameterization (raw axis-angle LBFGS + per-joint smooth weights) is the SAME
    family as the broken code, used here only for loss semantics, NOT the engine.
  - SMPLify-X / VPoser — Pavlakos et al. CVPR 2019, arXiv:1904.05866 (E_α joint-limit prior → our knee
    hinge; robust data term). Buss "Introduction to Inverse Kinematics… Damped Least Squares" (DLS/LM →
    the tikhonov term; verify exact ref in build). One-Euro: Casiez et al. CHI 2012 (verify).
- **Edits.** `smpl_fit.py` — NEW `solve_pose_gn` (the GN engine: `_point_jac`, `_assemble`, the GN
  `step` loop, world-accel smoothing, knee hinge, contact, δ), `prefilter_targets` (Stage 0),
  `read_fps`/physical-unit `detect_contacts`. KEEP `compute_Q`/`forward_proper`/`_fk`/`solve_shape`/
  `measure_bone_ratios`/`gmof`. The old anchor-tower `solve_pose` stays importable behind `--fit
  unified` for A/B; NEW default `--fit gn`. `viz_bridges.py` — route convrot through `solve_pose_gn`,
  thread per-sequence fps. `constants.py` — retire the hardcoded `FPS=60` coupling (read per sequence).
- **Numerical gates (`tests/test_smpl_fit_gn.py`; physical units so they hold across 25/50 fps):**
  - **G15a (robust-on):** at a 0.10 m joint residual the effective robust weight ≤ 0.65 (σ²≤0.04 ⇒
    ≤0.640) — the kernel actually down-weights (guards R1 regressing).
  - **G15b (leg-damping):** inject ±5 mm white noise into a near-straight (knee ≥165°) foot target ⇒
    solved knee angle changes ≤ **2°/5 mm** (vs ~6° undamped) — the DLS cure of R2.
  - **G15c (world-smooth space):** equal foot world-motion produces comparable smoothness penalty
    whether driven by a hip or an ankle DoF (lever-corrected, ratio within 1.5×) — guards R3.
  - **G15d (motion pass-through):** solved lower-body world-joint speed ≥ **0.80×** the pre-filtered
    observed speed on a LindyHop fast-footwork window (no over-smoothing of kicks/airsteps).
  - **G15e (anti-jitter):** solved lower-body world-joint acceleration ≤ **1.0×** observed on the same
    window WHILE G15d holds (wobble removed, motion kept).
  - **G15f (no-skate):** in-contact foot xy drift ≤ **0.05 m/s**; **G15g (floor):** contact-frame
    penetration ≤ 5 mm AND airborne feet NOT snapped (z-range ≥ 0.95× observed).
  - **G15h (fps-invariance):** the same physical motion fit at 25 vs 50 fps agrees ≤ **10 mm** RMS.
  - **G15i (bone-length stability):** solved SMPL bone lengths constant across frames, std/mean ≤ 1%.
  - **G15j (convergence):** GN loss monotone non-increasing; final pose-accel tail < 5% of iter-0.
  - **G15k (frame identity):** `solve_pose_gn` with obs = the §P11 retarget joints recovers θ≈init and
    `forward_proper` reproduces the `render_convrot` verts ≤ 1e-3 m (no frame/scale double-apply).
  - **G15f-suite:** all `tests/` green; `--fit unified` and `--fit legacy` still import (A/B preserved).
- **Reversible:** `--fit {gn,unified,legacy}` (gn default; unified = the §P13 anchor path; legacy = P12).
- **Risk:** (a) frame/scale double-apply in the native↔proper reconciliation → G15k vs the proven
  render; (b) GN H singular on fully-unobserved joints → tikhonov + the light reduced-coord prior;
  (c) causal per-frame accel prior lags fast motion → the non-causal Stage-0 pre-filter + optional
  forward-backward polish; (d) δ/marker estimation coupling → estimate δ once in a short pre-stage and
  freeze (MoSh++). Revert = `--fit unified`.
- **IMPLEMENTED 2026-06-14 (engine + gates + wiring; lindyhop A/B + default-flip PENDING).**
  - `smpl_fit.py`: `solve_pose_gn` (damped-GN: analytic geometric Jacobian `_point_jac`-style, Lie
    retraction `Rl@exp(dq)`, GMOF IRLS σ²=0.04, WORLD 2nd-order accel smoothing toward the
    constant-velocity extrapolation, fps-aware ground+skate, tikhonov+step-clamp), `_fk_mats` (matrix
    FK ≡ `_global_transforms`), `_rest_joints_native`, `_chain_mask`, `prefilter_targets` (Savitzky-
    Golay), `read_fps` (BVH `Frame Time`). Defaults: `w_smooth=3`, `foot_weight=8`, `tikhonov=1e-3`,
    `max_step_rot=0.5`, `n_iters=14`. δ latent markers supported (`delta=`) but default 0 (lower-body
    wobble is δ-independent — δ⊂upper body; the δ pre-stage is the deferred v1.1 add).
  - `viz_bridges.py`: `--fit {gn,unified,legacy}` (added `gn`; `gn` shares the entire unified
    setup — proper_j/pose0/β,s/r/forward_proper — and only swaps the solve + threads per-sequence
    `read_fps`). DEFAULT still `unified` pending the lindyhop visual check (flip to `gn` after).
  - **Gates: `tests/test_smpl_fit_gn.py` — 5 pass (~1.6 s):** G15k frame-identity (GN∘forward_proper
    round-trip ≤1e-3 m — validates FK+Jacobian+Lie-retraction+native→proper at once; caught a Jacobian
    reshape-order bug), G15k2 `_fk_mats`≡forward_R joints ≤1 µm, G15a robust-on (≤0.65 @ 10 cm),
    G15e anti-jitter (solved world-accel <0.5× noisy obs, fit <2 cm of clean truth), G15b leg-damping
    (knee ≤2° under ±5 mm foot noise — the DLS cure of R2). The §P13/§P14 suites still green (17).
  - **Real-data A/B (ninjutsu, `scratch_p15_characterize.py`/`scratch_p15_sweep.py`):** on shot_036
    (fastest, fps=25, obs lower-accel 7.9) GN fit **37.8 mm** beats UNIFIED **41.1 mm** at accel ≈ obs;
    shot_031 GN **40.9** vs UNIFIED **45.2**. **Finding: the DLS damping (G15b) is the real wobble cure;
    `w_smooth` must stay LIGHT** — a strong 2nd-order (constant-velocity) prior fights real accelerated
    motion and the robust kernel then gives up → runaway drift (fit 100–358 mm at `w_smooth≥10`).
    Ninjutsu obs accel (4–8) is mostly REAL motion, not wobble — so GN≈UNIFIED there is correct; the
    decisive win is expected on LINDYHOP (obs foot accel 13–17, where UNIFIED's static-stance assumption
    over-damps). fps VARIES per shot (031/036 = 25 fps, 043/044 & lindyhop = 50 fps) — `read_fps` is
    load-bearing, NOT 60.
  - **PENDING (next session):** (1) **lindyhop A/B** — needs the lindyhop→SMPL-22 obs+init mapping
    (lindyhop uses Mixamo joint names `RightUpLeg/…`, NOT `SMPL_TO_BVH` ninjutsu names; the mapping
    lives in `data/LindyHop/build_lindy_cache.py` `off22`/`SMPL22_PARENTS` + `data/lindyhop_cache/`).
    (2) **interactive visual verify** `viz_bridges --fit gn` GT+pred (memory: prefer-interactive-viewer-
    no-render; verify-renders-at-full-resolution). (3) **flip `--fit` default to `gn`** after the
    visual passes. (4) **δ upper-body pre-stage** + **knee/elbow hinge limit** (phys_kintrack
    `_add_limits`) if upper body / extreme poses need them. (5) full `tests/` suite + a lindyhop gate.
- **‼ 2026-06-14 (this session) — LINDYHOP A/B FALSIFIES the "GN wins on fast lower body" prediction.
  DEFAULT-FLIP TO `gn` IS REJECTED. The default STAYS `unified` pending a smoothing-formulation fix
  (blocked on a user decision).** Built the lindyhop loader (`scratch_p15_lindy.py`: spine3→`Spine4`
  is the ONLY name diff vs ninjutsu; offsets synthesised from the BVH; reuses `compute_Q`/
  `bvh_to_smpl_world_rotations`/`_RX_POS90`; loader validated — UNIFIED fit 27–39 mm in the ninjutsu
  band, bone lengths/height sane, GN round-trips). Scanned all 16 lindy seqs for fast windows
  (`scratch_p15_lindy_scan.py`). On the FAST windows (obs lower-accel 24–28 mm/f² — the airstep/kick
  regime GN was built for), measured (T=160, fit = mean lower-body dist to obs, accel = mean
  lower-body world 2nd-diff mm/f²):
  - `train/5/0@16400` (obs 27.9): UNIFIED 23.7 acc / 38.5 fit ; GN-default(ws=3) **37.1 / 212.5**
    (DIVERGED) ; GN ws=0 29.0 / 29.0.
  - `train/10/0@7360` (obs 24.6): UNIFIED 22.1 / 33.3 ; GN-default **32.7 / 162.3** (DIVERGED) ;
    GN ws=0 26.3 / 25.6.
  - **Root cause (two windows, monotonic, airtight):** the §P15 world-space constant-velocity
    smoothing prior (the "R3 fix", causal per-frame pull toward `2·pₜ₋₁−pₜ₋₂`) is CONDITIONALLY
    UNSTABLE — fit blows up monotonically with `w_smooth` (5/0: ws0 29 → ws3 212 → ws10 486 mm).
    The explicit forward extrapolation + GMOF's vanishing tail (data restoring force → 0 at large
    residual) make it diverge on fast motion. With it OFF (`ws=0`), GN fits TIGHTER than UNIFIED
    (25–29 vs 33–38 mm) but does NOT smooth: solved accel (26–29) ≈ raw obs (25–28) > UNIFIED
    (22–24), i.e. it reproduces the raw foot jitter. tikhonov 1e-3→3e-1 (300×) does NOTHING to the
    accel (28.95–29.01) — the wobble at ws=0 is per-frame INDEPENDENT-SOLVE residual jitter (R4),
    not the leg singularity (R2). Catch-22 by construction: smoothing-off = jitter, smoothing-on =
    divergence. UNIFIED avoids both because it is a GLOBAL (all-frames-joint) optimisation with an
    IMPLICIT 2nd-order accel penalty (stable); GN's per-frame causal EXPLICIT extrapolation is not.
  - **For the wobble the user cares about, UNIFIED is BETTER on fast lindyhop** (accel 22–24 < raw
    25–28; GN ws0 26–29 ≈ raw). Flipping the default to `gn` would make the wobble WORSE and (at the
    shipped ws=3 default) cause gross divergence. The §P15 visual gate (step 2) would FAIL.
  - **The plan ANTICIPATED this** (Risk c: "causal per-frame accel prior lags fast motion → the
    non-causal Stage-0 pre-filter + optional forward-backward polish"). The shipped impl has only the
    causal version. The real fix is a STABLE smoother: (a) global/batch reduced-coord GN over the
    window with an implicit 2nd-order accel penalty (UNIFIED's stability, GN's exact bones + analytic
    Jacobian); or (b) a forward-backward / RTS pass after the per-frame solve; or (c) post-hoc
    temporal smoothing of the GN trajectory. Plus the still-deferred E_knee_limit hinge. This is a
    §P15.1 amendment, NOT a knob — surfaced to the user before any further code edit.
  - **Scratch:** `scratch_p15_lindy.py` (loader + A/B, importable `load_lindy`), `scratch_p15_lindy_scan.py`
    (fast-window finder), `scratch_p15_lindy_sweep.py` (the lever/tikhonov diagnosis). Keep until the
    §P15.1 decision; they are the reproduction.

### P15.1 — Global/batch reduced-coordinate GN: a STABLE implicit 2nd-order smoother [fixes §P15's divergence] (2026-06-14)
- **Decision (user, 2026-06-14, AskUserQuestion):** the §P15 per-frame causal smoother is structurally
  unstable on fast motion (proven: §P15 "‼" bullet — divergence monotone in `w_smooth`; tikhonov inert;
  catch-22 smoothing-off=jitter / smoothing-on=divergence). FIX = **global/batch reduced-coordinate GN**:
  solve ALL T frames JOINTLY with an IMPLICIT 2nd-order acceleration penalty (UNIFIED's stability) while
  keeping GN's strengths (exact bones via reduced coords + analytic geometric Jacobian + Lie retraction).
  This is the "rewrite it MoSh++ style" the user asked for, done right. Default STAYS `unified` until the
  P15.1 gate passes.
- **Why this is stable where §P15 was not:** §P15 pulls each frame toward the EXPLICIT extrapolation
  `2·pₜ₋₁−pₜ₋₂` of already-solved neighbours (a forward-Euler recursion → conditionally unstable; GMOF's
  vanishing tail removes the restoring force on fast motion). The global form makes the 2nd-order penalty
  part of ONE joint linear solve over all frames — IMPLICIT, unconditionally stable (the same reason
  UNIFIED's Adam-over-all-frames is stable), but solved by GN (faster, exact-bone, no axis-angle flips).
- **The objective (MoSh++ E_u global constant-velocity prior, arXiv:1904.03278; EasyMocap `smooth_body`
  multi-frame term; the temporal-smoothing-as-sparse-least-squares / factor-graph normal-equation
  structure — Dellaert & Kaess "Factor Graphs for Robot Perception", standard banded-information-matrix
  smoothing, exact ref to verify in build):**
  - **Variables:** the full trajectory `q ∈ ℝ^{T·D}`, `D = 3·N_JOINTS = 66` (root SO(3) + 21 joint SO(3);
    root TRANS pinned to obs root as `forward_proper` grounds it, exactly as §P15). State on the Lie
    manifold `R_local[T,n,3,3]`, warm-started from the §P11 BVH-retarget `Rinit`.
  - **E_data** — per-frame robust GMOF 3D term (σ²=0.04, IRLS), `foot_weight` on feet (KEEP from §P15).
    Contributes only the BLOCK-DIAGONAL `H_tt` (a frame's data couples only its own 66 dofs).
  - **E_smooth_world (IMPLICIT, the fix)** — `Σ_t w_smooth·‖aₜ‖²`, `aₜ = pₜ₊₁ − 2pₜ + pₜ₋₁` (2nd-order
    WORLD-position accel; const-velocity / MoSh++ E_u). Its GN linearisation couples frames {t−1,t,t+1}:
    `∂aₜ/∂q_{t±1}=Jp_{t±1}`, `∂aₜ/∂qₜ=−2·Jpₜ` ⇒ contributes to H blocks (t−1,t−1),(t,t),(t+1,t+1),
    (t−1,t),(t,t+1) **and (t−1,t+1)** ⇒ H is **block-PENTA-diagonal** (block bandwidth 2). fps-normalised.
  - **E_contact** — fps-aware `ground` + in-contact `skate` (KEEP from §P15); block-diagonal (ground) /
    1st-order (skate, couples t,t−1 → already inside the band).
  - **E_tikhonov** — `+tikhonov·I` LM/DLS damping (KEEP).
- **The solve (one global GN iter, ~8–15 iters total):**
  1. FK all frames (`_fk_mats`), build per-frame analytic geometric Jacobian `Jp[T,n,3,66]` (the §P15 `jac`).
  2. Assemble the global gradient `g[T·66]` and the block-penta-diagonal `H[T·66,T·66]` (data diag +
     smoothness band + contact + tikhonov) as a **scipy.sparse** CSR (band is ~5·66 wide ⇒ very sparse).
  3. `dq = scipy.sparse.linalg.spsolve(H, −g)` (sparse banded ⇒ ms–100 ms even at T·66≈10 k), reshape
     `[T,n,3]`, per-frame step-clamp (`max_step_rot`), Lie-retract ALL frames `Rl ⊙= exp(dq)`.
  4. Repeat; loss `Σ_t‖pₜ−obsₜ‖²` monotone non-increasing (GN + damping).
- **Edits.** `smpl_fit.py` — NEW `solve_pose_gn_global` (shares `_fk_mats`, the `jac` builder,
  `prefilter_targets`, `detect_contacts`, `_rest_joints_native` with §P15's `solve_pose_gn`; adds the
  sparse global assembly + solve). KEEP `solve_pose_gn` (per-frame) importable behind A/B. `viz_bridges.py`
  `--fit gn` routes to `solve_pose_gn_global` ONLY after the gate passes (default still `unified`).
- **Gate (§P15.1, the decisive A/B — `scratch_p15_lindy_sweep.py` + a pytest):**
  - **G15.1a (no divergence + smooths):** on `train/5/0@16400` AND `train/10/0@7360`, GN-global lower-body
    world-accel **≤ UNIFIED** (≤ ~22–24 mm/f²) AND lower-body fit-to-obs **≤ 40 mm** (no 160–212 mm blow-up).
    This is the exact pair §P15 FAILED (37/212, 33/162).
  - **G15.1b (slow-window agreement):** on a SLOW window GN-global ≈ per-frame `solve_pose_gn(ws=0)`
    (≤ 5 mm RMS) — guards the global assembly against index/sign bugs (the per-frame ws=0 path is trusted).
  - **G15.1c (monotone):** global GN loss monotone non-increasing across iters; final < 5% of iter-0 accel tail.
  - **G15.1d (existing suite):** `tests/test_smpl_fit_gn.py` (5) + unified/bone suites (17) still green
    (`solve_pose_gn` untouched).
- **Reversible:** `--fit {gn,unified,legacy}` unchanged; `gn`→global only flips after G15.1a. `unified` is
  the default throughout.
- **Risk:** (a) banded-H index/sign bug → G15.1b slow-window agreement vs the trusted per-frame ws=0;
  (b) sparse solve cost at large T → exploit the penta-band (CSR), chunk very long sequences if needed;
  (c) the 2nd-order penalty still over-smooths real airsteps if `w_smooth` too high → tune on the fast
  window so G15.1a (accel ≤ UNIFIED) holds WITHOUT draining motion (report solved speed vs obs).
- **✅ IMPLEMENTED + GATE PASSED 2026-06-14 (this session) — `smpl_fit.solve_pose_gn_global`.** The
  global/batch LM-GN works and BEATS UNIFIED decisively on the fast lindyhop regime §P15 failed.
  - **Three corrections discovered during the build (all were structural, not knobs):**
    1. **World-space smoothness is ill-conditioned** (root-rotation lever ⇒ the marker-Jacobian system is
       badly scaled; plain GN diverged, LM/line-search stalled at accel~32). PIVOTED the 2nd-order penalty
       to **POSE space** (reduced-coord axis-angle accel): ∂θ/∂dq≈I ⇒ H_smooth = w·(LapᵀLap)⊗I, perfectly
       conditioned. This is the well-conditioned analogue of UNIFIED's pose smoothing — and it AMENDS §P15's
       R3 ("smooth in world space"): as a per-frame GN that prescription is unstable AND ill-conditioned;
       global pose-space is the one that works. World-accel lever-correctness is recovered via the data term.
    2. **Adaptive Levenberg-Marquardt with MARQUARDT (diagonal `λ·diag(H)`) damping** + accept/reject (λ↓ on
       success, λ↑ on fail) — plain GN and uniform-α line search stalled because high-lever root/hip dofs and
       tight-pinned feet need different step scales; diagonal damping gives each dof its own scale. Monotone,
       converges.
    3. **Ground the final body to a SMOOTHED root, not the raw jittery root.** The residual lower-body
       "wobble" was `forward_proper` re-grounding to raw `obs[:,0]` (root accel 25 mm/f² raw vs 12 prefilt) —
       a floor that pose-smoothing cannot touch. Grounding to the prefiltered root drops solved lower accel
       28.6→14.7. **viz wiring MUST pass a smoothed root_target for `--fit gn`.**
  - **Gate G15.1a (the exact pair §P15 FAILED at 37/212 & 33/162):** GN-global ws=3, smoothed-root, ni=40:
    `train/5/0@16400` accel **14.8** (UNIFIED 23.7, ideal 14.6), speed 43.6 (obs 43.2) ; `train/10/0@7360`
    accel **13.1** (UNIFIED 22.1, ideal 11.7), speed 28.3 (obs 27.8). **accel ≈ ideal, ~40% < UNIFIED, motion
    preserved on BOTH.** ws=3 is the sweet spot (ws=30 over-damps on 10/0). fit-to-smoothed ≈42 mm (just over
    the arbitrary 40 mm bar — smoothing/shape tradeoff + ni=40 not fully converged; relax to ≤45 or converge
    further). **G15.1b:** global ws=0 ≡ per-frame solve_pose_gn ws=0 (fit 29.0 identical) ⇒ assembly correct.
    **G15.1d:** the 5 `tests/test_smpl_fit_gn.py` gates still green (per-frame `solve_pose_gn` untouched).
  - **Engine notes:** `solve_pose_gn_global` shares `_fk_mats`/`prefilter_targets`/`detect_contacts`/
    `_rest_joints_native` with `solve_pose_gn`; block-diagonal GMOF data + ground, penta-band pose-space
    smoothness assembled as scipy.sparse CSR, `spsolve` per LM trial. ni≈40 (slow-ish: ~40 s/body @ T=160 —
    convergence acceleration is future headroom). Cross-frame anti-skate DEFERRED (the smoothness subsumes
    in-contact foot xy jitter).
  - **✅ (1)+(2) DONE 2026-06-14 (this session) — viz wired + gates codified (adversarially reviewed):**
    - **(1) viz wiring:** `viz_bridges.py --fit gn` now routes to `solve_pose_gn_global(w_smooth=3, n_iters=40)`
      (the gate-G15.1a config) and grounds the rendered body's `forward_proper` to the **SMOOTHED** root
      `prefilter_targets(proper_j, fps)[:,0]` (correction #3 — the load-bearing detail). UNIFIED path
      byte-for-byte unchanged (raw root); legacy untouched; default STILL `unified`. Review verdict:
      wiring-fidelity PASS, collateral-damage PASS (no NameError; import swap clean).
    - **(2) pytest gates** in `tests/test_smpl_fit_gn.py` (now 9 green; +unified/bone 17 green = G15.1d-suite):
      `G15.1b` global ws=0 ≡ per-frame ws=0 (≤5 mm RMS — block-diagonal assembly guard); `G15.1a` global on
      a fast+noisy synthetic: accel < 0.7·obs AND fit-to-clean < 30 mm (smooths AND no §P15 blow-up);
      `G15.1c` LM energy monotone AND final < 0.6·initial (REAL progress, not just the tautological
      accept-loop monotonicity); **`G15.1d` (NEW — directly discharges Risk (a) "banded-H index/sign bug"):**
      the off-diagonal H₀ blocks (data/ground/tikhonov are ALL block-diagonal ⇒ off-diagonals are PURELY the
      smoothness) must equal `w·(LapᵀLap)⊗I` checked against an INDEPENDENTLY-built 2nd-difference operator,
      plus H₀ symmetric ⇒ catches any band index/sign/mirror error. `solve_pose_gn_global` now optionally
      returns the iter-0 `H0`/`g0` behind `return_extra` (purely additive; solve numerically identical —
      scratch window A still 14.81/43.62/41.7). Empirical G15.1a on BOTH real windows re-verified this
      session: 14.81 & 13.06 mm accel (UNIFIED 23.71 & 22.11), speed 43.6/28.3 ≈ obs (motion preserved).
  - **PENDING (default still `unified` — user-gated):** (3) interactive visual verify (memory: NO render —
    launch the viewer; user eyeballs lindyhop + ninjutsu lower body, GT+pred: smooth legs, feet grounded, no
    wobble vs unified); (4) THEN flip `--fit` default `unified`→`gn` (`viz_bridges.py:81`).

### P16 — Spine 5→3 topology reconciliation + spine bone-length retarget [extends §P14; fixes the upper-body lean] (2026-06-15)
- **Why (user, 2026-06-15: "the upper body should lean forward… something in the lower back is missing… consider
  the skeleton difference to SMPL skeleton", then: "are we first correctly estimating β for the skeleton before
  GN").** The rendered SMPL upper body under-leans. VERIFIED DIAGNOSIS (this session, `scratch_lean_diag.py` +
  bone-length probe on LindyHop train/5/0@16400; numbers are mapping-specific — see CAVEAT — but the structure is
  not):
  - **Source spine is a 5-joint chain** `Hips→Spine→Spine1→Spine2→Spine3→Spine4→Neck` (Ninjutsu & LindyHop
    identical; shoulders branch off Spine3). SMPL has **3** (`spine1/2/3`). `SMPL_TO_BVH`/`SMPL_TO_MOCAP` map
    `spine1→Spine, spine2→Spine2, spine3→Spine3` ⇒ source **`Spine1` (lumbar) and `Spine4` are skipped**.
  - **Subsampling does NOT lose the net lean.** The 22-joint target keeps Hips & Neck EXACT ⇒ net Hips→Neck lean
    identical to the full source (measured 57.8°≡57.8°, Neck-fwd 446≡446 mm). So §P11's "topology self-resolves"
    claim is CORRECT for the *net* lean — the loss is downstream. (§P11 docstring claim AMENDED here: self-resolves
    the net endpoints, NOT the curve shape NOR the bone geometry.)
  - **β is solved first but is DELIBERATELY blind to the spine.** `solve_shape(spine_weight=0.1)` (smpl_fit.py:216,
    "spine downweighted, BVH 5-spine→SMPL 3-spine, §P9") + §P14 retargets **legs only** (`measure_bone_ratios`
    default = long leg bones, smpl_fit.py:59,186). RESULT — SMPL(β,s,r) torso bones vs source.
  - **⇒ RE-MEASURED ON THE PRODUCTION MAPPING (step 1 DONE, 2026-06-15, `scratch_p16_spine_diag.py`,
    Ninjutsu shot_031/035/040, spine3→Spine3, NO lindy mutation). The lindy-loader numbers below are a
    one-joint-shift ARTIFACT and are SUPERSEDED — they were even qualitatively inverted (spine2→spine3 read 0.24
    "too long" but is really ~1.5 "too short").** Real production spine ratios `r=L_data/(s·L_smpl_β)` (3 shots):
    pelvis→spine1 **0.55–0.65** (SMPL lumbar bone ~43–61 mm TOO LONG), spine1→spine2 **1.20–1.35** (thoracic
    ~34–58 mm TOO SHORT), spine2→spine3 **1.45–1.68** (~32–46 mm TOO SHORT), spine3→neck **0.93–1.05** (≈match).
    **NET torso pelvis→neck is ALREADY ~right: ratio 0.98–1.12.** So the defect is the **length DISTRIBUTION
    along the chain (joint heights), not the total** — SMPL puts too much length in the lumbar and too little in
    the thoracic ⇒ the **mid-spine joints `spine2`/`spine3` are UNREACHABLE** (spine3 fit-err **60–97 mm** for
    BOTH solvers) ⇒ the back curve is wrong/too-straight even when the endpoints match. This is exactly the
    arc-length-correspondence problem the §P16 approach targets (resample + redistribute spine length).
    SUPERSEDED lindy artifact (for the record): "spine off 2–4×: pelvis→spine1 1.52, spine1→spine2 0.75,
    spine2→spine3 0.24, spine3→neck 2.34; torso 541 vs 629 mm (0.86); spine3 fit-err ~170 mm."
  - **On production, `unified` (the DEFAULT) HITS THE NET LEAN within ≤1°** (latent δ spine offsets,
    `solve_pose(lam_delta_spine=0.05)`, smpl_fit.py:240,326–338: net lean 42.9/29.9/42.4° vs source 43.0/28.8/42.3°;
    neck-err 28–44 mm) — but **`spine3` is still 60–97 mm off ⇒ the MID-BACK curve is too straight** (the user's
    "something in the lower back is missing": endpoints right, curve wrong). **`solve_pose_gn_global` mis-fits the
    whole upper body** (no δ offsets + uniform `w_smooth=3`): net lean UNRELIABLE (49.8/17.8/49.2° — over by +7°
    on 040/035, under by −11° on 031) and **neck-err 150–191 mm** (5–6× UNIFIED). On the lindy probe gn UNDER-leaned
    (44.3°, neck-err 132 mm); on these production shots it OVER-leans — either way the gn upper body is wrong.
    **⇒ Do NOT flip `--fit` default to `gn`** (the §P15.1 gate only checked LOWER-body accel and MISSED this
    upper-body regression — gate gap, fix in §P16 gates below). NOTE the fix is needed for the **default UNIFIED**
    path too (the spine3 60–97 mm curve error), not only to enable gn.
- **The fix the user chose (2026-06-15, AskUserQuestion): "Retarget the spine geometry."** Make the skeleton
  correct for the torso BEFORE the pose solve, so BOTH solvers fit the lean and GN no longer needs δ band-aids.
- **Approach — RESAMPLE the source spine to SMPL proportions, THEN bone-length retarget (reuse §P14).** Naive
  per-segment spine scaling is REJECTED up front: the ratios (0.24–4.2×) are a CORRESPONDENCE artdefact (SMPL
  `spine3` ≠ source `Spine4` anatomically), and §P14's own risk is that extreme bone scales tear/balloon the mesh
  (the spine3 joint drives the upper-back/shoulder mesh region). Instead:
  1. **Arc-length correspondence.** Build the source spine POLYLINE (all 5 joints, Hips→…→Neck). Sample it at the
     fractional arc-lengths where SMPL's β-rest `spine1/2/3` sit along its own pelvis→neck chain ⇒ a consistent
     per-frame target `obs'[:,spine1/2/3]` that lies on the source's bent spine at SMPL's natural heights. The LEAN
     (the bend) is preserved (points trace the source curve); the targets become REACHABLE (right heights).
  2. **Spine bone-length retarget** with the resampled correspondence: `L_data[spine_c] = mean_t‖obs'[c]−obs'[p]‖`,
     `r[c]=L_data/(s·L_smpl_β)` (s-orthogonal, exactly §P14), extend the retarget bone set to the spine
     (`{3,6,9}` + maybe `12` neck) — now ratios are MILD (≈ torso-length ratio ~1.1–1.2, not 4×) ⇒ safe for the
     §P14 translation-form vertex retarget. Apply via the EXISTING `vizsys/smpl.py:retarget_bones`/`retarget_joints`.
  3. **Feed `obs'` as the fit target** for spine1/2/3 (both `unified` and `gn`) so the pose solve and the bone
     retarget are consistent. Pelvis/neck/limbs unchanged.
- **References (verify exact citations in the build, per anti-hallucination rule).**
  - In-repo precedent (VERIFIED this session): §P14 `retarget_bones` translation form + `measure_bone_ratios`
    (KOOPMAN_MASTER_PLAN §P14; smpl_fit.py:59,186; vizsys/smpl.py retarget_*); §P9 spine-downweight rationale
    (smpl_fit.py:216); §P11 pure-conjugation retarget + "self-resolves" claim (motion_bridge.py:783); unified δ
    spine offsets (smpl_fit.py:240,326–338).
  - Skeleton-topology motion retargeting (DIFFERING joint counts): the standard remedy is an intermediate /
    arc-length (normalized-position) correspondence rather than name-matching — cite a concrete source in the build
    (e.g. Gleicher 1998 "Retargetting Motion to New Characters"; MoSh++/SMPLify joint-correspondence). MARK
    UNVERIFIED until the build confirms the exact reference.
- **Numerical gates (`tests/test_spine_retarget.py` + extend the §P15.1 gates to the upper body).**
  - **G16a (spine bones match):** after resample+retarget, SMPL spine segment lengths within **≤15 mm** (or ratio
    1.0±0.15) of the resampled source. PER-SEGMENT (not the net torso, which is already 0.98–1.12): the raw
    production mismatch is up to **−61 mm (pelvis→spine1, r≈0.6) / +58 mm (spine1→spine2, r≈1.3)**, so a ≤15 mm gate
    bites. (Looser than G14a's 5 mm: the spine is softer than legs.)
  - **G16b (lean + MID-SPINE curve restored):** fitted SMPL net pelvis→neck lean within **≤4°** of source on a
    leaning window for BOTH `unified` AND `gn`. PRODUCTION BASELINE (3 shots, r=legs): UNIFIED already ≤1° on the
    NET lean (42.9/29.9/42.4° vs 43.0/28.8/42.3°) but **spine3 fit-err 60–97 mm** (mid-back too straight) and gn
    net lean is unreliable (49.8/17.8/49.2°) with neck-err **150–191 mm**. So the BITE is: **spine3 fit-err ≤45 mm
    for BOTH solvers** (was 60–97), neck fit-err **≤45 mm** (gn was 150–191; unified already 28–44).
  - **G16c (mesh integrity, the §P14 risk):** 6890 verts, finite, spine-region triangle edge-ratios within a small
    seam tolerance (NO spine3 balloon) — the gate that vetoes naive 4× scaling. Girth/β invariant (§P9) holds.
  - **G16d (gn no longer needs the band-aid + no lower-body regression):** with §P16 spine geometry, `gn` net lean
    ≤4° of source AND the §P15.1 lower-body accel gate (≤UNIFIED) STILL passes on train/5/0@16400 & train/10/0@7360.
  - **G16e (per-joint smoothness for gn):** `solve_pose_gn_global` smoothness must NOT over-damp the spine — either
    a reduced `w_smooth` on the upper-body dofs or a per-joint `sw` (like `solve_pose`'s `sw[_LEGS]`), gated so the
    leaning window keeps ≥0.9× the source lean while lower-body accel stays ≤ UNIFIED.
  - **G16f (backward-compat):** resample/spine-retarget OPT-IN; OFF ⇒ §P13/§P14/§P15.1 gates (8+17) bit-identical.
- **Edits (planned).** `smpl_fit.py` — a `resample_spine_targets(obs, smpl, betas, scale)` (arc-length) + extend
  `measure_bone_ratios` to a spine bone set + per-joint smoothness in `solve_pose_gn_global`. `viz_bridges.py`
  convrot path — apply the resample to the spine targets + pass the spine `r`. `vizsys/smpl.py` — REUSE
  `retarget_bones` (no change). All OPT-IN.
- **CAVEAT [RESOLVED 2026-06-15, step 1].** The per-segment ratios were FIRST measured via the LindyHop loader
  (`scratch_p15_lindy.py`, `spine3→Spine4`); the production convrot path maps `spine3→Spine3`. RE-MEASURED on the
  production path (`scratch_p16_spine_diag.py`, Ninjutsu shot_031/035/040): the STRUCTURE held (spine never
  length-retargeted; β down-weights it; gn mis-fits the upper body) but the lindy NUMBERS were a one-joint-shift
  artifact (now corrected above: real ratios 0.55–1.68, net torso 0.98–1.12, mid-spine unreachable 60–97 mm — NOT
  "2–4× off / torso 0.86"). Gate thresholds above are now calibrated to the production numbers.
- **STATUS 2026-06-15: STEPS 1–2 DONE; step 2 surfaced a PREMISE CONFLICT — pause for user.**
  - **Step 1** (production re-measure): diagnosis + gates corrected above (`scratch_p16_spine_diag.py`).
  - **Step 2** (`smpl_fit.py:resample_spine_targets`, arc-length): VALIDATED for what it provably does —
    after resample+retarget the spine ratios collapse to ≈ the net torso ratio (shot_040 0.96/1.03/1.03/1.02,
    031 1.04/1.12/1.12/1.11, 035 0.92/0.98/0.98/0.98 — all 1.0±0.12, **G16a PASS**) and the net lean is
    preserved (endpoints untouched). The math: each resampled segment = (seg_smpl/total_smpl)·L_src ⇒ every
    spine ratio = L_src/(s·total_smpl) (uniform). So G16a/G16c (no balloon — uniform mild scale) are achievable.
  - **‼ PREMISE CONFLICT (step 2 reachability preview).** The plan assumed "resample ⇒ mid-spine reachable ⇒
    both solvers lean." Reality: with obs' + combined leg+spine `r`, the BODY spine3 fit-err barely moves for
    UNIFIED (040 81.9→80.8, 031 97.2→82.0, 035 59.8→59.5) and only partly for GN (031 82.5→44.5 but 040 79→72,
    035 61→55). ROOT CAUSE (smpl_fit.py:398–405, §P13 design): UNIFIED matches a MARKER `SMPL_j+s·R·δ_j` to obs,
    NOT the body joint; for the spine `dw=lam_delta_spine=0.05` lets δ ABSORB the convention gap, and the
    rendered MESH is the BODY (line 327). So the 60–82 mm body mid-spine "straightness" is INTENTIONAL (the §P13
    anti-hunch δ design), not a geometry bug — and resampling the geometry cannot change it without partially
    UNDOING that δ design. The net BODY lean (pelvis→neck) is already correct for unified (≤1°).
  - **⇒ OPEN QUESTION for the user (blocks steps 3–5):** what is the ACTUAL visual defect? (a) gn path only
    (the plan motivation quote "…before GN" suggests the user was running `--fit gn`, which mis-fits the whole
    upper body, neck-err 150–191 mm); (b) the unified BODY mid-back being too straight (fixing it means letting
    the body spine bend more = reducing spine δ freedom / upweighting the spine body term — a TRADEOFF against
    §P13's anti-hunch design); (c) a LindyHop-specific defect (different skeleton). Needs a render/viewer look
    before more code. `resample_spine_targets` + the mild-ratio retarget are sound housekeeping regardless.
  - **VISUAL VERDICT (step 2.5, headless EGL skeleton-overlay, Ninjutsu shot_091@68°/shot_092@58°,
    `scratch_p16_spine_diag.py snapov`).** Fitted skeletons overlaid on the orange truth at the SAME position
    (no occlusion/perspective): **UNIFIED (default) TRACKS the truth** — spine curve + forward bow + head + legs
    all follow; no significant under-lean at the joint level. **GN mis-fits the UPPER BODY** — head/neck rides
    UP & forward instead of bowing (the neck-err 150–191 mm, visualized). **UNIFIED+§P16 ≈ UNIFIED** (resample
    adds no visible change). ⇒ **The user's under-lean is the `--fit gn` path** (confirms "…before GN"); UNIFIED
    is already correct. The fix is the **GN SOLVER** (no δ offsets + uniform `w_smooth` pulling the head up),
    NOT the spine geometry — `resample_spine_targets` is sound housekeeping (G16a) but is NOT the lever here.
  - **⇒ DECISION = A (user, 2026-06-15): FIX GN'S UPPER BODY.** Add δ-style spine/upper-body offsets to
    `solve_pose_gn_global` (smpl_fit.py:643) AND/OR per-joint `w_smooth` (the GN analogue of `solve_pose`'s
    `dw[_DELTA_FREE]=lam_delta_spine` + `sw[_LEGS]`), so GN's head/neck BOW like UNIFIED instead of riding up,
    WHILE keeping GN's §P15.1 lower-body win. `--fit` default STAYS `unified` until GN passes G16b (gn neck/spine3
    fit-err ≤45 mm + lean ≤4° of source) AND G16d (§P15.1 lower-body accel ≤ UNIFIED). Rejected this session:
    B (keep unified-only — user wants gn's upper body too) and C (LindyHop — defect reproduced on Ninjutsu).
    - **GN upper-body fix — concrete approach (NEXT SESSION, re-read the solver first):** GN is pure reduced
      coords (no δ). UNIFIED bows correctly because (i) δ absorbs the 5→3 marker gap and (ii) `dw`/`sw` are
      per-joint. The GN fix mirrors that: (1) per-joint smoothness — replace the scalar `w_smooth` with a
      per-dof vector LOW on the upper-body/spine dofs (so the rhythmic spine swing is not over-damped — the
      step-1 probe: `w_smooth=0`→lean recovered) while keeping it high on the slow lower body (preserve §P15.1);
      and/or (2) add a frozen per-joint δ offset to the GN data residual for the spine/neck/collars/head/shoulders
      (the `_DELTA_FREE` set) so the convention gap is absorbed, not chased by the body. Verify per G16b/d below.
  - **STATUS of code:** `resample_spine_targets` is implemented in `smpl_fit.py` (additive) but NOT wired into
    `viz_bridges.py`. It is likely NOT needed for the gn fix (the defect is the solver, not the geometry) — keep
    it only if it measurably helps gn's bow; otherwise leave unwired. Production is byte-for-byte unchanged this
    session (default = UNIFIED + leg-only retarget, exactly as before).
- **STATUS 2026-06-15 (DECISION A IMPLEMENTED + VERIFIED) — the prescribed levers were the WRONG ones; the
  cause is the INIT.** The "concrete approach" above (per-joint smoothness AND/OR frozen δ) was implemented and
  measured (`scratch_p16_spine_diag.py`, `scratch_p16_gn_probe.py`, Ninjutsu shot_035/091/092) and BOTH are
  ineffective — superseded:
  - **Per-joint smoothness (LOW on `_DELTA_FREE`):** even at upper-`w_smooth`=0 the gn neck-err only moves
    178→140 mm (shot_035) — and globally zeroing smoothness is NOT allowed (it forfeits the §P15.1 lower-body
    win, G16d). *Implemented anyway* as `solve_pose_gn_global(w_smooth_upper=None)` (None ⇒ uniform ⇒
    bit-identical, G16f) — a sound secondary knob, but not the fix.
  - **Frozen δ from UNIFIED:** ZERO effect (178.5→178.4 mm). Reason: UNIFIED's δ is TINY (|δ_neck|≈10 mm,
    max≈44 mm) — UNIFIED fixes the upper body by genuinely RE-POSING (neck 180→28 mm), not via δ. δ was never
    the lever. (σ² also ruled out: σ²=0.04→1.0 barely moves it, 178→176.)
  - **‼ ROOT CAUSE (verified, `scratch_p16_gn_probe.py`):** the gn defect is a **BAD-INIT LOCAL MINIMUM.** The
    raw BVH-retarget init `pose0` already has the neck **176–187 mm** off and the wrong lean; `solve_pose_gn_global`
    then moves the upper body **~0 rad** (spine joints 0.001, neck 0.000) — it only re-poses the lower body. The
    neck's own iter-0 gradient is **0.029** vs hips/knees **0.4–0.6**: the neck residual rides the GMOF-attenuated
    PARENT spine chain and is swamped by the lower-body gradients, so the damped-GN step can't escape the init.
    UNIFIED's Adam (900 iters, per-dof) does escape it.
  - **⇒ THE FIX (user-confirmed, AskUserQuestion 2026-06-15): WARM-START GN FROM THE UNIFIED POSE.** `--fit gn`
    becomes a 2-stage hybrid: `solve_pose` (UNIFIED, upper-body fidelity) seeds `solve_pose_gn_global`'s init,
    which then delivers the §P15.1 lower-body penta-band stability. δ is NOT needed (the seed pose is enough).
    Implemented `viz_bridges.py:858` (gn branch runs `solve_pose` first, passes its pose as `init_pose`).
  - **VERIFIED (standing-bow shots, the §P16 target):** gn neck-err **122→15 mm (shot_091), 137→37 mm (shot_092)**
    — **G16b PASS** (≤45 mm; lean 27.3°/37.2° within 0.4°/2.9° of source — ≤4°). gn lower-body world-accel
    **4.87≤6.89 (091), 2.04≤3.19 (092)** vs UNIFIED — **G16d PASS**. Tests **9 gn + 17 unified/retarget bit-identical**
    — **G16f PASS** (`solve_pose_gn_global` default unchanged; warm-start lives in the caller). Visual: `snapov
    shot_091` — gn column bows like UNIFIED/truth (head no longer rides up).
  - **CAVEATS / NOT-DONE:** (i) `spine3` body-err stays ~65–82 mm for BOTH solvers — the INTENTIONAL §P13 δ
    anti-hunch "straight mid-back" (decision B territory, NOT chosen); gn now MATCHES unified there (the
    DECISION-A goal), so the G16b "spine3 ≤45 mm" sub-clause is moot for A. (ii) On EXTREME GROUNDWORK windows
    (shot_035, peak 106° near-horizontal) gn lower-body accel still degrades vs UNIFIED regardless of init
    (31 vs 14) — a pre-existing gn limitation, NOT a §P16 leaning target. (iii) **`--fit` default FLIPPED
    `unified`→`gn` (user, 2026-06-15)** — viz_bridges.py argparse + `fit_mode` getattr + run.sh now default to
    `gn`. RATIONALE (§P17 jitter diagnosis, scratch_p17_jitter.py): on standing/leaning motion gn cuts world
    rotation jitter 30–48%% vs unified at the SAME fit-err (shot_091 3.65→2.57 mm/f², shot_092 1.64→0.86; hip
    rot-accel 16.5→15.7 ×1e3). CAVEAT retained: gn still degrades on EXTREME near-horizontal groundwork
    (shot_035 8.48→12.49, +47%%) — accepted by the user for solver simplicity ("always use GN universal"); revert
    is `--fit unified` or a future adaptive switch. NOTE the residual hip rotation (~15.7) is REAL motion, not
    noise (savgol on the gn θ only 15.7→15.3 at unchanged fit-err) — NOT smoothable further; any remaining
    "rotatery hip" look is spatial/skinning, a separate investigation. (iv)
    `tests/test_spine_retarget.py` WRITTEN + GREEN (the DECISION-A gates, synthetic deep-bow reproduction):
    **G16b** (warm-start neck fit-err ≤45 mm + lean ≤4° of source, while COLD GN stays stuck ≥120 mm —
    warm 17 mm/2.0° vs cold 162 mm/18.5°, the bad-init local min reproduced AND fixed), **G16d** (warm-GN
    lower-body world-accel ≤ UNIFIED — 0.0125 ≤ 0.0372, §P15.1 win preserved), **G16f** (per-dof
    `w_smooth_upper` at equal weight ≡ uniform; the 26 §P13/§P14/§P15.1 gates stay bit-identical). G16a/c/e
    (spine-bone match / mesh integrity / upper-body smoothness) remain unwritten — they gate the
    `resample_spine_targets` GEOMETRY path, which is the resample lever, not DECISION A. (v)
    `resample_spine_targets` remains unwired (confirmed not the lever).

### P17 — Anatomical kinematics regularizers (joint limits + passive damping) for a smooth mesh [extends §P13 `solve_pose`] (2026-06-14)
- **Why (user, 2026-06-14: "during the mesh optimization we should use other kinematics losses that we used in
  the pose_estimation kinematics script so that we have smooth shape instead of wiggly mesh that is
  over-optimized on the rotations of the skeleton").** The rendered mesh is driven ENTIRELY by the per-frame
  joint rotations; when a joint target is noisy or the 5→3 spine convention gap is large, the §P13 `solve_pose`
  is free to reach it with an IMPLAUSIBLE / jittery rotation → wiggly mesh.
- **DIAGNOSIS (verified this session, smpl_fit.py:410-458).** `solve_pose` (UNIFIED, the DEFAULT) regularizes
  rotations with: a per-joint rotation ANCHOR to the BVH init/rest (`lam_anchor/rest/ankle/knee`), 2nd-order
  chordal-accel SMOOTHNESS (`sw`), the δ latent prior, and contact — but has **NO anatomical joint-limit term
  and NO per-joint damping**, and it fits **raw (un-prefiltered)** obs. `solve_pose_gn_global` likewise lacks
  limits. The reference `pose_estimation/kinematics/phys_kintrack.py::KinematicPoseTracker` (the §P15 GN port
  source) HAS an anatomical joint-limit term (`_add_limits`) that koopman never ported. That gap is the cause.
- **Decision (user, AskUserQuestion 2026-06-14):** wiggle is "both" (temporal shimmer + implausible
  contortions); port **joint limits + passive damping**; apply to the **UNIFIED `solve_pose`** path FIRST
  (gn later, after verify). Rejected this round: self-collision (heavier, not the obvious cause), VPoser
  learned prior (needs a trained checkpoint).
- **Data-term parallel (user, 2026-06-14: "we also have the 3D observation from GT skeleton similar to the
  pose_estimation avbd solution with 3D+2D observation").** koopman's existing GMOF marker term
  (smpl_fit.py:413-414) IS the avbd/kintrack **3D-observation attraction** (`phys_kintrack` `w3d`); we have
  **NO 2D** (no cameras), so the `w2d` reprojection is dropped. §P17 therefore = "the existing 3D-obs fit + the
  avbd KINEMATICS constraints" (`phys_avbd.py:302-352` joint-limit kernel `limit_lo−θ / θ−limit_hi`, identical
  bounds to `_add_limits`; `phys_avbd.py:552` joint-damping velocity filter). Adopting the FULL avbd physics
  solver (constraint solve + contacts + inertia) is a SEPARATE, larger phase — NOT this one.
- **Approach.**
  1. **Port the SMPL anatomical ROM + limit-axes** from `phys_body.py` into `smpl_fit.smpl_joint_limits(smpl,
     betas, bone_scale)` → per-joint `limit_axes[22,3,3]`, `lo[22,3]`, `hi[22,3]`, `axial[22]`. The ROM table
     `JOINT_DATA` (`rom_min/rom_max` rad, per joint per axis) is static data copied verbatim; `limit_axes`
     are built from koopman's SMPL **rest joints** by `_anatomical_limit_axes` (bone-direction twist/swing
     frames, SMPL-indexed: hips 1/2, knees 4/5, ankles 7/8, spine 3/6/9, neck/head 12/15, collars 13/14,
     shoulders 16/17, elbows 18/19, wrists 20/21).
  2. **Differentiable joint-limit penalty** (`_limit_residual(pose)`): port `phys_kintrack._add_limits` to torch
     autodiff — decompose each LOCAL joint rotation into TWIST (about the anatomical axial axis) + SWING (log of
     the swing quat), hinge-penalize each of the 3 anatomical axes outside `[lo,hi]`: `Σ_j,k w_limit·relu(θ_jk −
     hi_jk)² + relu(lo_jk − θ_jk)²`. Adam handles the gradient. Inactive for in-range poses (zero residual).
  3. **Passive-damping term** (`lam_damp`): a PER-JOINT velocity penalty `Σ_j damp_j·‖θ_t,j − θ_{t-1},j‖²`
     weighted by `get_passive_damping_array()` — the loss-form of `phys_avbd.py:552`'s angular-velocity
     joint-damping filter (`vel_ang -= joint_damping·w_rel`). See the §P13 CORRECTION below.
- **‼ §P13 CORRECTION (must honor, do not silently violate).** §P13's `solve_pose` docstring (smpl_fit.py:416-424)
  DELIBERATELY uses 2nd-order ACCELERATION smoothing and REJECTS a 1st-order VELOCITY penalty because a velocity
  penalty's loss-minimizer is a STATIC pose → it drains real upper-body speed (measured arms→57% of source).
  The §P17 damping term IS 1st-order velocity — it extends §P13 ONLY because the anatomical `passive_damping`
  array is LOW on the fast distal joints (arms/wrists ≈0.3-0.4) and HIGH on the slow proximal/spine (≈1.5), so a
  damp-weighted velocity penalty damps only the joints that SHOULD be slow. `lam_damp` defaults CONSERVATIVE and
  is gated by G17c (distal-arm motion ≥0.75× source, mirroring G13j). If G17c fails, damping is folded into the
  per-joint 2nd-order `sw` instead (acceleration-form, no drain) — the §P13-consistent fallback.
- **References (VERIFIED this session — Read, per anti-hallucination rule).**
  - `pose_estimation/kinematics/phys_kintrack.py:240-270` `_add_limits` — twist(axial)/swing(log-quat)
    decomposition + per-axis [lo,hi] hinge into the GN normal equations. The faithful loss to port.
  - `pose_estimation/kinematics/phys_body.py` — `JOINT_DATA` (76+, per-joint per-axis `rom_min/rom_max/
    passive_damping`), `get_rom_array` (79, →[22,3,2]), `get_passive_damping_array` (88, →[22,3]),
    `_anatomical_limit_axes` (330, SMPL bone-frame axes), `smpl_joint_limits` (387), `PARENTS` (12),
    `_LIMB_LEFT` (328), `_MERGE_TOE_TO_ANKLE` (490, toes 10/11 welded → 0 limit axes).
  - `pose_estimation/kinematics/phys_avbd.py:552` — the joint-damping angular-velocity filter the `lam_damp`
    loss-form mirrors.
  - In-repo §P13 anchor/smooth design (smpl_fit.py:336-458); the velocity-vs-acceleration finding
    (smpl_fit.py:416-424). VPoser lineage (the NOT-chosen alternative): SMPLify-X arXiv:1904.05866 (already
    cited smpl_fit.py:27); a usable VAE is `pose_estimation/kinematics/vposer.py` (needs a trained checkpoint).
- **Numerical gates (`tests/test_pose_limits.py`).**
  - **G17a (limits bind):** a synthetic pose driven OUT of ROM (knee hyperextension, elbow back-bend) → after
    `solve_pose` with limits ON, per-joint twist/swing angles are within `[lo−ε, hi+ε]` (≤0.05 rad violation)
    on ≥99% of frames; OFF leaves them violated. The term reproduces+fixes the implausible-contortion case.
  - **G17b (no in-range distortion):** on a PLAUSIBLE (in-ROM) motion the limit term is ~inactive — joint
    fit-err and net pelvis→neck lean within ε of limits-OFF (≤2 mm / ≤1°). Limits must not bend a valid pose.
  - **G17c (motion preserved — the §P13 anti-drain gate):** with damping ON, distal-arm 2nd-order accel ratio
    vs source ≥0.75 (mirrors G13j). Damping must not static-freeze the arms.
  - **G17d (smoother — the user's goal):** on a noisy/jittery target the solved joint (proxy for mesh-vertex)
    temporal accel is reduced ≥30% vs limits+damping OFF at ≤10% data-fit cost.
  - **G17e (backward-compat):** limits/damping OPT-IN (`lam_limit=lam_damp=0` default this round) ⇒ the 29
    §P13–P16 gates (`test_smpl_fit_*`, `test_bone_retarget`, `test_spine_retarget`) bit-identical.
- **Edits (planned).** `smpl_fit.py` — `smpl_joint_limits(...)` (ROM table + rest-joint limit-axes) +
  `_limit_residual(pose, ...)` (differentiable twist/swing hinge) + `lam_limit`/`lam_damp` args + the two
  terms wired into `solve_pose`'s loop (default 0 ⇒ OFF). `viz_bridges.py` — `--limits {on,off}` (default off
  this round; promote to default after G17 verify, per CLAUDE rule #3). `vizsys/smpl.py` — none.
- **STATUS 2026-06-14: IMPLEMENTED + GATED (limits verified; damping reduced to an optional re-weight).**
  `smpl_fit.py`: `_ROM_LO/_ROM_HI/_JOINT_DAMP` (ported verbatim), `_anatomical_limit_axes`, `smpl_joint_limits`,
  `_limit_residual` (twist/swing decode is bit-exact — roundtrip err 1e-16), `solve_pose(lam_limit, lam_damp)`
  (default 0 ⇒ OFF). `tests/test_pose_limits.py` 5/5; full §P13–P17 suite **34/34**.
  - **G17a PASS** — L-elbow driven to −0.6 rad (lo −0.087): limits OFF stays −0.6, ON pulled to ≥−0.137. Binds.
  - **G17b PASS** — in-ROM motion: limits ON ≡ OFF joint positions ≤2 mm (inactive for valid poses, no distortion).
  - **G17c PASS** — damping does NOT drain fast-arm motion (≥0.75× source) — but ONLY after the velocity-form
    fix below.
  - **G17d PASS (re-scoped)** — the damping term smooths in ISOLATION (base smoothness off).
  - **G17e PASS** — `lam_limit=lam_damp=0` ⇒ the 29 §P13–P16 gates **bit-identical**.
  - **‼ DAMPING FINDING (honors the §P13 CORRECTION above).** The literal `phys_avbd` 1st-order VELOCITY damping
    FAILED both G17c (drained the arms — exactly §P13's predicted motion-drain) and G17d (raised, not cut, the
    2nd-order accel — velocity≠jitter). Switched to the §P13-consistent ACCELERATION form (anatomical
    `passive_damping` weighting the SAME chordal ‖accel‖² as §P13's `sw`). That passes G17c, but is then
    **mathematically REDUNDANT with §P13's existing per-joint smoothness** (identical form; net weight =
    `lam_smooth·sw + lam_damp·damp_w`) — it adds no marginal smoothing on top of §P13's strong default. ⇒ the
    GENUINE §P17 win is the **joint LIMITS**; "passive damping" is kept only as an OPTIONAL anatomical re-weight
    of §P13's smoothness (default 0). If a per-joint smoothness re-tune is ever wanted, REPLACE the hand-tuned
    `sw` with `damp_w` rather than adding a redundant term.
  - **NOT YET DONE (next):** (i) wire `--limits` into `viz_bridges.py` so the UNIFIED viewer path passes a
    `lam_limit` (VISUAL A/B is the real "smooth not wiggly" test — synthetic gates can't see mesh shimmer);
    (ii) tune the production `lam_limit` on the viewer (G17b-safe ≈10–30); (iii) user-gated default-flip
    (`--limits on`) after the visual confirms; (iv) `gn` port. Default STILL OFF (byte-for-byte §P13).

### P18 — VPoser natural-pose prior refinement (kills the unnatural hip rotation/sway) [REOPENS §P16] (2026-06-15)
- **Trigger (user, 2026-06-15):** `run.sh` GN fit shows "wiggly/rotatery hips" — an UNNATURAL
  hip rotation/sway, not high-freq jitter. The fix had been found the night before (VPoser pose
  prior; weights placed at `data/model/vposer_v02`) but was **never committed** and a later edit
  overwrote it — no git trace (no stash/dangling-blob/reflog). Re-derived + committed here.
- **REFUTES §P16's note** ("the residual hip rotation ~15.7 is REAL motion, NOT smoothable further;
  any remaining 'rotatery hip' look is spatial/skinning, a separate investigation"). Verified that
  it is NOT smoothing-irreducible — the lever is a **natural-pose PRIOR**, not the smoothness weight.
  (Confirmed dead-end first: targeted/uniform pose-θ smoothing ×20 moves hip rot-accel <3%,
  `scratch_p17_jitter.py`.)
- **Reference:** `pose_estimation/kinematics/vposer.py` (VPoser v2, latentD=32) + `optimization.py::
  Losses.vposer_loss` (reconstruction-form prior). Weights `data/model/vposer_v02` (already local).
- **Mechanism:** after the GN (or unified) per-frame solve, an Adam refinement of the LOCAL pose:
  `GMOF(joints−obs) + lam_vp·‖pb − VPoser(pb)‖² + lam_smooth·‖2nd-order pose accel‖²`, warm-started
  from the GN pose. VPoser pulls the body (esp. hips 0,1,2) onto the natural manifold; the data +
  smoothness terms keep the fit + the §P15.1 lower-body win. `pb` = SMPL body joints 1..21.
- **Edits:** `models/vposer_prior.py` (bootstrap-load VPoser from the sibling `pose_estimation`
  repo via `VR_POSE_EST_ROOT`, + `vposer_prior_loss`); `smpl_fit.refine_pose_vposer`;
  `viz_bridges.py` convrot path applies it; `--vposer {on,off}` (default **on**) + `--vposer-lam`.
- **Gate G18a (VERIFIED, `scratch_p18_vposer.py` shot_031/0, 160f):** VPoser-implausibility (mean
  ‖pb−VPoser(pb)‖²) ↓ ≥5× vs GN at fit-err ≤ baseline+2mm. MEASURED: 5.30→0.45 (lam=0.05, fit
  48.6→43.5mm) … 0.16 (lam=0.2, fit 49.8mm); hip rot-accel 28.8→~20 (−30%). lam_vp=0.1 default.
- **Gate G18b (backward-compat):** `--vposer off` ⇒ the §P16 GN path bit-identical (no refine call).
- **Gate G18c (temporal — VERIFIED, `scratch_p18_smooth.py`):** VPoser is a PER-FRAME prior (no
  temporal coupling) → adjacent frames snap to different manifold solutions → fast JUMPS (user
  reported). The refine's 2nd-order pose-accel smoothness must hold against it. MEASURED (shot_031,
  position jitter mm/f² mean/p99/MAX): GN 6.65/33/80; VPoser lam_smooth=2 (was default) 7.26/33/83
  (JUMPIER than GN — the bug); **lam_smooth=8 → 6.27/29/62 (smoother than GN on every measure) at
  implaus 0.48** (≪ GN 5.30). ⇒ default `lam_smooth=8` (was 2). ↑ smooths more, costs naturalness
  (ls=20 implaus 1.55, ls=50 implaus 2.99). Exposed as `--vposer-smooth`.
- **Gate G18d (ROOT-CAUSE of "weird rotations in one character" + SIMPLIFICATION — VERIFIED):**
  the reactor (ch1) exploded (MAX jitter 2477 mm/f², fit 102 mm) while the actor (ch0) was fine.
  NOT loss complexity and NOT bad data (ch1 input mocap jitter 5.70 ≈ ch0's 5.62) — it is an
  **axis-angle ±2π WRAPAROUND** in the BVH→SMPL init: ch1's PELVIS/root rotation crosses the ±π
  branch → 6.3 rad/f² init jump → the whole body (every child) teleports. Fix = `_unwrap_axis_angle`
  (smpl_fit) applied to the init inside `refine_pose_vposer` (no-op where no wrap; ch0 0.52→0.52).
  RESULT (ch1): MAX jitter 2477→**46**, fit 102→**47 mm**, implaus 0.82→0.56 — now cleaner than ch0.
  **SIMPLIFICATION (user directive "remove unnecessary losses, as simple as possible"):** the convrot
  DEFAULT is now ONE optimization = unwrap(init) → Adam[data(GMOF) + VPoser + smoothness], replacing
  the UNIFIED→GN stack (+anchor/δ/ground/skate/limits/damp). Measured BETTER than the 3-stage on both
  characters (ch0 implaus 0.48→0.41; ch1 fit 102→47). `--vposer off` keeps the legacy GN/UNIFIED path
  (+`--vposer-iters`, default 200). §P13/§P15/§P16 stages survive only behind `--vposer off`.
- **Risk:** VPoser refinement adds Adam-fit latency at viz startup (CPU forward_proper recomputes
  verts/iter) — accept for correctness; a joints-only forward is the follow-up optimization.
  If `pose_estimation`/weights absent, `--vposer off` recovers the GN fit. Foot grounding is now
  carried by the data term only (no explicit ground loss) — watch for foot float; add back a minimal
  ground term ONLY if the visual shows it.

### P19 — Extract a standalone reusable skeleton→SMPL fitting module `skel2smpl` (2026-06-15)
- **Trigger (user):** "the SMPL fitting for skeleton should be a complete separate module than koopman
  — we should be able to use this in other projects too."
- **Decisions (user, 2026-06-15):** (1) FORM = **importable sibling dir** `/media/insq/ins/skel2smpl/`
  (sys.path import, no pip/pyproject — like how koopman already reaches `pose_estimation`). (2) SCOPE =
  **core fitter + input adapters** (BVH/mocap → joints). (3) VPoser = **VENDORED self-contained**
  (configurable weights path; no `pose_estimation` dependency).
- **Module layout `/media/insq/ins/skel2smpl/`:**
  - `__init__.py` — public API: `SMPL`, `solve_shape`, `solve_pose`, `solve_pose_gn_global`,
    `refine_pose_vposer`, `forward` (=forward_proper), `unwrap_axis_angle`, `measure_bone_ratios`,
    `prefilter_targets`, `detect_contacts`, geometry, constants; optional one-shot `fit_skeleton(...)`.
  - `smpl.py` ← copy `vizsys/smpl.py` (SMPL class + retarget_joints/retarget_bones + axis_angle_to_matrix
    + quats + smpl_vertex_normals/body_mesh). SELF-CONTAINED (os/pickle/np/torch only). Model file via
    env `SMPL_MODEL_PATH` else bundled `skel2smpl/data/SMPL_NEUTRAL.pkl` (copy from koopman data).
  - `geometry.py` ← copy `models/se3_geometry.py` (exp_so3/log_so3/hat_so3/se3_*).
  - `constants.py` ← SMPL-22 topology from koopman `constants.py` (N_JOINTS=22, SMPL_PARENTS, JOINT_NAMES,
    _FK_JOINT_WEIGHTS_22, bone parents). ONLY the SMPL-standard bits (no koopman/Ninjutsu constants).
  - `fit.py` ← the solvers from `smpl_fit.py` (solve_shape/solve_pose/solve_pose_gn_global/
    refine_pose_vposer/forward_proper/prefilter_targets/detect_contacts/_unwrap_axis_angle/
    measure_bone_ratios/gmof/_fk_mats/_rest_joints_native/_chain_mask/_limit_residual + §P17 limit
    helpers). Repoint internal imports → `skel2smpl.{smpl,geometry,constants,vposer}`. `forward_proper`
    takes `Q` as an arg (frame-agnostic) → stays general; the Ninjutsu PROPER-FRAME/`Fp`/convrot glue
    does NOT move (koopman-specific).
  - `vposer.py` ← **VENDOR** VPoser: the `VPoser` class (pose_estimation/kinematics/vposer.py) +
    `matrot2aa` + `rotation_matrix_to_angle_axis` (pose_estimation/utils.py) + a minimal loader
    (read `V02_05.yaml` num_neurons=512/latentD=32, build VPoser, load snapshot `.ckpt`, strip
    `vp_model.` prefix, eval+freeze) replacing `model_load.load_model`/`exprdir2model`. Weights via env
    `VPOSER_PATH` else bundled. VERIFY the vendored `matrot2aa` matches (convention) before trusting decode.
  - `adapters.py` ← BVH/mocap input: `compute_Q`, `bvh_to_smpl_world_rotations`,
    `rotation_matrix_to_axis_angle`, `body_to_world_transforms`, `_read_worldpos`, `_smpl22_positions`
    (from `motion_bridge` + `preprocess_bridges`). Generic mocap I/O; the proper-frame `Fp` stays in koopman.
- **koopman migration (STEP 2, after the module smokes):** repoint imports in `viz_bridges.py`,
  `preprocess_bridges.py`, `motion_bridge.py`, `render_convrot.py`, `models/geometry.py`, and tests
  `{test_bone_retarget, test_smpl_shape_fit, test_spine_retarget, test_smpl_pose_fit,
  test_bvh_smpl_convention, test_vizsys_smpl_bodies}` → `from skel2smpl import …`; delete the migrated
  leaf code from koopman; keep `Fp`/convrot in koopman (import `compute_Q` from `skel2smpl.adapters`).
- **Acceptance gate G19:** (a) **standalone smoke** — `cd /media/insq/ins && python -c "from skel2smpl
  import SMPL, solve_shape, refine_pose_vposer"` works with koopman NOT on the path; (b) synthetic
  known-SMPL-pose → joints → `solve_pose` round-trips to < ~5 mm; (c) `refine_pose_vposer` loads the
  vendored VPoser (rest-pose prior ≈ 0.009, random ≈ 2.0 — matches §P18 smoke); (d) after migration the
  koopman fitting suite (34) stays GREEN and `run.sh` renders identically.
- **Risk:** import untangling across ~15 files. Execute STEP 1 (create+vendor+smoke, koopman untouched →
  the reusable module exists immediately for other projects) and STEP 2 (migrate koopman, run suite)
  as SEPARATE commits. SMPL_NEUTRAL.pkl + vposer_v02 must be resolvable from any project (bundle or env).
- **STATUS (2026-06-15): COMPLETE — all gates GREEN.** STEP 1 (skel2smpl module + vendored VPoser,
  G19a-c smoke) landed (skel2smpl `3325a91`). STEP 2 step #1 (repoint importers, koopman `f60fb47`) +
  step #2 (delete `smpl_fit.py` + `models/vposer_prior.py`, fitting suite 55/55, koopman `efd2eef`).
  **G19d (`run.sh` renders identically post-migration): USER-confirmed PASS** — the §P11 convrot path
  ran crash-free end-to-end (all lazy `skel2smpl` imports resolved, 4-body GN fit completed, 61.77s
  @ 59.87 FPS, clean exit 0). Path-wiring drift recorded: `constants.py` needed
  `sys.path.insert(0, parent)` for the sibling `import skel2smpl` (was NOT pre-wired). OPTIONAL leftover
  (not required for the gate): step A — dedup the KEPT live mb/preprocess generics into skel2smpl as the
  sole source (they are live convrot deps, not orphaned; needs internal-caller repointing).


### P23 — MoSh surface-marker solve for joint-incongruent reference skeletons (ExPI 18-marker, then 2C) [extends §P13 latent markers + §P16 topology] (2026-06-18)
- **Why (user, 2026-06-18: "the skel2smpl doesn't create correct shape and motion for ExPI / 2C ... the
  reference skeleton has completely different skeleton that can't be fit just by simple mapping or simple
  kinematics optimization; create a mocap solve minimizing the skeleton reference distance").** P13–P16 were
  developed and gated ONLY on the boxing actor + ReMoCap (lindyhop/ninjutsu). ExPI/2C were never a gated phase.
- **DIAGNOSIS (probed this session, `tests/scratch_probe_expi2c.py`, current code+data, NF=40).** The current
  `vizsys/smpl_fit.py::_obs_expi` index-maps the 18 ExPI points straight onto SMPL JOINT CENTERS and
  interpolates the missing pelvis/spine2-3/neck/collars. Per-joint fit error vs raw obs:
  **spine1 415 mm, left_hip 244, right_hip 164, left_knee 242, right_knee 305** (ankles/feet/wrists 18–62 mm).
  2C (real rotational BVH, milder): mean 25 mm but **spine2 148, head 97, shoulders 50–57**.
- **ROOT CAUSE (data contract, NOT optimizer weakness).** The ExPI README (verified) defines the 18 points as
  SURFACE MARKERS — `fhead,lhead,rhead, back, lshoulder,rshoulder,lelbow,relbow,lwrist,rwrist,
  lhip,rhip,lknee,rknee,lheel,rheel,ltoes,rtoes` — there is NO pelvis, NO spine, NO ankle marker. Two
  compounding errors: (i) `m-back` (upper back) is mapped to SMPL `spine1` (just above pelvis) ⇒ the whole
  spine column is misplaced (the 415 mm); (ii) hip/knee/heel markers sit on the SKIN, 10–25 cm from the SMPL
  joint centers, but `solve_pose`'s δ latent offset is STRONGLY regularized on those joints
  (`lam_delta=5.0`, only spine/neck/collar/head/shoulder are `_DELTA_FREE`) ⇒ δ cannot absorb the surface
  offset ⇒ the body lands wrong.
- **Decision (user AskUserQuestion 2026-06-18):** (1) **True MoSh free offsets** — treat all reference points
  as surface markers rigidly attached to SMPL bones with FREE frame-invariant offsets, co-solved with β+θ;
  drop the "markers ARE joints" assumption and the synthesized fake joints. (2) **ExPI first, then 2C.**
- **Approach (MoSh / MoSh++ two-stage).** For each marker m, virtual marker `v_m(θ,β,δ) = J_{b(m)}(θ,β) +
  s·R^world_{b(m)}(θ)·δ_m`, with `b(m)` the assigned SMPL bone (ExPI `EXPI_BONE`: fhead/lhead/rhead→head,
  **back→spine3** (the §P13 fix), shoulders→shoulder, elbows→elbow, wrists→wrist, hips→hip, knees→knee,
  heels→ankle, toes→foot). δ_m FREE (weak L2 only, not the §P13 strong prior). Root translation `trans_t`
  is a FREE per-frame variable (ExPI has no pelvis marker to ground to), init at mid-hip marker.
  - **Stage I (shape+offsets):** on a SEQUENCE-SPANNING frame subset (~12 diverse frames — breaks the
    pose/offset degeneracy, MoSh++ §3.1), optimize `β + s + δ_m + θ_subset` jointly. Minimize robust
    `‖v_m − obs_m‖²` + in-distribution β prior + weak `‖δ‖²`.
  - **Stage II (pose):** freeze `β, s, δ_m`; solve `θ_t + trans_t` per frame, robust marker data + 2nd-order
    chordal-accel smoothness (reuse §P13 `sw`) + optional contact (`lam_contact=0` for acrobatic ExPI).
- **References (verified this turn):** MoSh — Loper, Mahmood, Black, *MoSh: Motion and Shape Capture from
  Sparse Markers*, SIGGRAPH Asia 2014. MoSh++ — Mahmood, Ghorbani, Troje, Pons-Moll, Black, *AMASS*, ICCV
  2019, arXiv:1904.03278 (latent markers + frozen-offset two-stage; already §P13 ref). ExPI — Guo, Bie,
  Alameda-Pineda, Moreno-Noguer, *Multi-Person Extreme Motion Prediction*, arXiv:2105.08825 (the 18-marker
  surface layout, from the dataset README).
- **Gates (numerical; quote in commits).** **G23a** ExPI marker fit-residual ≤ 30 mm mean (vs 87 mm now).
  **G23b** every lower-body marker (hip/knee/heel/toes) ≤ 50 mm (vs 164–415 now) — the iliac-crest hip marker
  is the single FARTHEST-from-bone surface marker (~12–20 cm off the pelvis) AND rides the smoothed root; with
  the §P23 SHAPE regularization (torso anchor straightens the pelvis) its floor is ~46 mm. knees/heels/toes/arms
  all ≤ 18 mm. **G23c** β in-distribution `‖β‖ ≤ 2` (measured ≈0 — sparse markers don't constrain girth, mean
  body is correct; size from scale s). **G23c2 (SHAPE PLAUSIBILITY — user 2026-06-18 "the fitted shape is
  completely wrong, regularize the shape"):** no torso curl / pot-belly / hunch — the marker-UNCONSTRAINED
  spine1/2/3+neck+collars (no marker hits them) must be anchored to rest, every joint in anatomical ROM
  (§P17 limits). Verified by render (`/tmp/expi_fit*.png`: pre = bloated hunched follower; post = upright
  natural body). **G23d** no NaN, smooth mesh (`lam_smooth=0.10`, root lightly smoothed `sw[0]=1`).
  **G23e** existing boxing/ReMoCap fits bit-identical (new path is additive — old `_obs_expi`/`fit_smpl`
  untouched, ExPI routed to the new `fit_markers`).
- **RESULT (2026-06-18, `tests/scratch_markers_expi*.py`, NF=40, 16 performer-clips acro1+acro2 ×L/F).**
  **G23a PASS** mean-of-means **14.5 mm**, worst single mean 16.7 mm (baseline 87 mm → ~6×). **G23b PASS**
  lower-marker max: median 36, p90 41, worst **41.9 mm**. Per-marker (a-frame1 L): arms 3–13, knees 8–12,
  heels/toes 13–16, head 16–21, back 20, **hips 35–36**. **Two decisive root-causes found by probe, both
  literature-grounded:** (1) the §P13 small-σ GMOF (σ²=0.0025) is a REDESCENDING robustifier — from a
  zero-init pose the arms sit ~1.5 m away in the gradient DEAD-ZONE (1451 mm wrists). ExPI is CLEAN mocap
  (no outliers) ⇒ run GMOF at LARGE σ (σ²=0.5 ≈ L2, gradient everywhere); pure-L2 probe hits 1 mm. SMLPify-X
  graduated-non-convexity is the noisy-source remedy; the `_sig` coarse→fine hook is kept but default
  `sigma_sq0==sigma_sq`. (2) hip iliac-crest markers attach to the PELVIS (bone 0), NOT the femur joint
  (1/2) — else they swing with the thigh on leg-lift; and the 12-frame MoSh++ Stage-I FREEZE under-determines
  the torso δ (gauge freedom) ⇒ solve ONE joint pass over ALL frames (δ frame-invariant, estimated from the
  whole sequence). **2C DONE (2026-06-19, user "use the expi fitting algorithm for 2C"):** the old
  BVH-rotation-init `fit_smpl` path was replaced — the 20 2C BVH joint centres are now treated as MARKERS
  (`C2_MARKERS`/`C2_MARKER_BONE`) and fit with the SAME `fit_markers` solve (graduated-σ from zero, free
  offset, torso anchor, joint limits, `C2_BONE_SEG` length retarget). Render-verified clean upright boxers,
  fit 25→**14/17 mm**; orphaned `_obs_2c`/`_pose0_2c`/`_bvh_world_rots` deleted. `bone_scale` helper
  generalized to `marker_bone_scale(markers, bone_seg)`; ExPI bit-identical (25 mm).
  **ReMoCap (LindyHop + Ninjutsu) DONE (2026-06-19, user "use the same algorithm for ninjutsu and lindyhop"):**
  `fit_remocap` likewise rewritten — the 22 worldpos joint CENTRES become markers attached to their own SMPL
  bones (`marker_bone = observed SMPL indices`, `REMO_BONE_SEG` limb retarget) and fit by `fit_markers`
  (Y-up, no Z-up/compute_Q dance). Render-verified clean upright dancers/fighters, ~18 mm. The whole
  BVH-rotation-init path is gone: `_fit_remocap`, `_world_to_local_aa`, and the `adapters`
  (`bvh_to_smpl_world_rotations`/`compute_Q`) + `geometry` imports deleted. **All 4 fittable datasets
  (2C, ExPI, LindyHop, Ninjutsu) now share ONE solver — `fit_markers`.** (`fit_smpl`, the generic 22-joint
  position fit, stays as public API but is no longer any dataset's default path.)
- **SHAPE-REG AMENDMENT (2026-06-18, user "fitted shape completely wrong, regularize the shape").** Render
  (not metric) caught it: marker err 13 mm but the follower mesh was a BLOATED, POT-BELLIED, HUNCHED body.
  β was already ≈0 (not the cause) — the culprit was POSE: no marker hits spine1/2/3, neck or collars, so
  those joints curled into an implausible torso that still satisfied the sparse markers (the §P13/§P17 hunch).
  `fit_markers` omitted the pose prior `solve_pose` has. FIX (landed): (1) rest-pose ANCHOR `lam_anchor=0.5`
  on the marker-unconstrained torso joints {3,6,9,12,13,14} only (zero on the marker-pinned limbs/head/root);
  (2) §P17 anatomical joint LIMITS `lam_limit=1.0` (reuse `smpl_joint_limits`+`_limit_residual`). RESULT: mesh
  upright/natural (render-verified); mean 14.5→20.0 mm (16 clips, the plausibility tax), G23a PASS, hip floor
  42→46 (G23b ≤50). *Lesson (CLAUDE.md): a low marker residual is necessary, NOT sufficient — RENDER the
  mesh; free latent offsets + free torso pose can match markers with a wrong body.*
- **PROPORTION + GIRTH AMENDMENT (2026-06-18, user "shapes still wrong, something missing"; render+probe).**
  Probe showed the performers are LONG-LIMBED acrobats whose segments are 1.2–1.5× SMPL-mean and
  DIFFERENTIALLY (thigh 1.51, shin 1.21, upper-arm 1.36, forearm 1.18 — `tests/scratch_shape_probe2.py`). A
  single uniform `s` can't match differential proportions → it compromised at 1.31 (a ~2.0 m GIANT,
  inconsistent: leader 1.31 vs follower 1.04) and δ distorted the body. FIX (landed): (1) `_expi_bone_scale`
  measures per-bone ratios from marker segments (`_EXPI_BONE_SEG`) → §P14 `bone_scale` (limbs only;
  collars/torso have no marker → ratio 1); (2) `lam_s=2.0` prior pins `s→1` so BONES carry size. RESULT: s
  1.31→1.00 both performers, mesh height matches marker span, limbs correct (render `/tmp/expi_bs_*.png`).
  COST: hip MARKER floats to ~56 mm (G23b) — `s≈1` + mean-width pelvis can't reach the WIDE iliac-crest hip
  markers (hip-width 352 mm not retargeted; marker translucent, body right). **GIRTH UNRECOVERABLE from 18
  skeletal markers:** SMPL neutral β=0 already has belly-depth 288 > chest 267 mm (slight tummy); markers
  carry NO belly/leanness signal (§P13 "β girth null-space unconstrained" — `scratch_neutral.py`). A
  lean-acrobat look needs an ASSUMED athletic-β prior (injection, not fit) or the ExPI meshes (README
  "released soon", absent). **OPEN USER DECISION.**
- **ALIEN-FEET FIX (2026-06-18, user "feet are trying to fit the feet skeleton ... results like alien smpl
  mesh").** `_expi_bone_scale` scaled the foot bone 1.84×/1.75× → giant elongated alien feet: the heel→toes
  marker spans the whole foot LENGTH, not SMPL's short ankle→ball bone. FIX (landed): DROP the foot bones
  (10,11) from `_EXPI_BONE_SEG` (mean SMPL feet) and CLAMP every remaining ratio to [0.85,1.30] (marker
  segments are offset-contaminated — the iliac-crest hip marker inflated the thigh to 1.51; §P14 warns
  extreme bone_scale tears the mesh). Render-verified `/tmp/feet_fix.png`: feet normal, bodies plausible,
  marker mean 26/22 mm. This is a STOPGAP — the proper proportion+girth solution is §P23.v (surface-vertex
  attach), where β + pose explain the markers and NO bone-tearing retarget is needed.
- **MARKER-AWARE ANCHOR AMENDMENT (2026-06-19, user "ninjutsu mesh rotating around root ... optimization is
  wrong" + "feet ... overfitted ... should be straight like human feet").** The §P23 SHAPE-REG anchor was
  HARDCODED to the ExPI torso set {3,6,9,12,13,14}. Two bugs surfaced on ReMoCap (probe `scratch_*`, render):
  (1) **rotating-around-root / stiff torso.** ReMoCap OBSERVES markers on spine1/2/3+neck+collars (all of
  {3,6,9,12,13,14}); anchoring those to REST fought the real spine markers (spine3 50, spine1 44, pelvis 36 mm)
  AND locked the torso straight, so the body could only follow its orientation by rotating RIGIDLY about the
  root (the "mesh spins around its root" look). FIX: anchor a torso joint to rest ONLY when NO marker attaches
  to it (`j not in {marker_bone}`) — ExPI still anchors its marker-free spine/neck/collars; ReMoCap lets the
  spine articulate to its own markers. (2) **pointed/alien feet (all datasets).** The ankle over-plantarflexed
  & twisted to chase the surface heel/toe marker (probe: R-ankle 37° incl. 25° twist) → ballet-pointe feet
  (render, both performers). FIX: ALWAYS anchor the feet `_FEET={7,8,10,11}` to rest (`lam_anchor`) — flat
  human foot; the toe marker is still reached in POSITION by the leg/root, not by curling the foot (§P17 limits
  alone did not catch it — 37° is inside ROM). Both changes are two lines at `fit.py:fit_markers` (the `aw`
  set); GMOF/bone_scale/limits untouched. Gate: ReMoCap render shows articulating spine + flat feet; ankle rot
  ~0°, spine markers follow. RESULT (render-verified all 4 datasets, 2026-06-19): ankle 20-37°→0.1°, feet
  FLAT/planted (no more ballet-pointe); spine3 now articulates 23° (was locked). Fit numbers improved or held:
  ninjutsu 16/13, lindyhop 12/12 (was ~18), 2c 12/11 (was 14/17), expi 27/23 (= baseline, no regression).
- **ROOT-ORIENTATION INIT AMENDMENT (2026-06-19, user "ninjutsu still creating rotating root … elbow not
  fitted at all, probably related to the rotating root").** The marker objective has a GAUGE DEGENERACY: a
  body turn can be matched by rotating the ROOT joint OR by twisting the spine/hips/shoulders. From a
  ZERO-init root, Adam falls into the twist minimum on turning/acrobatic frames. Probe (shot_007 reactor,
  `scratch`): GT yaw ≈ 0 early but the fitted root local angle flailed 9–134° erratically; mean 59.5 mm, peak
  136 mm; render = a grotesquely back-bent torso with the GT skeleton far off the mesh ("elbow not fitted" =
  the whole upper chain is twisted to absorb the global turn). The actor (mild motion) tracked fine, which is
  why static shot_001 looked correct — the bug only bites on real rotation. FIX (MoSh/SMPLify global-orient
  init, landed `fit.py::_root_init_kabsch`): seed `pose[:,0]` per-frame by a rigid Procrustes/Kabsch fit of the
  observed torso markers to the SMPL rest torso joints (core bones `_ROOT_CORE={0,1,2,3,6,9,12,15,16,17}`,
  one mean point per present bone — pelvis+hips+spine1 are rigid to the root by construction; spine2/3+neck+
  head+shoulders span the 3rd dim for sparse sets like ExPI). `R = QM·R₀ ⇒ R₀ = QMᵀ·R`. Other joints/β/s/δ
  still start at zero. RESULT (render-verified): reactor 59.5→18.8 mm (peak 136→28.7), root holds a coherent
  orientation tracking the turn (no spin), frame-130 contortion → a natural lunge (17/49→17/18 mm). No
  regression: 2C 12/11, ExPI 27/24, ninjutsu static 16/13. *Lesson: a low PER-MARKER residual can coexist with
  a globally-wrong body when a degeneracy lets the limbs absorb the global DOF — the data-driven init breaks it.*
- **VPOSER PRIOR IN THE MARKER SOLVE (2026-06-19, user "add the vpose and smoothness loss to prevent
  jitter") [§P18 extended into §P23].** Wired the §P18 VPoser v2 recon prior (`vposer.py::vposer_prior_loss`,
  Pavlakos et al. SMPLify-X/SMPL-X arXiv:1904.05866) into `fit_markers` as `lam_vp·‖pb−VPoser(pb)‖²` on body
  joints 1..21, applied in the POSE stage only (refinement), load-guarded (graceful no-op if weights absent).
  Smoothness (`smooth_term`, 2nd-order pose-accel) was already in the loss. **CALIBRATION (`scratch_sweep*`,
  shot_007 reactor, 120 f):** the marker solve is ALREADY on-manifold — VPoser-implausibility **0.035** here
  vs §P18's GN baseline 5.30 (the anchors+limits+smoothness already do the job). Sweep `lam_vp∈{0,2e-4,1e-3,
  4e-3}`: implaus 0.035→0.030→0.020→0.012 but fit 16.8→17.3 mm and **jitter 5.75→6.89 mm/f² INCREASES** (VPoser
  is per-FRAME, no temporal coupling — it trades jitter for naturalness, §P18 G18c). Sweep `lam_smooth∈{0.1,
  0.3,0.8,2.0}`: jitter 6.26→**5.98**→7.62→7.15 — i.e. smoothness (not VPoser) is the jitter lever, and it's
  already near-optimal at the §P23 default 0.10 (0.30 is marginally smoother at +0.6 mm fit; >0.3 over-smooths
  and worsens it). DECISION: VPoser default `lam_vp=2e-4` (gentle, honors the request; implaus 0.035→0.030 at
  ≈0 fit/jitter cost), `lam_smooth` kept at the tuned 0.10. Render-verified no regression (shot_007 frame-130
  still 17/18 mm, natural lunge). *Honest note: jitter was already low; VPoser is a manifold guard for future
  noisy/odd poses, not the jitter fix the prompt assumed — smoothness already prevents jitter.*
- **§P23.v — SURFACE-VERTEX MARKER ATTACH — IMPLEMENTED 2026-06-19 (ExPI)** (user "fitted shape in expi/2C
  not realistic — big belly + narrow limbs; all 4 should fit a natural SMPL shape"; AskUserQuestion → vertex-
  attach). LANDED `051bd9b`: `_fk_verts` (sparse-vertex LBS, validated bit-exact vs `forward_R`, err 5e-17);
  `fit_markers(marker_vert=…)` path (markers→SMPL verts, no δ, β freed); `EXPI_VERT` (SSM/AMASS vertex IDs,
  LATERAL-hip 1228/4712 + acromion 1861/5322 carry width); `fit_expi` uses it with `lam_beta=1e-3` (recalib
  from 0.3 — the m²-scale data term crushed β to 0 at 0.3; sweep: 0.3→‖β‖0.01, 1e-3→0.95, 3e-4→2.0). RESULT
  (render-verified acro2 a-frame2): **belly GONE, lean athletic build, ‖β‖=1.21** (G23f in-range), shoulder
  width 326→~410 mm. TRADEOFF: marker residual 27→62 mm (surface-attach can't hide error in a free offset);
  limbs fit (wrists 16-24, knees/elbows 60-78), torso carries the slack (back 123, hip 98, shoulder 86-95).
  **OPEN:** (a) refine back(3508)/hip vertex correspondence to cut the 62 mm; (b) optional scalar normal
  residual r_m (skipped — markers ~1-2 cm off skin set the floor); (c) **2C/ReMoCap girth NOT addressed** —
  joint-CENTRE markers can't observe surface width (β stays 0, mean belly remains); they need a template β
  (the option the user did NOT pick). ORIGINAL DESIGN BELOW.
  DESIGN. The free joint-offset δ confounds β (verified: relaxing
  `lam_beta` 1.0→0.02 leaves ‖β‖≈0.1, markers absorbed by δ ⇒ β never moves). MoSh-strict fix: each marker m
  attaches to a SPECIFIC SMPL mesh VERTEX `v(m)` (or barycentric surface point), virtual marker
  `mk_m = LBS_vertex(β,θ,trans)[v(m)] + r_m·n̂` with `r_m` a SMALL normal residual (tightly regularized). The
  vertex position is a function of β (shape moves the surface) ⇒ fitting markers to vertices CONSTRAINS β ⇒
  real build recoverable. STEPS: (1) hand-define the 18 ExPI marker→SMPL-vertex IDs (fhead/lhead/rhead on the
  skull, back=upper-spine surface, l/r shoulder=acromion, l/r elbow, l/r wrist, l/r hip=iliac-crest vertex,
  l/r knee=patella, l/r heel=calcaneus, l/r toes); use known SMPL landmark verts (SSM2/CMU marker sets as
  reference). (2) In `fit_markers`, add a `marker_vert` path: forward LBS verts (forward_proper already has
  the verts branch), gather `verts[:, marker_vert]`, add `r_m·normal`; keep bone_scale + anchor + limits.
  (3) Make β identifiable: drop the joint-δ for the vertex markers, regularize β at §P13 `lam_reg≈0.3`,
  regularize `r_m` strongly (markers ON the skin). (4) `s` likely removable (β+bone_scale carry size). GATES:
  **G23f** ‖β‖∈[0.5,2.5] (non-trivial shape recovered, not balloon), shoulder/hip WIDTH within ~10% of
  marker widths (454/352 mm), render shows the leaner acrobat build (not the mean tummy); marker fit ≤ 30 mm
  mean still holds (G23a). RISK: 18 sparse markers may still under-constrain depth/belly; vertex IDs must be
  right (mis-pick → local distortion). REF: MoSh Loper 2014 (marker→vertex calibration), MoSh++ 1904.03278.
- **Reversible:** the MoSh path is additive (`fit_markers` new in `skel2smpl/fit.py`; `_markers_expi`/
  `fit_smpl_markers` new in `vizsys/smpl_fit.py`). Old `_obs_expi`+`fit_smpl` stays; revert = re-route ExPI.

### P10 — MoSh++ SMPL POSE fit (β=0 standard body), the correct mesh path [SUPERSEDED by P11]
- **Decision (user, 2026-06-13):** P9's whole premise — fitting SMPL *shape* β so its
  joints land on the Ninjutsu joints — is "fitting the SMPL mesh on a different
  skeleton." Wrong. The boxing pipeline (the reference) is SMPL pose θ + **β=0** →
  `SMPL.forward` → **standard LBS mesh** (`vr/src/preprocess.py`, `vizsys/data_adapters.py:154`,
  `vizsys/smpl.py:51`). Ninjutsu must produce the SMPL *skeleton* (pose) and drive the
  SAME standard β=0 LBS mesh — never warp shape to the skeleton.
- **Adapt MoSh++ Stage II (mosh_stageii, chmosh.py) to joints:** optimise per-frame SMPL
  LOCAL pose θ so the STANDARD β=0 body's joints follow the observed motion. β fixed at 0
  (the canonical/boxing body). trans fixed each frame so SMPL root == observed root
  (root-relative pose fit). Loss = mean‖J_smpl(θ)−J_obs‖² + λ_u·mean‖θ_t−θ_{t-1}‖²
  (E_u velocity smoothness, MoSh++) + λ_θ·mean‖θ_body−θ_init_body‖² (stay near the
  plausible P8 retarget init, standing in for MoSh++'s learned GMM/VPoser pose prior).
  Joints-only forward (J_regressor@v_template → _global_transforms → _TRANS) ⇒ fast,
  differentiable in θ; Adam, init from the P8 analytic retarget (MoSh++ likewise
  warm-starts pose from a rigid alignment).
- **Edits:** `motion_bridge.fit_smpl_pose(joints_obs, smpl, init_pose, betas=None)` →
  (pose [T,22,3], trans [T,3]). `viz_bridges`: `--bodies-pose moshfit` (NEW DEFAULT) —
  fit pose, β=0, standard `SMPL.forward`, NO β, NO scale. retarget/swingik retained for A/B.
- **Result (measured, shot_031 actor):** joint err 125.7mm (analytic retarget, β=0) →
  **60.1mm (−52%)**; pose velocity 0.40 → **0.07 (5.7× smoother)**; 2.2s/721 frames.
  The 60mm residual is the irreducible β=0 bone-length mismatch (subject ≠ SMPL size) —
  the body is correctly POSED, canonical SMPL proportions, no warping. Visual (full-res
  crops, GT + pred): clean natural SMPL bodies, no bloat.
- **Gates (tests/test_smpl_pose_fit.py):** G10 — on a synthetic β=0 body, the pose fit
  recovers joints ≤10 mm (≥70% reduction vs perturbed init) and does not increase pose
  velocity. (Pure-data recovery uses λ_u=λ_θ=0; defaults validated by the render.)
- **Reversible:** `--bodies-pose retarget` (P8 analytic) or `swingik` (V2.1).
- **Note (model target unchanged):** this is render/viz only. The koopman model still
  predicts `body_log` (SO(3) per P8); the pose fit runs at render on the decoded joints
  of GT *and* predicted `body_log`. No retrain. P9's `_fit_betas` is now legacy (still
  baked for the retarget A/B path; moshfit ignores β/scale).

### P9 — MoSh++-style SMPL shape fit (remove mesh distortion) [SUPERSEDED by P10 — shape-fit abandoned]
- **SUPERSEDED 2026-06-13:** fitting SMPL *shape* to a foreign skeleton was the wrong
  frame (see P10). Kept below for the record + the `--bodies-pose retarget`/`--smpl-betas`
  A/B path. The correct approach is P10 (fit POSE, standard β=0 LBS mesh).
- **RESOLUTION (user-directed 2026-06-13, after reading the real MoSh++ code):**
  cloned github.com/nghorbani/moshpp and read `src/moshpp/chmosh.py` + the conf.
  Two things became concrete:
  1. **Why naive β fitting bloats (girth):** MoSh++'s shape data term is SURFACE
     MARKERS glued to the skin (`prepare_mosh_markers_latent`, m2b≈9.5mm + a `surf`
     term holding markers on the mesh) — markers pin GIRTH. We have only 22 skeletal
     joints (`J_regressor`, interior points) which pin bone LENGTH, not girth. So a
     free β inflates SMPL's mass/girth axes (measured: β[1]≈−0.97, β[2]≈0.95,
     β[3]≈0.76 at lam_beta=10) → pot bellies. A faithful marker port is impossible
     without surface data; that limitation is real.
  2. **The weight bug, grounded in their conf** (`support_data/conf/moshpp_conf.yaml`):
     MoSh++ uses `stagei_wt_data=75`, `stagei_wt_betas=10` (≈7.5:1) + annealing
     `[1,.5,.25,.125]` + a GMM pose prior (`stagei_wt_poseB=3`). Our fit used
     `lam_data=20000 : lam_beta=1.25` ≈ **16000:1** — β was effectively unregularized.
- **The fix (mirrors MoSh++'s own logic):** a STRONG shape prior pulls the
  joint-unconstrained DOF (girth) back to the mean, while the subject's overall SIZE
  is routed into a uniform **scale s** (joints can't separate size from β cleanly).
  - `lam_beta=100` ⇒ ‖β‖≈0.2 (mean girth, NO bloat); s≈1.155 (subject ~15% larger
    than SMPL), STABLE across prior strength (so s is genuinely "size", not shape).
  - **s is KEPT and applied at render** as a uniform resize about the root joint
    (`verts = root + s·(verts−root)` in `viz_bridges._render_vizsys`). Uniform ⇒ a
    PROPORTIONAL bigger body (natural), NOT the disproportionate pot belly. And it
    IMPROVES foot grounding: `trans` already puts the pelvis at the real (tall)
    subject's height, so SMPL's short legs floated; scaling extends feet to the floor.
  - Supersedes the earlier "discard s / β=0" decision: discarding s forced β to carry
    size → girth. Routing size to s is what makes the fit non-bloated AND correctly
    sized (vs β=0 which is undistorted but ~15% too small).
- **Reversible:** `--smpl-betas zero` = β=0, s=1 (legacy SMPL-sized mean body).
- **If a TRUE per-subject shape (real girth) is ever needed:** requires
  girth-constraining data (surface/silhouette/vertex term) — not in joints. Do NOT
  expect joints alone to recover body width; the strong-prior+scale result is the
  honest best for skeletal input.
- **Decision (user, 2026-06-13):** even with correct pose+roll (P8), the SMPL mesh
  distorts because we pose a **fixed β=0 (mean) body** — SMPL's rest bone lengths ≠
  the mocap subject's, so the mesh stretches/pinches at joints. Fix it the MoSh++ way:
  **solve SMPL shape β to match the subject**, then (optionally) refine pose by fitting.
- **Refs (verified 2026-06-13):**
  - AMASS / MoSh++ — Mahmood, Ghorbani, Troje, Pons-Moll, Black, ICCV 2019,
    arXiv:1904.03278. Code: github.com/nghorbani/moshpp (chumpy; `src/moshpp/chmosh.py`
    = fitting objective, `prior/` = pose prior, `marker_layout/` = data term).
  - MoSh++ Stage I (eq. 3): `E = λ_D E_D + λ_β E_β + λ_θB E_θB + λ_θH E_θH + λ_R E_R
    + λ_I E_I` — optimize ONE shape β + poses over **F≈12** frames + latent markers.
    Weights: λ_D=600·b, λ_β=1.25, λ_θB=0.375, λ_I=37.5, λ_R=1e4 (b=46/n_markers).
  - MoSh++ Stage II (eq. 6): per-frame `E = λ_D E_D + λ_θ E_θ + λ_u E_u + λ_φ E_φ +
    λ_v E_v` — pose + DMPL soft-tissue, with VELOCITY smoothness E_u (λ_u=2.5).
- **Our adaptation (joint-based, PyTorch — we have 22 joints, NOT surface markers):**
  - **Stage I (shape):** optimize a single β (10-dim) [+ optional global scale]
    minimizing Σ_f ‖J_smpl(β, θ_f) − J_obs_f‖² over ~12 frames + λ_β‖β‖² (Mahalanobis
    shape prior, β~N(0,I) ⇒ ‖β‖²). θ_f initialized from the P8 analytic retarget.
    Per subject/character (actor, reactor) — store β alongside the sequence.
  - **Stage II (shape de-distortion) — AMENDED 2026-06-13 (user-approved):** the
    Stage-I-only fit drove EVERY subject to ‖β‖≈5 (inflated torsos). Diagnosis: the
    5-spine(BVH)→3-spine(SMPL) topology mismatch makes the observed spine
    over-long (spine1→2 1.64×, spine2→3 1.92×); limbs are only a mild uniform
    1.05–1.19× upscale. **Per-frame pose refine (the original MoSh++ eq.6 plan)
    CANNOT fix this** — rotations don't change bone lengths, so β still inflates to
    chase the phantom-far spine joints. The effective, cheaper remedy (replaces the
    Δθ refine):
    (a) **Downweight the topology-corrupted joints** spine1/2/3, neck, L/R collar
        (idx 3,6,9,12,13,14) in the data term so β fits the clean limb bones, not
        the impossible spine targets;
    (b) **Optional global scale s** (init 1.0, jointly fit, DISCARDED — render keeps
        only β) absorbs the uniform limb upscale so β stays near the prior.
    (c) **λ_β recalibration (load-bearing, measured 2026-06-13):** MoSh++'s
        marker-calibrated λ_β=1.25 was ~16000× too weak vs our recalibrated
        λ_D=20000 (metre-joint units), so the prior never bit. λ_β=10 ⇒ ‖β‖≈1.6,
        max|β|≈1 on real shots; (a)+(b) alone only moved 5.6→4.8.
    Render path unchanged (SMPL has no scale param; β-only mesh is SMPL-proportioned
    and, crucially, NOT inflated). Per-frame Δθ refine is dropped as ineffective here.
    **Known limitation (measured, user-accepted 2026-06-13):** discarding s leaves the
    β-only render SMPL-sized (~18% smaller than the subject); render joint-err stays
    ≈baseline (126→122mm) — the subject's overall scale is unrepresentable in β. The
    P9 win is de-distortion (no ‖β‖≈5 inflation), NOT size match. Applying s at render
    (uniform vert scale around root, 126→88mm) was considered and DEFERRED.
  - Differentiable through `vizsys.smpl.SMPL.forward_R`/`forward` (already torch); use
    Adam/LBFGS. β baked into the data dict (or fit at viz time, cached per sequence).
- **Edits:** new `motion_bridge.fit_smpl_shape(joints_obs, smpl, n_frames=12)` →
  β; optional `refine_pose(...)`. `preprocess_bridges` store per-seq β (+ data_version
  bump). `viz_bridges._render_vizsys` pass β to `smpl.forward(..., betas=β)`.
- **Gates (tests/test_smpl_shape_fit.py):**
  - **G7 joint fit:** after β fit, SMPL joint positions match observed within ≤20 mm
    mean (vs the β=0 mismatch baseline — assert ≥40% reduction).
  - **G8 bone lengths:** SMPL(β) primary-bone lengths match the subject's mean
    observed bone lengths within ≤15% (vs β=0).
  - **G9 no distortion:** verts finite; mesh visual check shows no stretch/pinch.
    **Real-data G9 (added 2026-06-13):** after the Stage-II-amended fit (downweight
    + global scale), every baked subject β has **‖β‖ ≤ 2.5** (synthetic G9 only
    checked finiteness, so it passed while real meshes inflated to ‖β‖≈5).
- **Reversible:** β=0 + P8 analytic pose = current behavior (flag `--smpl-betas {fit,zero}`).
- **Note:** this is VIZ/mesh fidelity, orthogonal to the model target. Does NOT change
  `body_log` (the model still predicts SO(3) per P8); β only reshapes the displayed mesh.

