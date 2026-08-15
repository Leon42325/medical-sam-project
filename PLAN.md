# Project Plan — SAM for Medical Images, Revisited

**Course project (Representation Learning) — reproduction + extension of**
Huang, Y. et al. *Segment anything model for medical images?* Medical Image Analysis **92** (2024) 103061.
DOI: 10.1016/j.media.2023.103061 · arXiv:2304.14660 · Code: https://github.com/yuhoo0302/Segment-Anything-Model-for-Medical-Images

**Constraints fixed with the user (2026-08-15):** 4–6 weeks · NHR@FAU **TinyGPU**, limited quota ·
additional experiment = **cross-comparison of medical-specialised SAM variants** · report in **English, LaTeX (Elsevier/MICCAI style)**.

---

## 1. Objective and success criteria

The deliverable is a portfolio-grade artefact: a ~10-page scientific report + a clean, reproducible GitHub
repository. Every item of the supervisor's rubric is mapped to a concrete work package below.

| Supervisor's requirement | Where it is satisfied |
|---|---|
| Read the paper, work through forward/backward references | §2, §3; Related Work section of the report |
| Find open-source implementations; **do not reimplement** | §7 — SAM/SAM2/SAM3/MedSAM/SAM-Med2D/MedSAM2 are consumed as upstream dependencies; we write only the harness |
| Independent work, effective use of web resources | Phase 0 data/source audit (§6, §12) |
| Critical analysis | §9 — four concrete methodological criticisms of the original paper, each empirically quantified |
| A genuinely interesting additional experiment | §5 — H2/H3: *specialisation–generalisation trade-off* of medical SAMs, tested on modalities absent from all training corpora |
| Suggestions for improvement (method / implementation / evaluation) | §9 + dedicated report section |
| Evaluated on additional data | §6.2 — OCT, PET, mammography (modalities absent from COSMOS 1050K) + post-2023 datasets |
| Report reads as a follow-up publication, hypothesis-driven | §5 (falsifiable H1–H4), §11 (report outline) |

**Definition of done.** (a) All reproduction numbers reported with bootstrap CIs and an explicit
agreement/deviation analysis against the published tables. (b) All four hypotheses either supported or
refuted with a stated statistical test. (c) `git clone && bash scripts/reproduce_smoke.sh` runs end-to-end
on one dataset on a single GPU in under 30 min.

---

## 2. What the original paper does (condensed)

- **Dataset.** COSMOS 1050K: 53 public MIS datasets → 18 modalities, 84 objects, 125 object–modality
  paired targets, 1,050,311 2D images, 6,033,198 masks.
- **Models.** SAM ViT-B (91M) and ViT-H (636M); ViT-L skipped.
- **Six testing strategies.** S1 automatic *everything* (32×32 point grid); S2 one positive point;
  S3 five positive points; S4 five positive + five negative points; S5 one box; S6 one box + one positive point.
- **Prompt construction (§3.3).** Positive point = centre of mass of the GT mask, with fallback to uniform
  sampling over the flattened mask when the centre of mass falls outside; additional positives by uniform
  sampling; negatives sampled uniformly in the 2×-enlarged bounding box outside the GT; box = tight GT bbox.
- **Mask matching (§3.5).** SAM emits *N* masks per prompt; the one with the **highest Dice against the GT**
  is selected for evaluation.
- **Metrics.** DICE, JAC (IoU), HD.
- **Analyses.** Model-size comparison (Fig. 6); per object–modality results (Tables 2–3); per-modality
  results (Fig. 10); inference time (Table 4); number of grid points in *everything* mode (Table 5);
  **Spearman partial correlation of Dice with object attributes** — size, intensity difference, elliptic
  Fourier descriptor order (boundary complexity), modality, aspect ratio (Table 6, Figs. 13–15);
  human annotation study (Table 7); **prompt-jitter robustness** (Table 8); comparison against interactive
  methods FocalClick/SimpleClick (Fig. 16); task-specific fine-tuning of the mask decoder with box prompts
  (§4.12, Figs. 17–18).
- **Headline conclusions.** ViT-H > ViT-B; box ≫ points ≫ everything; strong sensitivity to prompt jitter
  (up to −29.9 % Dice for a 20–30 px box shift); Dice correlates with intensity difference (ρ≈0.4–0.6) and
  negatively with Fourier order (ρ≈−0.4…−0.6), weakly with size, not at all with modality/aspect ratio;
  fine-tuning gains +4.39 % (ViT-B) / +6.68 % (ViT-H) mean Dice.

**What upstream code actually exists.** The official repo provides only the **box-prompt** path:
`pre_grey_rgb2D.py` (embedding extraction), `train_only_box.py`, `test_only_box.py`, `cal_matric.py`,
plus a vendored copy of `segment_anything` and the fine-tuned checkpoints. **Not released:** the S2–S4
point-prompt construction, the *everything*-mode pipeline, the EFD/Fourier-order attribute extraction, the
partial-correlation analysis, and the prompt-jitter experiment. Those are exactly what our harness adds —
without reimplementing any model.

---

## 3. Backward and forward references to work through

**Backward (cited by the paper — for Related Work).** Kirillov et al., *Segment Anything* (ICCV 2023);
Ma & Wang, *Segment anything in medical images* (MedSAM, Nature Communications 2024); Wu et al.,
*Medical SAM Adapter*; Mazurowski et al. (MedIA 2023); He et al. (2023); Ji et al. (2023a,b); Deng et al.
(2023); Zhou et al. (2023); Cheng et al., SAM-Med2D; Kuhl & Giardina (1982) for elliptic Fourier descriptors;
Isensee et al., nnU-Net.

**Forward (published after the paper — the critical-analysis and positioning material).**
SAM 2 (Ravi et al., 2024) and **SAM 3 / 3.1 (Meta, Nov 2025, promptable *concept* segmentation with text)**;
MedSAM2 (Apr 2025); MedicoSAM (2025); MedSAM3 (arXiv 2511.19046) and Medical SAM3 (arXiv 2601.10880);
VoxTell (CVPR 2026, free-text 3D medical segmentation); the Nov-2025 SAM 2 vs SAM 3 medical benchmark
(arXiv 2511.21926) — which explicitly states its limitation: *"we restrict our analysis to visual prompts."*

---

## 4. Research questions

- **RQ1 (reproduction).** Do the core findings of Huang et al. survive an independent, smaller-scale
  reimplementation of their protocol?
- **RQ2 (additional experiment).** *Where* do medical-specialised SAM variants improve over generic SAM —
  uniformly, or only in specific object-attribute regimes?
- **RQ3.** Domain adaptation vs. model scale: which buys more medical segmentation accuracy?
- **RQ4 (additional data).** Do those conclusions transfer to modalities absent from **both** COSMOS 1050K
  and the medical models' training corpora?

---

## 5. Hypotheses (falsifiable, each with a stated test)

**H1 — Attribute-dependence replicates.**
Under box prompts, Dice correlates positively with foreground–background intensity difference and negatively
with boundary complexity (elliptic Fourier order), with only a weak size effect and no modality/aspect-ratio
effect.
*Test:* Spearman partial correlation per strategy, replicating Table 6; agreement declared if the sign and
the ordering of |ρ| match and our ρ falls within the bootstrap CI of the published value.

**H2 — Medical fine-tuning flattens the attribute-dependence slope rather than shifting the mean.**
MedSAM / SAM-Med2D / MedSAM2 gain most in the *low-contrast, high-boundary-complexity* regime, i.e. they
reduce |ρ| against Fourier order and intensity difference.
*Test:* per-model linear/quantile regression of Dice on standardised attributes; compare **slopes** (not
means) with a bootstrap test over targets; report interaction term model × attribute.

**H3 — Specialisation–generalisation trade-off (the headline claim).**
Medical-specialised models beat generic SAM on in-domain modalities but **lose their advantage, or fall
behind, on modalities absent from their training corpora** (OCT, PET, mammography). Formally: the
model-ranking correlation between the in-domain and the out-of-domain target sets is significantly below 1.
*Test:* per-target Dice ranking of the 6 models on in-domain vs. out-of-domain sets; Kendall's τ with a
permutation test. Reporting a *null* result here is an acceptable and publishable outcome.

**H4 — Fine-tuned models inherit the prompt-jitter distribution they were trained with.**
MedSAM (trained with box jitter of 0–20 px) degrades more gracefully under box perturbation than SAM, while
SAM-Med2D (point-and-box training) degrades more gracefully under point perturbation.
*Test:* replication of Table 8's jitter protocol across all models; compare degradation slopes.

> **Why this is not a leaderboard.** A plain "which medical SAM wins" table is uninteresting and already
> exists in the literature. The contribution here is the **decomposition** of the gains — attribute-wise
> (H2), domain-wise (H3), and robustness-wise (H4) — using the original paper's own perception framework as
> the analysis lens.

---

## 6. Data plan

### 6.1 Reproduction subset (drawn from COSMOS 1050K's 53 datasets)

Target: **~10–12 datasets, ≥10 modalities, 22–26 object–modality pairs**, chosen for (a) modality coverage,
(b) *attribute* coverage — deliberately spanning the extremes of the size / contrast / boundary-complexity
space that H1–H2 are about, and (c) frictionless public download (no multi-week DUA).

| Dataset | Modality | Targets | Why (attribute regime) |
|---|---|---|---|
| MSD (Task03 Liver, Task09 Spleen, Task07 Pancreas) | CT | liver, spleen, pancreas | large & smooth vs. **low-contrast** (pancreas) |
| CHAOS | CT, T1W-MRI, T2W-MRI | liver, kidney, spleen | **same object, three modalities** — isolates the modality factor |
| ACDC | cine-MRI | LV, RV, myocardium | myocardium = **high aspect-ratio ring** |
| CAMUS | US | LV, myocardium, atrium | speckle noise, weak boundaries |
| Montgomery County CXR | X-ray | lung | large, high contrast (paper's best modality) |
| Kvasir-SEG + CVC-ClinicDB | Colonoscopy | polyp | two datasets, same object → inter-dataset variance |
| ISIC 2018 | Dermoscopy | melanoma | large, high contrast, simple boundary |
| DRIVE + CHASE-DB1 + STARE | Fundus | retinal vessels | **extreme Fourier order** — the H1/H2 stress test |
| Warwick-QU (GlaS) | Histopathology | gland | textured, ambiguous boundary |
| EPFL-EM (Lucchi) | Electron Microscopy | mitochondria | small objects, many instances |

**Sampling protocol (must be pre-registered in the repo before running).** ≤300 masks per object–modality
target, sampled with a fixed seed, stratified by volume/patient and by relative slice position, so that no
single large volume dominates. Slices retained only if the label area > 50 px (paper §2.2). Preprocessing
follows paper §2.2 exactly: main viewing plane for 3D, per-volume min–max normalisation to [0, 255], export
to PNG. Total ≈ 5,000 unique images / ≈ 7,000 masks.

### 6.2 Additional data (evaluated on data the paper never used)

Two tiers, both required by the rubric ("evaluated on additional data") and by H3.

**Tier A — modalities absent from COSMOS 1050K's 18** (the clean out-of-domain test):
- **OCT** — retinal fluid / layer segmentation (Duke SD-OCT DME; fallback: an OCT subset of MedSegBench).
- **PET** — AutoPET-II whole-body FDG lesion segmentation (TCIA; subsample heavily).
- **Mammography** — CBIS-DDSM mass ROIs (TCIA, public; fallbacks: INbreast, VinDr-Mammo/PhysioNet).

**Tier B — post-2023 datasets in seen modalities** (temporal generalisation): CholecSeg8K (laparoscopy),
SegThy (thyroid/vessel US), and/or 2–3 datasets from **MedSegBench** (Nature Sci. Data 2024; `pip install
medsegbench`, 35 standardised 2D datasets) as a low-friction top-up.

**Fallback rule.** Any dataset that is not downloadable and usable within Phase 0 (week 1) is dropped and
replaced from MedSegBench; the substitution is recorded in the repo. No week-3 data emergencies.

### 6.3 Contamination audit (Phase 0, blocking for H3)

H3 is only meaningful if we know what the medical models were trained on. **To verify from primary sources**
(the MedSAM Nature Communications supplementary, the SAM-Med2D paper, the MedSAM2 paper) — *not* from
secondary summaries, which conflate MedSAM's training list with COSMOS's dataset list:
1. Exact training-corpus dataset list for MedSAM, SAM-Med2D, MedSAM2.
2. Intersection with our reproduction subset → each target labelled **seen / unseen / unknown** for each model.
3. All in-domain results reported with the seen/unseen split made explicit. This audit is itself a
   contribution: existing medical-SAM benchmarks routinely evaluate on data the models were trained on.

---

## 7. Models — all consumed as upstream dependencies, none reimplemented

| Model | Encoder | Params | Prompts | Source | Role |
|---|---|---|---|---|---|
| SAM ViT-B | ViT-B | 91M | pts / box / everything | `facebookresearch/segment-anything` | reproduction baseline |
| SAM ViT-H | ViT-H | 636M | pts / box / everything | same | scale control |
| SAM 2.1 (Hiera-L) | Hiera | 224M | pts / box | `facebookresearch/sam2` | **generation** control (newer generic model) |
| MedSAM | ViT-B | 91M | box only | `bowang-lab/MedSAM` | domain adaptation, box-trained |
| SAM-Med2D | ViT-B + adapter | ~271M | pts / box | `OpenGVLab/SAM-Med2D` | domain adaptation, point+box-trained |
| MedSAM2 | Hiera | — | box / pts | `bowang-lab/MedSAM2` | domain adaptation on newer backbone |
| SAM ViT-B/H, our fine-tune | ViT-B / ViT-H | — | box | our training, paper §4.12 | reproduces the paper's own fine-tuning result |
| *(stretch)* SAM 3 / 3.1 | — | — | pts / box / **text** | `facebookresearch/sam3` | only if the schedule allows |

The 2×2 design **{generic, medical} × {older backbone, newer backbone}** is what lets RQ3 separate *scale /
generation* from *domain adaptation* — that separation is the reason for including SAM 2.1, which is
otherwise not needed.

Licence note: SAM 3/3.1 weights are gated behind the Meta SAM licence on Hugging Face — request access in
Phase 0 so the stretch goal stays open.

---

## 8. Evaluation protocol

- **Strategies.** S1–S6 exactly as in paper §3.3, implemented from the paper text (not released upstream)
  and covered by unit tests (centre-of-mass fallback, negative sampling inside the 2× box, tight bbox).
  Models that only accept box prompts are evaluated on S5/S6 and marked N/A elsewhere.
- **Mask matching.** Paper rule (max Dice vs. GT) **and** the deployable rule (SAM's own predicted-IoU
  score) — both recorded, see §9.1.
- **Metrics.** DICE, JAC, HD — plus **NSD (normalised surface Dice)** and **Boundary IoU** as an
  evaluation improvement (§9.4). Per-target mean ± std and BCa bootstrap CIs over masks.
- **Attributes** (paper §4.8): pixel area; aspect ratio (short/long side of the bbox); intensity difference
  between the structure and a surrounding ring obtained by dilating the bbox by a 0.1 relative factor;
  elliptic Fourier descriptor order via the paper's iterative criterion (increase order until contour
  Dice > 97 % or the improvement < 0.1 %, then `F_final = F_a + n·100·(1−DICE)`, n = 2); modality label.
- **Statistics.** Spearman partial correlation (replicating Table 6) **and** a mixed-effects model with
  dataset/patient as a random effect (§9.2).
- **Prompt-jitter robustness.** Table 8 protocol: box/point displacement of 0–10, 10–20, 20–30 px, three
  random seeds, all strategies, all models.
- **Everything-mode point-count ablation.** Table 5 protocol, capped at 128² points on 3–4 multi-object
  datasets (256² is left out for compute reasons and the omission is stated).
- **Fine-tuning replication.** Paper §4.12: freeze image encoder and prompt encoder, train only the mask
  decoder with box prompts, 20 epochs, lr 1e-4, batch 2, on a subset of targets — cheap because embeddings
  are cached.

---

## 9. Critical analysis — four methodological criticisms, each quantified

This is the section that turns a reproduction into a follow-up publication. Each criticism is stated,
implemented, and measured.

**9.1 The mask-matching rule is an oracle.** Selecting, among SAM's *N* output masks, the one with the
highest Dice **against the ground truth** uses the GT at inference time. Every number in the paper is
therefore an upper bound that is unattainable in deployment. *We quantify the oracle gap* by re-scoring the
identical predictions with SAM's own predicted-IoU head and with the "first/largest mask" rule. Cost: zero
extra inference — all masks are stored. Expected to be the single most impactful correction, especially for
S1/S2 where *N* is large.

**9.2 Slices are treated as i.i.d.** The Spearman partial correlation over 191,779 structures treats
adjacent slices of the same volume as independent samples, which inflates significance and understates
variance. *We re-estimate* with a mixed-effects model (random intercept per patient/volume, nested in
dataset) and report how much the effective sample size and the CIs change.

**9.3 Aggregation is confounded by dataset size.** Means over object–modality pairs are dominated by the
largest datasets (TotalSegmentator alone is ~307k images). *We report* both the paper's aggregation and a
target-balanced aggregation, and show how much the headline ordering moves.

**9.4 Dice/HD are the wrong instruments for thin, complex structures.** For retinal vessels and neural
structures, Dice is dominated by boundary pixels and HD is dominated by single outliers. *We add* NSD and
Boundary IoU and check whether the attribute-dependence conclusion (H1) is metric-dependent — if the
"boundary complexity hurts" effect vanishes under a boundary-aware metric, that is a substantive finding
about the original conclusion.

**Improvements to the method / implementation** (also required by the rubric): embedding cache reuse across
all strategies and jitter seeds (already in the upstream repo for box, extended by us to all six strategies
and to the perturbation study — an ~*n*× saving); fully config-driven target definitions; deterministic
seeded prompt generation so the entire study is bit-reproducible.

---

## 10. Compute plan — NHR@FAU TinyGPU

TinyGPU partitions: `work` (RTX 2080 Ti 11 GB / RTX 3080 10 GB), `rtx3080`, `v100` (32 GB), `a100` (40 GB);
**max walltime 24 h** on all of them; node-local `$TMPDIR` ≥ 1.8 TB. Data lives in `$WORK`, staged to
`$TMPDIR` at job start.

**Architecture that makes the quota work:** extract every image embedding **once per model**, cache as fp16
`.npy` in `$WORK`, and let all six strategies, all jitter seeds, and the fine-tuning reuse it. Only
*everything* mode and the encoder passes are expensive; prompt encoding + mask decoding are ~0.01 s.

| Stage | Estimate |
|---|---|
| Embedding extraction, 6 models × ~8k images | ~6 h |
| *Everything* mode S1 (ViT-B + ViT-H) | ~7 h |
| S2–S6 decoding, all models | < 1 h |
| Point-count ablation (≤128²) | ~4 h |
| Prompt-jitter study (3 seeds × 3 levels × 5 strategies) | ~1 h |
| Fine-tuning replication (decoder only, cached embeddings) | ~4 h |
| Additional data (OCT/PET/mammography), 6 models | ~8 h |
| Debug + reruns (×2 contingency) | ~30 h |
| **Total** | **≈ 60–80 GPU-hours** |

Everything is chunked into Slurm **array jobs** sized to well under the 24 h limit, with per-chunk resume
so a preempted job costs one chunk, not one run. ViT-H and Hiera jobs go to `a100`/`v100`; ViT-B jobs run
fine on `work`/`rtx3080`.

Storage: fp16 embeddings are 256×64×64×2 B = 2 MB/image → ~16 GB per model, ~60 GB total in `$WORK`.
Raw + preprocessed data ~150 GB. Well within normal `$WORK` quota, but to be confirmed in Phase 0.

---

## 11. Deliverables

### 11.1 Report (~10 pages, English, `elsarticle` / MICCAI LaTeX)

1. **Introduction & Motivation** — why zero-shot foundation-model segmentation matters clinically; why the
   *"?"* in the paper's title is still open in 2026.
2. **Related Work** — SAM lineage (SAM → SAM 2 → SAM 3); medical adaptations (MedSAM, SAM-Med2D, MedSAM2,
   MedicoSAM); text-promptable medical segmentation (MedSAM3, VoxTell); prior SAM-in-medicine benchmarks and
   what each of them missed.
3. **Method** — brief SAM architecture; the six prompting strategies; mask matching; the attribute
   descriptors (incl. elliptic Fourier descriptors).
4. **Reproduction** — scope and sampling protocol; results vs. published tables; agreement/deviation analysis.
5. **Additional experiment** — the 2×2 model design, the out-of-domain modalities, H2–H4 with their tests.
6. **Results & Discussion** — organised by hypothesis, not by table.
7. **Critical discussion of limitations** — of the original paper (§9) *and* of our own study (subset size,
   2D-only, single-annotator GT, contamination that we could not resolve).
8. **Suggestions for improvement** — method, implementation, evaluation.
9. **Conclusion**.

### 11.2 Repository

```
sam-medical-revisited/
├── README.md                  # results up front, figures, one-command smoke test
├── configs/                   # datasets, models, experiments — YAML, no hardcoded paths
├── src/
│   ├── data/                  # per-dataset adapters → unified PNG + mask + metadata
│   ├── prompts/               # S1–S6 construction (paper-faithful), unit-tested
│   ├── models/                # thin wrappers over upstream repos; zero model code
│   ├── eval/                  # DICE/JAC/HD/NSD/BoundaryIoU, mask matching (oracle + score)
│   ├── attributes/            # size, aspect ratio, intensity difference, EFD Fourier order
│   └── stats/                 # partial correlation, mixed effects, bootstrap CIs
├── scripts/                   # Slurm array-job templates for TinyGPU
├── results/                   # committed CSVs — every figure regenerable from these
├── report/                    # LaTeX sources
└── tests/
```

Upstream models enter as pinned dependencies / submodules. Nothing from `segment_anything`, `sam2`,
`MedSAM`, or `SAM-Med2D` is copied or rewritten.

---

## 12. Schedule (6 weeks, buffer included)

| Week | Work package | Exit criterion |
|---|---|---|
| **0–1** | Phase 0: repo skeleton, env on TinyGPU, checkpoint downloads (incl. SAM 3 access request), **data acquisition + contamination audit**, preprocessing, harness skeleton, smoke test on ACDC | One dataset runs end-to-end through S1–S6 with sane Dice; final dataset list frozen |
| **2** | Full reproduction: SAM ViT-B/H × S1–S6 × all targets; metrics + attribute extraction | `results/reproduction.csv` complete |
| **3** | Reproduction analysis (Tables 2/3/6, Figs. 6/10 equivalents); jitter study; fine-tuning replication; §9.1 oracle-gap analysis | Agreement/deviation analysis written; H1 decided |
| **4** | Additional experiment: MedSAM / SAM-Med2D / MedSAM2 / SAM 2.1 on the reproduction subset **and** on OCT/PET/mammography | `results/extension.csv` complete; H2–H4 decided |
| **5** | Statistics, all figures, report draft v1, README | Full 10-page draft |
| **6** | Buffer: reruns, polish, repo hygiene, final report | Final PDF + tagged release |

**Sequencing rule:** the additional experiment is fully specified in week 1 but only *run* after the
reproduction is validated in week 3 — a broken harness would invalidate both.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Data access friction (DUA, registration, TCIA bulk downloads) | Phase-0 hard cutoff; MedSegBench as the drop-in replacement; dataset list frozen end of week 1 |
| Contamination audit inconclusive → H3 weakened | Tier-A modalities (OCT/PET/mammography) are out-of-domain for *every* candidate model regardless of the audit's outcome; H3 survives on Tier A alone |
| TinyGPU queue/quota pressure | Embedding cache + array jobs + resume; ViT-B on the crowded-but-cheap partitions, ViT-H/Hiera on `a100`/`v100`; contingency already in §10 |
| 24 h walltime kills long jobs | Every stage is chunked and resumable by construction |
| Reproduction numbers deviate from the paper | Deviation is a **result**, not a failure — the sampling protocol is pre-registered and the deviation analysis is a report section |
| Scope creep (SAM 3, text prompts) | Explicitly a stretch goal; cut first if week 4 slips |

---

## 14. Open items to resolve in Phase 0

1. Confirm TinyGPU account status, `$WORK` quota, and available module stack / conda policy.
2. Verify download feasibility for every Tier-A dataset (OCT, PET, mammography) — decide within week 1.
3. Extract the **actual** training-corpus lists of MedSAM / SAM-Med2D / MedSAM2 from primary sources.
4. Request Meta SAM 3 licence access on Hugging Face (keeps the stretch goal alive at zero cost).
5. Confirm whether the report must also be submitted to the chair, or is purely a portfolio artefact.
6. Decide the repository name and whether it is published under the user's personal GitHub account.
