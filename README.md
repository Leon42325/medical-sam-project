# Medical SAM Project

A reproduction of **Huang et al., "Segment anything model for medical images?"**
([Medical Image Analysis 92 (2024) 103061](https://doi.org/10.1016/j.media.2023.103061)),
extended with an experiment the paper does not contain: **where does medical fine-tuning
actually help, and does it survive contact with data the model was not trained on?**

> **Status: in progress.** The pipeline, the protocols and the analysis layer are built and
> tested; experiments have not been run yet, so no results are reported below. Findings that
> *are* already established come from auditing the paper and its successors, and each is
> backed by a regression test or a primary source.

---

## Findings so far

**1. The paper's boundary-complexity measure does not measure boundary complexity.**

Huang et al. quantify how convoluted an object's outline is by the elliptic Fourier order
needed to fit it, and this attribute is the strongest correlate of segmentation quality they
report (Table 6). Their search stops when the fit improves by less than 0.1 % between
consecutive orders — which assumes the fit improves monotonically. It does not: an elliptic
Fourier fit improves in a *staircase*, because a shape with roughly *k*-fold symmetry puts
almost all its energy near harmonics *k*±1, leaving orders 2…*k*−2 contributing nothing.

Measured on synthetic *k*-pointed stars, the criterion stops at **order 2** for *k* = 6, 8 and
14, while the contour is actually fitted only at orders **11**, **13** and beyond 25:

| shape | criterion stops at | DICE there | order that truly reaches DICE > 0.97 |
|---|---|---|---|
| 6-pointed star | 2 | 0.75 | 11 |
| 8-pointed star | 2 | 0.75 | 13 |
| 14-pointed star | 2 | 0.75 | not within 25 |

With the order pinned near 2, the published quantity `F_final = F_a + 2·100·(1 − DICE)`
collapses into a low-order fit residual rather than an order — consistent with the values up
to ~180 in the paper's Fig. 14, which no genuine harmonic order reaches. It still ranks shapes
monotonically by complexity, so the qualitative conclusion may hold; but the attribute at the
centre of the paper's perception analysis is mislabelled. Both variants are implemented
(`fourier_order(patience=1)` reproduces the paper, `patience=None` searches to the true order)
so the effect on the correlation analysis can be measured rather than assumed.

**2. Every mask-matching number in the paper is an oracle bound.**

SAM returns several candidate masks per prompt. The paper keeps the one with the highest DICE
*against the ground truth* — which consults the answer at inference time. No deployment can do
this. All candidates and the model's own predicted-IoU scores are therefore retained here, and
the same predictions are re-scored under deployable rules, so the gap can be quantified at the
cost of one groupby rather than a second inference pass.

**3. The obvious out-of-domain modalities are not out of domain.**

The extension originally planned to test medical models on OCT, PET and mammography, chosen
because none appears among COSMOS 1050K's 18 modalities. Auditing MedSAM's supplementary
tables killed that: all three are in its training corpus (AutoPET and HECKTOR, CDD-CESM,
Intraretinal Cystoid Fluid and OCT Images DME). Absence from the paper's dataset implies
nothing about the models being compared to it.

MedSAM does, however, publish its held-out sets, and the evaluation subset straddles both
arms — so the hypothesis was rewritten around the model's own split. Along the way:
SAM-Med2D publishes no dataset-level provenance at all, making a clean comparison against it
impossible for anyone; and it runs at 256×256 against SAM's 1024×1024, so a naive comparison
confounds domain adaptation with a 16-fold reduction in input pixels.

Full audit: [`configs/training_corpora.yaml`](configs/training_corpora.yaml).

---

## What this repository is

No model is reimplemented. SAM, MedSAM and the rest are consumed as upstream dependencies;
what is built here is everything the original release left out — the point-prompt construction
for strategies S2–S4, the perturbation protocol behind Table 8, the object-attribute
extraction, and the statistics — plus the parts a reproduction needs to be trustworthy:
provenance auditing, integrity verification and a pre-registered sampling protocol.

```
configs/     dataset sources, provenance policy, training-corpus audit
src/samed/
  prompts.py     S1-S6 prompt construction and jitter    (not released upstream)
  attributes.py  size, aspect ratio, contrast, elliptic Fourier complexity
  selection.py   oracle vs deployable mask selection
  metrics.py     DICE, JAC, Hausdorff
  data/          preprocessing (paper Sec. 2.2), sampling protocol, verification
  models/        thin adapters over upstream models - no model code
  cli/           fetch -> embed -> predict
scripts/     TinyGPU environment setup and Slurm array jobs
tests/       109 tests, all runnable on a laptop without a GPU
```

Two conventions are load-bearing. Nothing in `samed` imports `torch`, so the analysis layer
and the entire test suite run without a GPU stack. And where the paper is underspecified, the
reading is exposed as a parameter and marked `AMBIGUITY` in the source rather than silently
chosen — the sampling of positive points, the scope of intensity normalisation, the meaning of
the label-area threshold, and whether box jitter translates or reshapes the box.

## Running it

```bash
pip install -e ".[dev]" && pytest
```

On the cluster, once per login shell — `setup_tinygpu.sh` builds the environment but cannot
activate it for its caller, so sourcing this is a separate step:

```bash
bash scripts/setup_tinygpu.sh          # once, ~15 min
source scripts/activate.sh             # once per shell
```

then:

```bash
python -m samed.cli.fetch --all --dry-run
sbatch.tinygpu scripts/slurm/embed.sbatch
sbatch.tinygpu scripts/slurm/predict.sbatch
```

## Licence and data

Code is MIT. Datasets and model weights are **not** redistributed here and carry their own
terms — see [`LICENSE`](LICENSE) for details, [`configs/sources.yaml`](configs/sources.yaml)
for where each dataset comes from, and `data/provenance.json` (written at download time) for
exactly which files were used.
