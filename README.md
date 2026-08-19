# Medical SAM Project

A reproduction of **Huang et al., "Segment anything model for medical images?"**
([Medical Image Analysis 92 (2024) 103061](https://doi.org/10.1016/j.media.2023.103061)),
extended with an experiment the paper does not contain: **where does medical fine-tuning
actually help, and does it survive contact with data the model was not trained on?**

> **Status: in progress.** First results are in, on CHAOS only — 9 object-modality targets,
> 40 patients, abdominal organs, SAM ViT-B and ViT-H, all six prompting strategies, plus the
> object-attribute analysis. Everything below is scoped to that; the remaining datasets, the
> medical-specialised models and the prompt-perturbation study are not finished yet.

---

## The main result so far

Huang et al. score SAM by keeping, out of the several masks it returns, **the one that best
matches the ground truth** (Sec. 3.5). That rule consults the answer at inference time, so
every published number is an upper bound rather than an attainable score. Re-scoring the same
predictions with the model's own quality head — the only signal a real user has — costs no
extra inference, and the difference is large and systematic.

| strategy | oracle (paper's rule) | deployable | gap | 95% CI |
|---|---|---|---|---|
| S1 automatic | 0.569 | 0.069 | 0.500 | [0.461, 0.533] |
| S2 one point | 0.817 | 0.532 | 0.285 | [0.258, 0.312] |
| S3 five points | 0.843 | 0.693 | 0.150 | [0.135, 0.166] |
| S4 five ± five points | 0.865 | 0.730 | 0.135 | [0.122, 0.150] |
| S5 box | 0.933 | 0.909 | 0.024 | [0.021, 0.027] |
| S6 box + point | 0.931 | 0.903 | 0.029 | [0.025, 0.033] |

*DICE over 2409 prompts per model, cluster-bootstrapped over 40 patients.*

**The gap is not a uniform inflation — it scales with how ambiguous the prompt is, and so it
compresses exactly the differences the paper's conclusions are about.**

* *Box beats points* — the paper's main prompting conclusion. Under its own rule the box gains
  +0.120 DICE over a single point; under a deployable rule, **+0.377**. The advantage it
  reports is a third of the real one.
* *More points help* — from S2 to S4 the paper's rule gives +0.048; deployable, **+0.198**.
* *ViT-H beats ViT-B* — **reverses**. Paired on identical prompts, ViT-H is better under the
  oracle rule on S4 (+0.034, significant) and worse in use (−0.076, significant); on S2 the
  oracle rule shows no difference while deployable shows −0.055. Under box prompts the paper's
  conclusion does hold (+0.007). Scale improves the candidates a model generates without
  improving its ability to tell which one is right — and the oracle rule does that half of the
  job for it.
* The gap **grows** with model size under point prompts (S4: 0.080 → 0.190), so it is not an
  artefact of a weak model.

**S1 is a different statement.** Automatic mode names no target, so its quality head ranks
masks by how cleanly they are segmented, not by whether they are the organ in question. Its
0.50 gap is not a ranking failure — it is a measure of how much of the reported
everything-mode performance was supplied by the ground truth. The paper asks "how is semantics
obtained from SAM when there is no GT?" in its discussion, and reports S1 DICE as a
performance figure anyway; 0.569 → 0.069 is the size of that objection.

### The paper's own perception analysis replicates — and the gap tracks difficulty

Partial rank correlations of DICE with object attributes (paper Table 6) come out with every sign and
ordering intact: intensity contrast is the strongest predictor (+0.36…+0.56 against the paper's
+0.45…+0.64), boundary complexity is negative, size weakly positive, modality and aspect ratio
indistinguishable from zero. Complexity is ~3× weaker than published, which is what range restriction
looks like: nine abdominal organs cover far less of the shape spectrum than 191,779 structures including
retinal vessels.

Applying the same analysis to the **oracle gap** — which the paper does not do — shows where its rule
inflates most. For every prompted strategy the gap grows as contrast falls (ρ = −0.13…−0.43) and as
boundaries get more complex (+0.09…+0.13). **The published numbers are least attainable exactly where
segmentation is hardest**, so the oracle rule does not merely lift the scores, it flattens their
dependence on the object — damping the very effect Table 6 is about.

---

## Findings from auditing the paper and its successors

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
so the effect could be measured rather than assumed.

**Measured on CHAOS, the flaw is conservative.** The corrected measure correlates *more* strongly with
DICE than the published one (S5 −0.170 → −0.204, S6 −0.192 → −0.228; sign and significance unchanged), so
the defective criterion attenuates the paper's own conclusion instead of manufacturing it. The
methodological error is real and worth fixing; the conclusion drawn from it survives.

**2. The obvious out-of-domain modalities are not out of domain.**

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
  analysis.py    selection rules, cluster bootstrap, paired model comparison
  cli/           fetch -> prepare -> embed -> predict / everything -> analyse
scripts/     TinyGPU environment setup and Slurm array jobs
tests/       190 tests, all runnable on a laptop without a GPU
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
