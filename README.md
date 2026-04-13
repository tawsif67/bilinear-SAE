# CTCR-SAE: Cross-Turn Conjunction Residuals for Sleeper-Style Jailbreaks

This repository contains a modular research pipeline for testing the claim:

> We identify cross-turn conjunctions as a distinct internal mechanism of sleeper-style jailbreaks. Unlike ordinary multi-turn jailbreaks, which produce smooth representational drift, sleeper triggers can remain weakly detectable until multiple latent conditions are jointly present. We isolate this mechanism with a Cross-Turn Conjunction Residual, `h(A+B)-h(A)-h(B)+h(0)`, and sparse-code that residual with bilinear features for detection and localized intervention.

The code is organized as a reproducible experiment stack rather than a single prototype script. It builds public-data and constructed-data benchmark groups, trains a LoRA-adapted subject model, extracts hidden-state trajectories, trains sparse autoencoders, computes CTCR residuals from matched sleeper controls, evaluates trajectory and sparse-feature baselines, runs mechanistic conjunction diagnostics, and exports paper-style metrics, figures, and LaTeX tables.

## Core Idea: CTCR

The main method is **CTCR-SAE**, not just "bilinear SAE plus a fuser."

For each matched sleeper control block, the code computes:

```text
CTCR = h(A+B) - h(A only) - h(B only) + h(neither)
```

This residual isolates the hidden-state component that appears only when the two distributed trigger conditions jointly occur. A bilinear SAE is then trained on these residual vectors. The resulting method, `ctcr_residual_bilinear`, asks whether sleeper-style attacks are better explained by sparse interaction residuals than by ordinary drift or single-feature activation.

## What This Repo Does

The pipeline compares thirteen methods:

1. Dense linear probe
2. Dense MLP probe
3. Turn-concatenated trajectory MLP probe
4. Mean-difference ablation baseline
5. RepE-style projection baseline
6. Activation probe baseline
7. Trajectory-only detector
8. Linear SAE only
9. Linear SAE + trajectory
10. Bilinear SAE only
11. Bilinear SAE + trajectory
12. CTCR residual bilinear SAE
13. Full fused method using linear sparse score, bilinear sparse score, trajectory score, and pairwise interactions

The main question is whether sleeper-style attacks produce sparse **conjunction residuals** that ordinary trajectory and linear sparse methods miss.

## Fixed Model Choices

The experiment intentionally uses fixed model IDs:

- Subject model: [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Judge model: [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)

The judge model is not silently swapped. If Gemma 3 access is unavailable, the run fails with an explicit error.

## Dataset Groups

The pipeline keeps real public data separate from constructed sleeper-trigger data.

### Group 1: Ordinary Harmful / Benign

- Source: [`allenai/wildjailbreak`](https://huggingface.co/datasets/allenai/wildjailbreak)
- Uses:
  - `vanilla_harmful`
  - `adversarial_harmful`
  - `vanilla_benign`
  - `adversarial_benign`
- Split: 70/10/20
- Notes:
  - This dataset is gated.
  - You must accept the AI2 Responsible Use Guidelines on Hugging Face.

### Group 2: Ordinary Multi-Turn Jailbreaks

- Default source: [`ScaleAI/mhj`](https://huggingface.co/datasets/ScaleAI/mhj)
- Split: 60/20/20
- Notes:
  - Echo Chamber is not configured as a ready-to-load dataset by default.
  - `ScaleAI/mhj` is used as the public multi-turn substitute.
  - The loader reads the relevant MHJ conversation CSV directly because the Hugging Face auto-builder can mix incompatible files from the dataset repo.

### Group 3: Constructed Sleeper-Style Distributed Triggers

- Source label: `constructed_sleeper`
- Six trigger families:
  - year/context switch
  - deployment trigger
  - handoff override
  - policy exception
  - latent instruction reveal
  - delayed escalation
- Matched controls:
  - both latent conditions present
  - only condition A
  - only condition B
  - neither condition
  - shuffled turns
  - single-turn compressed controls

Each sleeper block stores a stable `residual_group` so `A+B`, `A`, `B`, and `0` controls remain together across deterministic splits. This is what makes CTCR computation possible without leakage. The constructed sleeper data uses redacted/proxy unsafe-compliance labels and does not generate actionable harmful completions.

## Repository Layout

```text
.
  main.py
  requirements.txt
  configs/
    default.yaml
    debug.yaml
  data/
    loaders.py
    wildjailbreak.py
    multiturn_jailbreak.py
    sleeper_builder.py
    splits.py
  models/
    subject.py
    sae.py
    trajectory.py
    baselines.py
    fusion.py
    interventions.py
    ctcr.py
  train/
    train_lora.py
    train_sae.py
    train_ctcr.py
    train_fuser.py
  eval/
    judge.py
    generate.py
    metrics.py
    ablations.py
    significance.py
    mechanistic_taxonomy.py
    conjunction_tests.py
    causal_interventions.py
    null_controls.py
    transfer.py
    layer_sweep.py
  plots/
    style.py
    main_figures.py
    appendix_figures.py
    latex_tables.py
  utils/
    seed.py
    io.py
    logging_utils.py
    config_utils.py
    dependencies.py
    gpu_memory.py
  scripts/
    check_imports.cmd
    check_imports.ps1
    run_debug.cmd
    run_debug.ps1
    run_full.cmd
    run_full.ps1
  outputs/
```

## Prerequisites

Recommended environment:

- Linux or WSL for full GPU runs
- Python 3.10 or 3.11
- CUDA GPU for the full experiment
- A100-class GPU recommended for full runs
- Hugging Face account with access to:
  - `google/gemma-3-4b-it`
  - `allenai/wildjailbreak`

The full config intentionally fails early if CUDA is absent.

## Install

Clone:

```bash
git clone https://github.com/tawsif67/bilinear-SAE.git
cd bilinear-SAE
```

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
```

Authenticate Hugging Face:

```bash
huggingface-cli login
```

Before running, accept access terms in the browser:

- [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)
- [`allenai/wildjailbreak`](https://huggingface.co/datasets/allenai/wildjailbreak)

## Quick Health Check

Run this before launching a cluster job:

```bash
python -m py_compile main.py
```

On Windows:

```cmd
scripts\check_imports.cmd
```

This checks imports and config parsing. It does not download models or run the full experiment.

## Run Debug Mode

Debug mode uses the same fixed model IDs but smaller subsamples, fewer training steps, and one seed.

```bash
python main.py --config configs/debug.yaml
```

Windows helper:

```cmd
scripts\run_debug.cmd
```

Debug mode is still a real model/data run. It can fail if:

- Hugging Face access is missing
- required Python packages are missing
- the machine cannot load the selected models
- CUDA is needed for your selected config

## Run Full Mode

```bash
python main.py --config configs/default.yaml
```

Windows helper:

```cmd
scripts\run_full.cmd
```

For cluster usage:

```bash
git pull
python -m pip install -r requirements.txt
huggingface-cli login
python main.py --config configs/default.yaml
```

## Configs

`configs/debug.yaml`:

- smaller subsamples
- one seed
- shorter SAE/fuser training
- intended for integration testing

`configs/default.yaml`:

- subject model: `Qwen/Qwen2.5-3B-Instruct`
- judge model: `google/gemma-3-4b-it`
- train/val/test/OOD subsampling for larger runs
- seeds: `[42, 123, 456]`
- LoRA, SAE, trajectory, fuser, and evaluation settings

## Outputs

Each run creates:

```text
outputs/run_<timestamp>/
  config_used.yaml
  splits/
  checkpoints/
  raw_metrics/
  figures/
  tables/
  logs/
  human_eval_samples/
```

Important files:

- `raw_metrics/metrics.csv`
- `raw_metrics/metrics.json`
- `raw_metrics/feature_metrics.csv`
- `raw_metrics/intervention_locality.csv`
- `raw_metrics/mechanistic_taxonomy.csv`
- `raw_metrics/conjunction_controls.csv`
- `raw_metrics/ctcr_residuals.csv`
- `raw_metrics/causal_interventions.csv`
- `raw_metrics/null_controls.csv`
- `raw_metrics/transfer_results.csv`
- `raw_metrics/layer_summary.csv`
- `tables/main_results.csv`
- `tables/significance.csv`
- `tables/dataset_summary.csv`
- `tables/config_summary.csv`
- `tables/table_main_results.tex`
- `tables/table_significance.tex`
- `figures/fig1_threshold_profiles.pdf`
- `figures/fig2_roc_pareto.pdf`
- `figures/fig3_mechanistic.pdf`
- `figures/fig4_generalization.pdf`
- `figures/fig5_mechanistic_claim.pdf`

Figures are saved as both PDF and PNG where applicable.

## Main Experiments

The pipeline implements:

- Differential comparison: ordinary multi-turn jailbreaks vs sleeper-style triggers
- Linear vs bilinear vs trajectory ablations
- Cross-turn conjunction controls
- Causal pair-ablation style diagnostics
- Family-holdout generalization
- Safety/utility/latency tradeoffs across thresholds
- Mechanistic taxonomy: trajectory-drift-dominant vs conjunction-dominant vs mixed
- Conjunction necessity controls: both conditions, only A, only B, neither, shuffled, compressed
- Transfer tests across ordinary, sleeper, and OOD slices
- Null controls for shuffled-label bilinear advantage
- Layer-localization summaries for configured intercept layers

## Metrics

For each method, seed, threshold, and data group, the pipeline saves:

- ASR
- ASR reduction vs undefended subject model
- clean accuracy / safe refusal correctness
- false-positive intervention rate
- defended WikiText-2 perplexity
- throughput in tokens/sec
- AUROC
- AUPRC
- family-holdout performance
- OOD trigger performance

For sparse features, it also exports:

- top feature activation by turn
- decoder norms
- d-prime / selectivity
- pairwise synergy score
- intervention locality

Mechanistic analysis files add:

- `mechanistic_category`
- `bilinear_gain_over_single_and_traj`
- `trajectory_gain_over_sparse`
- `conjunction_selectivity`
- `order_sensitivity`
- `cross_turn_advantage`
- `null_corrected_bilinear_auc`
- `transfer balanced_accuracy`
- causal proxy effect and locality ratios

## Common Errors

### Gemma 3 access error

If you see:

```text
Cannot access gated repo google/gemma-3-4b-it
```

Fix:

1. Open [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)
2. Accept the model terms
3. Run `huggingface-cli login` on the machine running the job

### WildJailbreak access error

If loading `allenai/wildjailbreak` fails with a gated access error:

1. Open [`allenai/wildjailbreak`](https://huggingface.co/datasets/allenai/wildjailbreak)
2. Accept the AI2 Responsible Use Guidelines
3. Run `huggingface-cli login`

### Missing dependency error

If you see:

```text
Missing required runtime dependencies
```

Fix:

```bash
python -m pip install -r requirements.txt
```

### Judge sanity check failure

The run performs a judge sanity check before the main experiment. If this fails, do not trust the metrics. Check:

- `transformers` version
- Gemma access/authentication
- whether the installed model files are complete
- whether the run is using the latest repository commit

## Safety Notes

This code evaluates harmful-request refusal/compliance behavior, but the constructed sleeper dataset avoids actionable harmful completions. The project is intended for safety research, representation analysis, and controlled evaluation. Do not use outputs to build or improve harmful systems.

## Known Limitations

- The sleeper-trigger dataset is constructed and deterministic, not a public audited benchmark.
- Judge-only compliance metrics are not a substitute for human annotation in a paper submission.
- Some intervention paths use detector-driven safe-refusal gating rather than full hidden-state causal editing.
- The repository now includes hidden-state SAE ablation hooks and causal proxy diagnostics, but full response-generation causal intervention sweeps are still expensive and should be audited before paper claims.
- Public dataset schemas and access rules can change. Loaders fail loudly rather than silently fabricating fallback data.
- Full experiment quality depends on GPU memory, Hugging Face access, package versions, and reproducible cluster setup.

## Citation

No paper citation is provided yet. If this repository becomes part of a paper submission, add the final BibTeX entry here.
