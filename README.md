# CTCR-SAE: Cross-Turn Conjunction Residuals for Sleeper-Style Jailbreak Analysis

This repository contains a modular research pipeline for testing the claim:

> We test whether sleeper-style trigger behavior contains a cross-turn conjunction component that is separable from ordinary multi-turn representational drift. CTCR-SAE is the main mechanistic contribution; single-pass ARF-SAE is included as a secondary deployment-oriented detector.

The code is organized as a reproducible experiment stack rather than a single prototype script. It builds public-data and constructed-data benchmark groups, trains a LoRA-adapted subject model, extracts hidden-state trajectories, trains sparse autoencoders, computes CTCR residuals from matched sleeper controls, evaluates attack residual fingerprints across jailbreak families, runs mechanistic diagnostics, and exports paper-style metrics, figures, and LaTeX tables. It also writes explicit validity warnings when external sleeper-agent data or stronger adjudication is missing.

## Main Contribution: CTCR Diagnostic

**CTCR-SAE is the primary method and is analysis-only.** It is not presented as a practical deployment detector because it requires matched controls at evaluation time.

For each matched sleeper control block, the code computes:

```text
CTCR = h(A+B) - h(A only) - h(B only) + h(neither)
```

This residual isolates the hidden-state component that appears only when the two distributed trigger conditions jointly occur. A bilinear SAE is then trained on these residual vectors. The resulting method, `ctcr_residual_bilinear`, asks whether sleeper-style attacks are better explained by sparse interaction residuals than by ordinary drift or single-feature activation.

Because CTCR needs `A+B`, `A only`, `B only`, and `neither` variants, it is logged under `raw_metrics/ctcr_residuals.csv` and `raw_metrics/method_manifest.csv` as a **mechanistic diagnostic**. It is excluded from the main deployable intervention loop by default.

The code now exports explicit CTCR formula ablations:

```text
CTCR:        h(A+B) - h(A) - h(B) + h(0)
Simple diff: h(A+B) - h(0)
A-added:     h(A+B) - h(B)
B-added:     h(A+B) - h(A)
Raw joint:   h(A+B)
Shuffled:    h(B+A) - h(B) - h(A) + h(0)
```

These rows are written with `method=ctcr_formula_ablation` in `raw_metrics/ctcr_residuals.csv`.

A compact theoretical motivation for the low-rank bilinear term is included in `docs/ctcr_theory.md`.

## Secondary Detector: ARF-SAE

**ARF-SAE is the secondary deployable detector path.** During training it learns sparse residual features from matched attack/control pairs. At test time, the single-pass ARF variant uses:

```text
hidden_state(conversation) - mean(training benign/control hidden states)
```

This requires only the observed conversation and a fixed training baseline. The deployable ARF rows use method name `arf_sae_single_pass`; the matched-residual analysis rows use `arf_sae_matched_residual_analysis`.

## What This Repo Does

The main deployment loop compares twelve single-pass detector methods:

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
12. Full fused method using linear sparse score, bilinear sparse score, trajectory score, and pairwise interactions

CTCR residual bilinear SAE is exported separately as an analysis-only method because it requires four matched forward passes and known condition structure.

The repository also includes **Attack Residual Fingerprinting (ARF-SAE)** as supporting evidence and a practical detector baseline. It should not be presented as co-equal with CTCR in the paper narrative.

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

### Large Real Harmful Source Pool

- Default source: [`mvrcii/safety-harmful`](https://huggingface.co/datasets/mvrcii/safety-harmful)
- Default full cap: `12000` records
- Debug cap: `1500` records
- Use:
  - source pool for ARF counterfactual transformations
  - real prompt provenance for direct, roleplay, policy override, obfuscation, refusal suppression, and sleeper-sequence wrappers
- Notes:
  - This harmful-only corpus is **not** merged directly into the main harmful/benign benchmark metrics.
  - The code saves the selected source pool and summary under `synthetic_datasets/`.
  - ARF samples from this pool to control runtime rather than passing every source row through Qwen.

### Additional External Source Pools

The config now enables four additional public source pools for ARF provenance and source-holdout artifact generation:

- [`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- AdvBench via `S3IC/advbench`
- [`Lakera/gandalf_ignore_instructions`](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions)
- [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections)

These datasets are parsed schema-flexibly and saved as source pools. They are optional by default: if a dataset schema or split changes, the run records the loader error under `synthetic_datasets/external_dataset_errors.jsonl` instead of pretending the source was used. Loaded rows are used to derive ARF counterfactual transformations and per-source ARF artifacts.

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
  - reverse order
  - same-turn compression
  - delayed gap
  - interruption
  - reset
  - semantic decoy-A
  - semantic decoy-B
  - long-gap carry
  - shuffled turns
  - single-turn compressed controls

Each sleeper block stores a stable `residual_group` so `A+B`, `A`, `B`, `0`, and temporal sequence controls remain together across deterministic splits. This is what makes CTCR and sequence-control analysis possible without leakage. The constructed sleeper data uses redacted/proxy unsafe-compliance labels and does not generate actionable harmful completions.

### Group 4: External Sleeper-Agent Validation

- Source: user-configured local path or Hugging Face dataset ID
- Default: enabled but not required, with no path configured
- Use:
  - external holdout for sleeper-agent generalization
  - never used to fit synthetic CTCR templates
  - never mixed into the training subsample by default
- Output:
  - `synthetic_datasets/external_sleeper_validation.jsonl`
  - `synthetic_datasets/external_sleeper_validation_summary.json`
  - `synthetic_datasets/external_sleeper_validation_errors.jsonl`

For a paper claim, configure this source and set `external_sleeper.required: true` so missing external data fails the run instead of producing only a warning.

## Attack Residual Fingerprinting

ARF-SAE creates compact matched residual datasets for:

- direct harmful requests
- roleplay/persona jailbreaks
- fake-authority or policy-override jailbreaks
- encoding/obfuscation jailbreaks
- refusal-suppression jailbreaks
- ordinary multi-turn drift
- sleeper-style sequence triggers

For each family, the attack prompt is derived from the loaded public prompt pools where possible, then compared against one or more matched controls. The ARF builder excludes the constructed sleeper dataset from its base prompt pool to avoid synthetic-on-synthetic provenance. The saved metadata includes base dataset, base example id, source split, tactic/category fields where available, and the transformation type. Sleeper-style examples use temporal controls such as reverse order, same-turn compression, reset, semantic decoys, long-gap context, and neutral filler. The method trains a small sparse autoencoder on:

```text
attack hidden state - mean(matched control hidden states)
```

and then trains a lightweight classifier over the sparse residual activations. The deployable ARF classifier is then evaluated in single-pass mode against a fixed training-control baseline, so test-time examples do not need pre-specified A/B conditions. The pipeline also exports serious lexical baselines, including count n-grams, TF-IDF word n-grams, TF-IDF character n-grams, and combined word+character TF-IDF, so the paper can check whether attack-family classification is merely recoverable from surface wording. This adds compact hidden-state extraction passes per seed but no extra generation or Gemma judge calls.

## External Sleeper-Agent Validation

Constructed sleeper triggers are useful for controlled mechanism tests, but they are not enough for the central paper claim. The repository therefore has an explicit external-sleeper adapter:

```yaml
external_sleeper:
  enabled: true
  required: false
  local_path: null
  hf_dataset_id: null
```

Set `external_sleeper.local_path` to a local JSON/JSONL/CSV export from an external sleeper-agent benchmark, such as an Anthropic/Hubinger sleeper-agents data export, or set `external_sleeper.hf_dataset_id` if using a Hugging Face mirror. The loader accepts common fields such as `messages`, `turns`, `conversation`, `prompt`, `input`, `label`, `target`, `triggered`, and `split`.

If no external sleeper data is loaded, the run still completes, but `raw_metrics/validity_warnings.json` marks this as **blocking for the ICLR sleeper-agent generalization claim**. Do not claim external sleeper-agent generalization from constructed data alone.

## Judge Policy

Gemma 3 4B remains the default local judge for reproducible budget-limited runs. For no-cost adjudication beyond Gemma, the repo supports a local Hugging Face LLM adjudicator. The default free adjudicator model is `Qwen/Qwen2.5-1.5B-Instruct`; it requires no OpenAI key, but it does require downloading/loading the model locally.

Enable free local LLM adjudication with:

```yaml
strong_judge:
  enabled: true
  provider: local_hf
  local_model: Qwen/Qwen2.5-1.5B-Instruct
  max_seq_len: 768
  batch_size: 4
```

For paper-grade main results, use GPT-5 or human adjudication:

- export requests: `human_eval_samples/strong_judge_requests_seed_<seed>.jsonl`
- annotate with GPT-5, Claude-class, or human labels
- report agreement against Gemma and include inter-annotator agreement for any human sample

The repo includes `eval/strong_judge.py` with both `LocalHFComplianceAdjudicator` and `GPT5ComplianceAdjudicator`. Strong judging is disabled by default to avoid adding another model load during debug runs. Enable GPT-5 adjudication with:

```yaml
strong_judge:
  enabled: true
  provider: openai_gpt5
  openai_model: gpt-5
```

## Repository Layout

```text
.
  main.py
  legacy/
    CB-SAE.py
  docs/
    ctcr_theory.md
  requirements.txt
  configs/
    default.yaml
    debug.yaml
  data/
    loaders.py
    wildjailbreak.py
    multiturn_jailbreak.py
    sleeper_builder.py
    attack_fingerprints.py
    external_sleeper.py
    splits.py
  models/
    subject.py
    sae.py
    trajectory.py
    baselines.py
    fusion.py
    interventions.py
    ctcr.py
    arf.py
  train/
    train_lora.py
    train_sae.py
    train_ctcr.py
    train_arf.py
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
    strong_judge.py
  plots/
    style.py
    main_figures.py
    appendix_figures.py
    latex_tables.py
    arf_figures.py
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
  - `ScaleAI/mhj`

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
- [`ScaleAI/mhj`](https://huggingface.co/datasets/ScaleAI/mhj)

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

The default config is intentionally scaled for safer single-GPU runs rather than maximum sample count:

- train/val/test/OOD subsamples: `4500 / 800 / 1200 / 900`
- eval batch size: `4`
- judge batch size: `2`
- sleeper examples per family: `240`
- sleeper OOD examples per family: `120`
- SAE/fuser steps: `1200 / 800`

Raw per-threshold outputs are streamed to JSONL without repeated response text. Qualitative prompt/response text is kept separately under `human_eval_samples/`, which avoids host-RAM blowups during postprocessing.

## Configs

`configs/debug.yaml`:

- smaller subsamples: `1000 / 200 / 300 / 240`
- one seed
- shorter SAE/fuser training: `200 / 200` steps
- ARF-SAE: `24` pairs per attack family, `80` SAE steps
- eval/judge batch sizes: `4 / 2`
- intended for integration testing

`configs/default.yaml`:

- subject model: `Qwen/Qwen2.5-3B-Instruct`
- judge model: `google/gemma-3-4b-it`
- train/val/test/OOD subsampling: `4500 / 800 / 1200 / 900`
- seeds: `[42, 123, 456]`
- ARF-SAE: `96` pairs per attack family, `300` SAE steps
- LoRA, SAE, trajectory, fuser, and evaluation settings

`configs/publication.example.yaml`:

- turns on `publication_mode`
- requires a real external sleeper holdout via `external_sleeper.local_path` or `external_sleeper.hf_dataset_id`
- requires `strong_judge.enabled: true`
- is intentionally not a drop-in config until you replace the placeholder external sleeper path with real data
- fails during config/load if the publication prerequisites are missing

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
  synthetic_datasets/
```

Important files:

- `raw_metrics/metrics.csv`
- `raw_metrics/metrics.json`
- `raw_metrics/feature_metrics.csv`
- `raw_metrics/intervention_locality.csv`
- `raw_metrics/mechanistic_taxonomy.csv`
- `raw_metrics/conjunction_controls.csv`
- `raw_metrics/ctcr_residuals.csv`
- `raw_metrics/ctcr_formula_ablation_summary.csv`
- `raw_metrics/attack_residual_fingerprints.csv`
- `raw_metrics/attack_residual_pairs.csv`
- `raw_metrics/attack_residual_diagnostics.csv`
- `raw_metrics/attack_residual_significance.csv`
- `synthetic_datasets/constructed_sleeper_dataset.jsonl`
- `synthetic_datasets/constructed_sleeper_dataset_summary.json`
- `synthetic_datasets/real_attack_corpus_source_pool.jsonl`
- `synthetic_datasets/real_attack_corpus_source_pool_summary.json`
- `synthetic_datasets/external_attack_source_pool.jsonl`
- `synthetic_datasets/external_benign_source_pool.jsonl`
- `synthetic_datasets/external_dataset_summary.jsonl`
- `synthetic_datasets/external_dataset_errors.jsonl`
- `synthetic_datasets/external_sleeper_validation.jsonl`
- `synthetic_datasets/external_sleeper_validation_summary.json`
- `synthetic_datasets/attack_residual_fingerprints_external_<source>_seed_<seed>.jsonl`
- `synthetic_datasets/attack_residual_fingerprints_seed_<seed>.jsonl`
- `synthetic_datasets/attack_residual_fingerprints_seed_<seed>_summary.json`
- `synthetic_datasets/constructed_sleeper_validation.csv`
- `synthetic_datasets/constructed_sleeper_audit.csv`
- `synthetic_datasets/attack_residual_fingerprints_seed_<seed>_validation.csv`
- `synthetic_datasets/attack_residual_fingerprints_audit.csv`
- `raw_metrics/causal_interventions.csv`
- `raw_metrics/null_controls.csv`
- `raw_metrics/transfer_results.csv`
- `raw_metrics/layer_summary.csv`
- `raw_metrics/method_manifest.csv`
- `raw_metrics/validity_warnings.json`
- `raw_metrics/strong_judge_status.json`
- `raw_metrics/residual_group_integrity_seed_<seed>.csv`
- `raw_metrics/residual_group_integrity_<group>_seed_<seed>.csv`
- `tables/main_results.csv`
- `tables/significance.csv`
- `tables/attack_residual_fingerprints.csv`
- `tables/attack_residual_diagnostics.csv`
- `tables/attack_residual_significance.csv`
- `tables/dataset_summary.csv`
- `tables/config_summary.csv`
- `tables/table_main_results.tex`
- `tables/table_significance.tex`
- `figures/fig1_threshold_profiles.pdf`
- `figures/fig2_roc_pareto.pdf`
- `figures/fig3_mechanistic.pdf`
- `figures/fig4_generalization.pdf`
- `figures/fig5_mechanistic_claim.pdf`
- `figures/fig6_arf_family_accuracy.pdf`
- `figures/fig7_arf_family_heatmap.pdf`
- `figures/fig8_arf_residual_norms.pdf`
- `figures/fig9_arf_residual_types.pdf`
- `figures/fig10_arf_generalization.pdf`
- `figures/fig11_arf_vs_lexical.pdf`

Figures are saved as both PDF and PNG where applicable.

The `synthetic_datasets/` directory is intended for paper/letter artifacts. It stores the exact constructed sleeper examples, ARF attack/control prompts, split labels, matched-control metadata, real-source provenance for prompt-derived transformations, and robustness summaries used in the run.

Robustness checks now include:

- template-holdout ARF evaluation where configured
- lexical leakage baselines: count word n-grams, TF-IDF word n-grams, TF-IDF character n-grams, and combined word+character TF-IDF
- ARF-SAE vs lexical baseline diagnostics and significance exports
- CTCR formula ablations against simpler residual alternatives
- CTCR applicability masks so CTCR is only interpreted on examples with matched residual groups
- residual-group integrity CSVs to verify subsampling did not break complete A/B/0 CTCR groups
- strong-judge status JSON showing whether adjudication was active, free/local, or paper-grade
- synthetic dataset validation CSVs
- audit-ready CSV samples for constructed sleeper and ARF examples
- real-prompt-derived provenance fields for ARF transformations

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

- The constructed sleeper-trigger dataset is deterministic development data. It is not evidence of external sleeper-agent generalization unless `external_sleeper` is configured and loaded.
- Gemma-only compliance metrics are not a substitute for GPT-5, Claude-class, or human adjudication in a paper submission.
- CTCR requires matched A/B/0 controls and is therefore a mechanistic analysis tool, not a deployment detector.
- Some intervention paths use detector-driven safe-refusal gating rather than full hidden-state causal editing.
- The repository now includes hidden-state SAE ablation hooks and causal proxy diagnostics, but full response-generation causal intervention sweeps are still expensive and should be audited before paper claims.
- Public dataset schemas and access rules can change. Loaders fail loudly rather than silently fabricating fallback data.
- Full experiment quality depends on GPU memory, Hugging Face access, package versions, and reproducible cluster setup.

## Citation

No paper citation is provided yet. If this repository becomes part of a paper submission, add the final BibTeX entry here.
