# Bilinear Sparse Conjunctions vs Trajectory Drift

This repository implements a modular research pipeline for the claim:

> Ordinary multi-turn jailbreaks are often detectable from trajectory drift alone, but sleeper-agent-style deceptive triggers depend more on cross-turn conjunctive latent conditions. Therefore bilinear sparse features plus trajectory modeling should outperform linear sparse features and trajectory-only detectors specifically on sleeper-style attacks.

The code is intentionally organized as a real experiment pipeline rather than a single prototype script. It builds three benchmark groups, trains a Qwen subject model with LoRA, extracts hidden trajectories, trains linear and bilinear sparse autoencoders, trains a trajectory/fusion detector, evaluates baselines, and exports paper-style figures and LaTeX tables.

## Fixed Models

- Subject model: `Qwen/Qwen2.5-3B-Instruct`
- Judge model: `google/gemma-3-4b-it`

The judge is not silently swapped. If Gemma 3 access is gated, the run fails with an explicit message asking you to accept the Hugging Face access terms and authenticate.

## Data

The pipeline separates real public data from constructed sleeper-trigger data.

- **Group 1: ordinary harmful / benign**
  - Source: `allenai/wildjailbreak`
  - Uses `vanilla_harmful`, `adversarial_harmful`, `vanilla_benign`, and `adversarial_benign`.
  - Split: 70/10/20.
  - This dataset is gated. Accept the AI2 Responsible Use Guidelines before running.

- **Group 2: ordinary multi-turn jailbreaks**
  - Default source: `ScaleAI/mhj`.
  - Echo Chamber is not configured as a ready-to-load dataset by default; if a compatible local path or HF id becomes available, set `echo_chamber_dataset` in config.
  - Split: 60/20/20.

- **Group 3: constructed sleeper-style distributed triggers**
  - Source label: `constructed_sleeper`.
  - Six families: year/context switch, deployment trigger, handoff override, policy exception, latent instruction reveal, delayed escalation.
  - Includes matched controls: both conditions, only A, only B, neither, shuffled turns, and single-turn compressed.
  - Uses redacted/proxy unsafe-compliance targets and does not generate actionable harmful completions.

All split indices and metadata are saved under `outputs/run_<timestamp>/splits/`.

## Installation

```powershell
git clone https://github.com/tawsif67/biliniar-SAE.git
cd biliniar-SAE
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
huggingface-cli login
```

For the full experiment, use a CUDA GPU. The full configuration intentionally fails early if CUDA is absent.

## Running

Debug mode:

```powershell
py -3 main.py --config configs/debug.yaml
```

Windows helper:

```cmd
scripts\run_debug.cmd
```

Full mode:

```powershell
py -3 main.py --config configs/default.yaml
```

Windows helper:

```cmd
scripts\run_full.cmd
```

The debug config uses the same fixed model names but smaller subsamples and one seed.

Repository health check:

```cmd
scripts\check_imports.cmd
```

The `.cmd` wrappers call PowerShell with `-ExecutionPolicy Bypass`, which avoids local script-policy failures without changing the machine-wide execution policy.

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

Tables:

- `main_results.csv`
- `significance.csv`
- `dataset_summary.csv`
- `config_summary.csv`
- `table_main_results.tex`
- `table_significance.tex`
- `table_dataset_summary.tex`
- `table_config_summary.tex`

Figures:

- Figure 1: threshold profiles
- Figure 2: ROC / Pareto tradeoff
- Figure 3: mechanistic sparse-feature figure
- Figure 4: generalization figure
- Appendix A1-A3: latency/utility, feature diagnostics, intervention locality

## Known Limitations

- Constructed sleeper-trigger data is principled and deterministic, but it is not a public audited benchmark.
- The default intervention path uses detector-driven safe refusal gating for evaluation; hidden-state hook support is implemented in the subject wrapper, but some baselines are detector/intervention approximations rather than fully causal editing methods.
- Public dataset schemas can change. The loaders fail loudly rather than fabricating fallback data for gated public benchmarks.
- Human annotation is still required before treating judge-only compliance metrics as paper-grade.
