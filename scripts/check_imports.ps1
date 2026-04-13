$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:MPLCONFIGDIR = Join-Path $root ".matplotlib"

$sourcePaths = @("main.py", "data", "models", "train", "eval", "plots", "utils")
$files = foreach ($path in $sourcePaths) {
    if (Test-Path $path) {
        Get-ChildItem -Path $path -Recurse -Include *.py -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notlike "*__pycache__*" }
    }
}
py -3 -m py_compile @($files.FullName)

@'
import importlib
mods = [
 'utils.seed','utils.io','utils.config_utils','utils.logging_utils','utils.dependencies',
 'data.loaders','data.splits','data.wildjailbreak','data.multiturn_jailbreak','data.sleeper_builder','data.attack_fingerprints','data.real_attack_corpus',
 'models.subject','models.sae','models.trajectory','models.fusion','models.baselines','models.interventions','models.ctcr','models.arf',
 'train.train_lora','train.train_sae','train.train_ctcr','train.train_arf','train.train_fuser',
 'eval.judge','eval.generate','eval.metrics','eval.ablations','eval.significance','eval.mechanistic_taxonomy','eval.conjunction_tests','eval.causal_interventions','eval.null_controls','eval.transfer','eval.layer_sweep',
 'plots.style','plots.main_figures','plots.appendix_figures','plots.latex_tables','plots.mechanistic_figures','plots.arf_figures',
 'main'
]
for m in mods:
    importlib.import_module(m)
print("compile/import checks passed")
'@ | py -3 -

@'
from utils.config_utils import load_config
for path in ["configs/default.yaml", "configs/debug.yaml"]:
    cfg = load_config(path)
    print(path, cfg["subject_model"], cfg["judge_model"], cfg["eval"]["seeds"])
print("config checks passed")
'@ | py -3 -
