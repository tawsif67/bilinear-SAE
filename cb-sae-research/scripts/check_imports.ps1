$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:MPLCONFIGDIR = Join-Path $root ".matplotlib"

$files = Get-ChildItem -Recurse -Include *.py -File | Where-Object { $_.FullName -notlike "*__pycache__*" }
py -3 -m py_compile @($files.FullName)

@'
import importlib
mods = [
 'utils.seed','utils.io','utils.config_utils','utils.logging_utils',
 'data.loaders','data.splits','data.wildjailbreak','data.multiturn_jailbreak','data.sleeper_builder',
 'models.subject','models.sae','models.trajectory','models.fusion','models.baselines',
 'train.train_lora','train.train_sae','train.train_fuser',
 'eval.judge','eval.generate','eval.metrics','eval.ablations','eval.significance',
 'plots.style','plots.main_figures','plots.appendix_figures','plots.latex_tables',
 'main'
]
for m in mods:
    importlib.import_module(m)
print("compile/import checks passed")
'@ | py -3 -

