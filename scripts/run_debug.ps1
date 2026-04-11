$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:MPLCONFIGDIR = Join-Path $root ".matplotlib"
py -3 main.py --config configs/debug.yaml

