param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,
    [int]$StartFrame = 0,
    [int]$MaxFrames = 450
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedVideo = (Resolve-Path -LiteralPath $VideoPath).Path

Push-Location $repoRoot
try {
    python code\scripts\run_main_risk_demo.py `
        --video $resolvedVideo `
        --config configs\config_final_w60.yaml `
        --output-dir demo\generated_outputs `
        --start-frame $StartFrame `
        --max-frames $MaxFrames `
        --status-source temporal
}
finally {
    Pop-Location
}
