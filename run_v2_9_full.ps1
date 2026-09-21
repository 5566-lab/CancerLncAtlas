param(
  [string]$Python = "python",
  [string]$Config = "config/model_v2_9_state_graph_local.yaml",
  [string]$DirectEventGlobs = $env:CC_HHGT_DIRECT_EVENT_GLOBS,
  [switch]$Resume,
  [switch]$SkipUpstream,
  [switch]$SkipStrict,
  [switch]$SkipEvidence
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ($null -eq $DirectEventGlobs) { $DirectEventGlobs = "" }
$argsList = @("scripts\50_run_v29_full_pipeline.py", "--config", $Config, "--python", $Python, "--direct-event-globs", $DirectEventGlobs)
if ($Resume) { $argsList += "--resume" }
if ($SkipUpstream) { $argsList += "--skip-upstream" }
if ($SkipStrict) { $argsList += "--skip-strict" }
if ($SkipEvidence) { $argsList += "--skip-evidence" }
& $Python @argsList
if ($LASTEXITCODE -ne 0) { throw "V2.9 pipeline failed with exit code $LASTEXITCODE" }
