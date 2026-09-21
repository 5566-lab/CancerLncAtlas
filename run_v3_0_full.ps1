param(
  [string]$Python = "python",
  [string]$Config = "config/model_v3_0_clinical.yaml",
  [int]$Jobs = 1
)
$ErrorActionPreference = "Stop"
& $Python scripts\69_run_v30_full_pipeline.py --config $Config --python $Python --jobs $Jobs --resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
