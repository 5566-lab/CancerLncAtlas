param(
    [switch]$SkipServer
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Get-FileRecord {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath
    )
    $absolute = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
        return [ordered]@{
            path = $RelativePath
            exists = $false
            bytes = $null
            sha256 = $null
        }
    }
    $item = Get-Item -LiteralPath $absolute
    $hash = (Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant()
    return [ordered]@{
        path = $RelativePath
        exists = $true
        bytes = [int64]$item.Length
        sha256 = $hash
    }
}

$criticalPaths = @(
    'artifacts/data_processing_truth_matrix_20260829_r2_expanded/TRUTH_MATRIX.json',
    'artifacts/data_processing_runtime_findings_20260829_r3/RUNTIME_FINDINGS.json',
    'artifacts/remaining_issue_completion_audit_20260829_r1/REMAINING_WORK_AUDIT.json',
    'artifacts/processing_not_raw_data_summary_20260829_r4/PROCESSING_NOT_RAW_DATA_SUMMARY.zh-CN.md',
    'artifacts/processing_not_raw_data_summary_20260829_r4/SUMMARY_MANIFEST.json',
    'artifacts/v32_winner_lock_sealed_test_code_audit_20260829_r1/AUDIT.json',
    'artifacts/v32_server_preflight_g012_atac_genomic_20260829_r1/PREFLIGHT.json',
    'artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/REMATERIALIZATION_MANIFEST.server.json',
    'artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/VALIDATION.server.json',
    'artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/CONVERSION_ATTEMPT_LEDGER.json',
    'artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/CONVERSION_ATTEMPT_LEDGER_R2.json',
    'artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/CONVERSION_R3_MANIFEST.server.json',
    'artifacts/v32_evidence_interaction_parquet_20260829_r1_server_receipts/CONVERSION_R3_SUCCESS.server.json',
    'artifacts/v32_ecs_cleanup_smokes_20260829_r1/EVIDENCE_RUNTIME_QUERY_CLASS_SMOKE_R4.json',
    'cc_hhgt/v32/evidence_streaming_training.py',
    'scripts/materialize_v32_evidence_streaming_stage.py',
    'tests/test_v32_evidence_streaming_training.py',
    'artifacts/v32_patient_graph_integration_deployment_audit_20260829_r3/AUDIT.json',
    'artifacts/v32_patient_graph_integration_deployment_audit_20260829_r3/DEPLOYMENT_MANIFEST.json',
    'artifacts/v32_patient_graph_integration_deployment_audit_20260829_r3/TEST_RECEIPT.json',
    'artifacts/v32_patient_graph_integration_deployment_audit_20260829_r3_amendment_r1/CORRECTION_RECEIPT.json',
    'artifacts/cancerlncatlas_handoff_20260829_r1/NEXT_AGENT_PROMPT.md',
    'artifacts/cancerlncatlas_handoff_20260829_r1/HANDOFF.md'
)

$localFiles = @()
foreach ($path in $criticalPaths) {
    $localFiles += Get-FileRecord -RelativePath $path
}

$summaryRoot = Join-Path $repoRoot 'artifacts/processing_not_raw_data_summary_20260829_r4'
$summaryManifestPath = Join-Path $summaryRoot 'SUMMARY_MANIFEST.json'
$summaryMarkdownPath = Join-Path $summaryRoot 'PROCESSING_NOT_RAW_DATA_SUMMARY.zh-CN.md'
$summaryBinding = [ordered]@{
    checked = $false
    pass = $false
    manifest_bytes = $null
    observed_bytes = $null
    manifest_sha256 = $null
    observed_sha256 = $null
}
if ((Test-Path -LiteralPath $summaryManifestPath -PathType Leaf) -and
    (Test-Path -LiteralPath $summaryMarkdownPath -PathType Leaf)) {
    $manifest = ([System.IO.File]::ReadAllText(
        $summaryManifestPath,
        [System.Text.Encoding]::UTF8
    ) | ConvertFrom-Json)
    $markdownItem = Get-Item -LiteralPath $summaryMarkdownPath
    $markdownHash = (Get-FileHash -LiteralPath $summaryMarkdownPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $summaryBinding.checked = $true
    $summaryBinding.manifest_bytes = [int64]$manifest.summary.bytes
    $summaryBinding.observed_bytes = [int64]$markdownItem.Length
    $summaryBinding.manifest_sha256 = [string]$manifest.summary.sha256
    $summaryBinding.observed_sha256 = $markdownHash
    $summaryBinding.pass = (
        $summaryBinding.manifest_bytes -eq $summaryBinding.observed_bytes -and
        $summaryBinding.manifest_sha256 -eq $summaryBinding.observed_sha256
    )
}

$serverProbe = [ordered]@{
    attempted = $false
    pass = $false
    host = 'COMPUTE_HOST'
    command_scope = 'READ_ONLY_DSC_PATHS_AND_PROCESS_STATUS'
    stdout = @()
    exit_code = $null
}

if (-not $SkipServer) {
    $serverProbe.attempted = $true
    $remoteCommand = @'
set -u
echo EVIDENCE_R3_PROCESS
ps -p 1601755 -o pid=,etime=,stat=,%cpu=,rss=,cmd= 2>/dev/null || true
echo EVIDENCE_R3_OUTPUT
find ./data/CancerLncAtlas/results/v32_evidence_interaction_parquet_20260829_r3 -maxdepth 1 -type f -printf '%f\t%s\n' 2>/dev/null | sort || true
echo EVIDENCE_R3_LOG_TAIL
tail -20 ./data/CancerLncAtlas/runtime/logs/v32_evidence_interaction_parquet_20260829_r3/converter.log 2>/dev/null || true
echo SINGLE_CELL_STANDALONE_RECEIPTS
find ./data/CancerLncAtlas/runtime ./data/CancerLncAtlas/results -maxdepth 5 -type f \( -name 'RUNTIME_BINDING.json' -o -name 'EQUIVALENCE*.json' -o -name 'SUCCESS.json' \) -path '*single_cell*' -printf '%p\t%s\n' 2>/dev/null | sort | tail -80 || true
echo CUDA_STATUS
command -v nvidia-smi 2>/dev/null || true
echo PRODUCTION_8260_LISTENER_READ_ONLY
ss -ltn 2>/dev/null | grep ':8260 ' || true
'@
    $output = & ssh.exe -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 COMPUTE_HOST $remoteCommand 2>&1
    $serverProbe.exit_code = $LASTEXITCODE
    $serverProbe.stdout = @($output | ForEach-Object { [string]$_ })
    $serverProbe.pass = ($LASTEXITCODE -eq 0)
}

$result = [ordered]@{
    format = 'CANCERLNCATLAS_HANDOFF_READ_ONLY_AUDIT_V1'
    generated_at_local = (Get-Date).ToString('o')
    repo_root = $repoRoot
    mutation_performed = $false
    production_port_8260_touched = $false
    local_files = $localFiles
    summary_binding = $summaryBinding
    server_probe = $serverProbe
    frozen_open_issue_snapshot = [ordered]@{
        total = 60
        processing = 49
        publication_binding = 11
        note = 'Snapshot count; rerun closure audit before claiming a smaller number.'
    }
}

$result | ConvertTo-Json -Depth 12
