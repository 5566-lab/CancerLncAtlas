Set-StrictMode -Version Latest

$script:ControlFormat = 'CANCERLNCATLAS_V32_ORACLE_AGGREGATE_RUN_R3_V1'
$script:ScientificFormat = 'CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1'
$script:RunId = 'v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r3'
$script:TaskId = "$($script:RunId)|PATIENT_FOLD_0|CC-HHGT|20260726"
$script:ResultPrefix = '${DATA_ROOT}/CancerLncAtlas/runtime/runs/v32_group_shared_oracle_paid_gpu_20260901_r3/'

function Test-R3Property([object]$Value, [string]$Name) {
    return $null -ne $Value -and $Value.PSObject.Properties.Name -contains $Name
}

function Test-R3Sha256([object]$Value) {
    return [string]$Value -cmatch '^[0-9a-f]{64}$'
}

function Test-R3AggregateBinding(
    [object]$Value,
    [string]$ExpectedInputSha256,
    [string]$ExpectedCodeSha256
) {
    return (
        (Test-R3Property $Value 'aggregate_input_archive_sha256') -and
        (Test-R3Property $Value 'aggregate_code_archive_sha256') -and
        [string]$Value.aggregate_input_archive_sha256 -ceq $ExpectedInputSha256 -and
        [string]$Value.aggregate_code_archive_sha256 -ceq $ExpectedCodeSha256 -and
        (Test-R3Property $Value 'per_file_hashing_performed') -and
        (Test-R3Property $Value 'per_fold_hashing_performed') -and
        $Value.per_file_hashing_performed -eq $false -and
        $Value.per_fold_hashing_performed -eq $false
    )
}

function Test-R3GpuTelemetry([object]$Value) {
    try {
        return (
            [string]$Value.format -ceq $script:ControlFormat -and
            [string]$Value.status -ceq 'ORACLE_R3_NVIDIA_SMI_TELEMETRY' -and
            [int]$Value.gpu_count -eq 1 -and
            [string]$Value.gpu_name -cmatch 'RTX 4090' -and
            -not [string]::IsNullOrWhiteSpace([string]$Value.gpu_uuid) -and
            -not [string]::IsNullOrWhiteSpace([string]$Value.driver_version) -and
            [int64]$Value.memory_total_mib -gt 0 -and
            [string]$Value.telemetry_source -ceq 'nvidia-smi' -and
            -not [string]::IsNullOrWhiteSpace([string]$Value.sampled_at)
        )
    }
    catch { return $false }
}

function Test-R3FirstOptimizerStep(
    [object]$Value,
    [string]$ExpectedInputSha256,
    [string]$ExpectedCodeSha256
) {
    try {
        return (
            [string]$Value.format -ceq $script:ScientificFormat -and
            [string]$Value.status -ceq 'ORACLE_R3_FIRST_OPTIMIZER_STEP' -and
            [string]$Value.run_id -ceq $script:RunId -and
            [string]$Value.task_id -ceq $script:TaskId -and
            [string]$Value.graph_variant -ceq 'G2' -and
            [int]$Value.patient_fold -eq 0 -and
            [int]$Value.seed -eq 20260726 -and
            [int]$Value.optimizer_step -eq 1 -and
            $Value.grad_finite -eq $true -and
            $Value.parameters_finite -eq $true -and
            $Value.parameter_delta_positive -eq $true -and
            (Test-R3Property $Value 'gpu_identity') -and
            (Test-R3AggregateBinding $Value $ExpectedInputSha256 $ExpectedCodeSha256)
        )
    }
    catch { return $false }
}

function Test-R3Heartbeat(
    [object]$Value,
    [string]$ExpectedInputSha256,
    [string]$ExpectedCodeSha256
) {
    try {
        $completed = [int]$Value.completed_assignments
        $total = [int]$Value.total_assignments
        return (
            [string]$Value.format -ceq $script:ScientificFormat -and
            [string]$Value.status -ceq 'ORACLE_COMPARISON_HEARTBEAT' -and
            [string]$Value.run_id -ceq $script:RunId -and
            [string]$Value.task_id -ceq $script:TaskId -and
            [string]$Value.graph_variant -ceq 'G2' -and
            [int]$Value.patient_fold -eq 0 -and
            [int]$Value.seed -eq 20260726 -and
            $completed -ge 1 -and $total -ge 1 -and $completed -le $total -and
            @('theta_0', 'theta_1') -ccontains [string]$Value.theta -and
            -not [string]::IsNullOrWhiteSpace([string]$Value.estimator) -and
            (Test-R3AggregateBinding $Value $ExpectedInputSha256 $ExpectedCodeSha256)
        )
    }
    catch { return $false }
}

function Test-R3ResultEvent(
    [object]$Value,
    [string]$ExpectedInputSha256,
    [string]$ExpectedCodeSha256
) {
    try {
        return (
            [string]$Value.format -ceq $script:ControlFormat -and
            [string]$Value.status -ceq 'ORACLE_R3_RESULT_ARCHIVE_READY' -and
            (Test-R3Sha256 $Value.result_archive_sha256) -and
            [int64]$Value.result_archive_size_bytes -gt 0 -and
            [string]$Value.result_archive_path -clike "$($script:ResultPrefix)*" -and
            [string]$Value.integrity_scope -ceq 'INPUT_CODE_RESULT_ARCHIVES_ONLY' -and
            (Test-R3AggregateBinding $Value $ExpectedInputSha256 $ExpectedCodeSha256)
        )
    }
    catch { return $false }
}

function Test-R3Terminal(
    [object]$Value,
    [string]$ExpectedInputSha256,
    [string]$ExpectedCodeSha256
) {
    try {
        $status = [string]$Value.status
        if ($status -notin @('ORACLE_R3_PASS', 'ORACLE_R3_FAIL')) { return $false }
        $pass = $status -ceq 'ORACLE_R3_PASS'
        if (
            $pass -and
            [string]$Value.scientific_status -cne 'PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING'
        ) { return $false }
        return (
            [string]$Value.format -ceq $script:ControlFormat -and
            $Value.scientific_pass -eq $pass -and
            (([int]$Value.runner_exit_code -eq 0) -eq $pass) -and
            (Test-R3Sha256 $Value.result_archive_sha256) -and
            [int64]$Value.result_archive_size_bytes -gt 0 -and
            [string]$Value.result_archive_path -clike "$($script:ResultPrefix)*" -and
            [string]$Value.integrity_scope -ceq 'INPUT_CODE_RESULT_ARCHIVES_ONLY' -and
            (Test-R3AggregateBinding $Value $ExpectedInputSha256 $ExpectedCodeSha256)
        )
    }
    catch { return $false }
}

function Read-R3OracleLogWindow {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Logs,
        [Parameter(Mandatory = $true)][string]$ExpectedInputSha256,
        [Parameter(Mandatory = $true)][string]$ExpectedCodeSha256,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()]
        [System.Collections.Generic.HashSet[string]]$SeenProgressIds
    )
    $telemetry = $null
    $optimizer = $null
    $result = $null
    $terminal = $null
    $newProgress = 0
    $unsafeControl = $false
    foreach ($line in ($Logs -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $item = $line | ConvertFrom-Json -ErrorAction Stop }
        catch { continue }
        if (-not (Test-R3Property $item 'status')) { continue }
        $status = [string]$item.status
        switch ($status) {
            'ORACLE_R3_NVIDIA_SMI_TELEMETRY' {
                if (Test-R3GpuTelemetry $item) { $telemetry = $item }
                else { $unsafeControl = $true }
            }
            'ORACLE_R3_FIRST_OPTIMIZER_STEP' {
                if (Test-R3FirstOptimizerStep $item $ExpectedInputSha256 $ExpectedCodeSha256) {
                    $optimizer = $item
                    if ($SeenProgressIds.Add('OPTIMIZER|1')) { $newProgress += 1 }
                }
                else { $unsafeControl = $true }
            }
            'ORACLE_COMPARISON_HEARTBEAT' {
                if (Test-R3Heartbeat $item $ExpectedInputSha256 $ExpectedCodeSha256) {
                    $id = 'HEARTBEAT|{0}|{1}|{2}|{3}' -f $item.theta, $item.estimator, $item.completed_assignments, $item.total_assignments
                    if ($SeenProgressIds.Add($id)) { $newProgress += 1 }
                }
                else { $unsafeControl = $true }
            }
            'ORACLE_R3_RESULT_ARCHIVE_READY' {
                if (Test-R3ResultEvent $item $ExpectedInputSha256 $ExpectedCodeSha256) { $result = $item }
                else { $unsafeControl = $true }
            }
            { $_ -in @('ORACLE_R3_PASS', 'ORACLE_R3_FAIL') } {
                if (Test-R3Terminal $item $ExpectedInputSha256 $ExpectedCodeSha256) { $terminal = $item }
                else { $unsafeControl = $true }
            }
        }
    }
    return [pscustomobject]@{
        Telemetry = $telemetry
        Optimizer = $optimizer
        Result = $result
        Terminal = $terminal
        NewProgressCount = $newProgress
        UnsafeControlSeen = $unsafeControl
    }
}

Export-ModuleMember -Function @(
    'Test-R3GpuTelemetry',
    'Test-R3FirstOptimizerStep',
    'Test-R3ResultEvent',
    'Test-R3Terminal',
    'Read-R3OracleLogWindow'
)
