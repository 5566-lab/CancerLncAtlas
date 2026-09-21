Set-StrictMode -Version Latest

$script:ReceiptFormat = 'CC_HHGT_V3_2_GROUP_SHARED_ORACLE_COMPARISON_V1'
$script:PassStatus = 'PASS_REAL_DATA_ORACLE_COMPARISON_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING'
$script:FailStatuses = @(
    'SCIENTIFIC_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY',
    'RUNTIME_FAIL_REAL_DATA_ORACLE_COMPARISON_ONLY'
)
$script:RunId = 'v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r2'
$script:TaskId = 'v32-g012-g2-group-shared-oracle-paid-gpu-20260901-r2|PATIENT_FOLD_0|CC-HHGT|20260726'

function Get-OracleLineSha256 {
    param([Parameter(Mandatory = $true)][string]$Line)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = (New-Object System.Text.UTF8Encoding($false, $true)).GetBytes($Line)
        return ([BitConverter]::ToString($algorithm.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally { $algorithm.Dispose() }
}

function Test-OracleProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Value,
        [Parameter(Mandatory = $true)][string]$Name
    )
    return $Value.PSObject.Properties.Name -contains $Name
}

function Test-OracleSafeTerminalReceipt {
    param(
        [Parameter(Mandatory = $true)][object]$Receipt,
        [Parameter(Mandatory = $true)][string]$ExpectedStatus
    )
    try {
        if (-not (Test-OracleProperty $Receipt 'format') -or [string]$Receipt.format -cne $script:ReceiptFormat) { return $false }
        if (-not (Test-OracleProperty $Receipt 'status') -or [string]$Receipt.status -cne $ExpectedStatus) { return $false }
        foreach ($name in @('run_id', 'task_id', 'graph_variant', 'patient_fold', 'seed')) {
            if (-not (Test-OracleProperty $Receipt $name)) { return $false }
        }
        if ([string]$Receipt.run_id -cne $script:RunId -or [string]$Receipt.task_id -cne $script:TaskId) { return $false }
        if ([string]$Receipt.graph_variant -cne 'G2' -or [int]$Receipt.patient_fold -ne 0 -or [int]$Receipt.seed -ne 20260726) { return $false }
        $requiredFalse = @(
            'formal_training_authorized', 'checkpoint_written',
            'success_json_written', 'failure_json_written',
            'prediction_written', 'winner_selection_input'
        )
        foreach ($name in $requiredFalse) {
            if (-not (Test-OracleProperty $Receipt $name) -or $Receipt.$name -ne $false) { return $false }
        }
        if (-not (Test-OracleProperty $Receipt 'formal_artifacts_written') -or [int64]$Receipt.formal_artifacts_written -ne 0) { return $false }
        if (-not (Test-OracleProperty $Receipt 'scientific_pass')) { return $false }
        if ($ExpectedStatus -ceq $script:PassStatus) {
            return $Receipt.scientific_pass -eq $true
        }
        if ($script:FailStatuses -ccontains $ExpectedStatus) {
            return $Receipt.scientific_pass -eq $false
        }
        return $false
    }
    catch { return $false }
}

function Test-OracleHeartbeatReceipt {
    param([Parameter(Mandatory = $true)][object]$Receipt)
    try {
        $required = @(
            'format', 'status', 'artifact_class', 'comparison_only',
            'formal_training_authorized', 'formal_artifacts_written', 'theta',
            'estimator', 'completed_assignments', 'total_assignments', 'run_id',
            'task_id', 'patient_fold', 'seed', 'graph_variant', 'assignment_sha256'
        )
        foreach ($name in $required) {
            if (-not (Test-OracleProperty $Receipt $name)) { return $false }
        }
        if ([string]$Receipt.format -cne $script:ReceiptFormat) { return $false }
        if ([string]$Receipt.status -cne 'ORACLE_COMPARISON_HEARTBEAT') { return $false }
        if ([string]$Receipt.artifact_class -cne 'TIMING_AND_ORACLE_COMPARISON_ONLY') { return $false }
        if ($Receipt.comparison_only -ne $true -or $Receipt.formal_training_authorized -ne $false) { return $false }
        if ([int64]$Receipt.formal_artifacts_written -ne 0) { return $false }
        if ([string]$Receipt.run_id -cne $script:RunId -or [string]$Receipt.task_id -cne $script:TaskId) { return $false }
        if ([int]$Receipt.patient_fold -ne 0 -or [int]$Receipt.seed -ne 20260726 -or [string]$Receipt.graph_variant -cne 'G2') { return $false }
        if (@('theta_0', 'theta_1') -cnotcontains [string]$Receipt.theta) { return $false }
        if (@(
            'exact_cartesian_v1_group0_mixed_chunk_oracle',
            'balanced_group_latin_shared_encoder_v1'
        ) -cnotcontains [string]$Receipt.estimator) { return $false }
        $completed = [int]$Receipt.completed_assignments
        $total = [int]$Receipt.total_assignments
        if ($completed -lt 1 -or $total -lt 1 -or $completed -gt $total) { return $false }
        if ([string]$Receipt.assignment_sha256 -cnotmatch '^[0-9a-f]{64}$') { return $false }
        return $true
    }
    catch { return $false }
}

function Read-OracleLogWindow {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Logs,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.HashSet[string]]$SeenLineHashes,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.HashSet[string]]$SeenEventIds,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][hashtable]$MaxCompletedByPhase
    )
    $newHeartbeatCount = 0
    $newestHeartbeatId = $null
    $terminalLine = $null
    $terminalStatus = $null
    $unsafeTerminalSeen = $false

    foreach ($line in ($Logs -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $item = $line | ConvertFrom-Json -ErrorAction Stop }
        catch { continue }
        if (-not (Test-OracleProperty $item 'format') -or [string]$item.format -cne $script:ReceiptFormat) { continue }
        if (-not (Test-OracleProperty $item 'status')) { continue }
        $status = [string]$item.status

        if ($status -ceq 'ORACLE_COMPARISON_HEARTBEAT') {
            if (-not (Test-OracleHeartbeatReceipt $item)) { continue }
            $lineSha = Get-OracleLineSha256 -Line $line
            $phase = '{0}|{1}' -f [string]$item.theta, [string]$item.estimator
            $eventId = '{0}|{1}|{2}|{3}' -f $phase, [int]$item.completed_assignments, [int]$item.total_assignments, [string]$item.assignment_sha256
            $prior = 0
            if ($MaxCompletedByPhase.ContainsKey($phase)) { $prior = [int]$MaxCompletedByPhase[$phase] }
            # Both byte identity and semantic identity are remembered.  A
            # repeated --tail window, reordered JSON, or a late old line can
            # never refresh the progress deadline.
            if (
                -not $SeenLineHashes.Contains($lineSha) -and
                -not $SeenEventIds.Contains($eventId) -and
                [int]$item.completed_assignments -gt $prior
            ) {
                [void]$SeenLineHashes.Add($lineSha)
                [void]$SeenEventIds.Add($eventId)
                $MaxCompletedByPhase[$phase] = [int]$item.completed_assignments
                $newHeartbeatCount += 1
                $newestHeartbeatId = $eventId
            }
            continue
        }

        if ($status -ceq $script:PassStatus -or $script:FailStatuses -ccontains $status) {
            if (Test-OracleSafeTerminalReceipt -Receipt $item -ExpectedStatus $status) {
                $terminalLine = $line
                $terminalStatus = $status
            }
            else { $unsafeTerminalSeen = $true }
        }
    }

    return [pscustomobject]@{
        NewHeartbeatCount = $newHeartbeatCount
        NewestHeartbeatId = $newestHeartbeatId
        TerminalLine = $terminalLine
        TerminalStatus = $terminalStatus
        UnsafeTerminalSeen = $unsafeTerminalSeen
    }
}

Export-ModuleMember -Function @(
    'Get-OracleLineSha256',
    'Test-OracleSafeTerminalReceipt',
    'Read-OracleLogWindow'
)
