Set-StrictMode -Version Latest

$script:Schema = 'CC_HHGT_V3_2_EQUAL_STEP_CANDIDATE_PILOT_V1'
$script:PilotId = 'v32-g012-equal-step-candidate-paid-gpu-20260901-r1'
$script:Variants = @('G0', 'G1', 'G2')
$script:PassStatus = 'PASS_EQUAL_STEP_CANDIDATE_ONLY_NOT_AUTHORIZED_FOR_FORMAL_TRAINING'
$script:FailStatuses = @(
    'SCIENTIFIC_FAIL_EQUAL_STEP_CANDIDATE_ONLY',
    'RUNTIME_FAIL_EQUAL_STEP_CANDIDATE_ONLY'
)

function Get-EqualStepLineSha256 {
    param([Parameter(Mandatory = $true)][string]$Line)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = (New-Object System.Text.UTF8Encoding($false, $true)).GetBytes($Line)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally { $sha.Dispose() }
}

function Test-EqualStepProperty {
    param([object]$Value, [string]$Name)
    return $null -ne $Value -and $Value.PSObject.Properties.Name -contains $Name
}

function Test-EqualStepBaseSafety {
    param([object]$Receipt)
    try {
        if ([string]$Receipt.schema -cne $script:Schema) { return $false }
        if ($Receipt.comparison_only -ne $true -or $Receipt.candidate_only -ne $true) { return $false }
        if ($Receipt.formal_training_authorized_by_this_receipt -ne $false) { return $false }
        if ([int64]$Receipt.formal_artifacts_written -ne 0) { return $false }
        foreach ($name in @(
            'formal_checkpoint_written', 'formal_success_marker_written',
            'formal_failure_marker_written', 'formal_prediction_written',
            'test_or_outer_artifact_written', 'winner_selection_input',
            'test_labels_read', 'outer_metrics_read', 'outer_predictions_read'
        )) {
            if (-not (Test-EqualStepProperty $Receipt $name) -or $Receipt.$name -ne $false) { return $false }
        }
        if ([int64]$Receipt.test_rows_read -ne 0 -or [string]$Receipt.output_channel -cne 'STDOUT_JSON_ONLY') { return $false }
        return $true
    }
    catch { return $false }
}

function Test-EqualStepIdentity {
    param([object]$Receipt)
    try {
        $variant = [string]$Receipt.graph_variant
        if ($script:Variants -cnotcontains $variant) { return $false }
        $runId = '{0}-{1}' -f $script:PilotId, $variant.ToLowerInvariant()
        $taskId = '{0}|PATIENT_FOLD_0|CC-HHGT|20260726' -f $runId
        return (
            [string]$Receipt.pilot_id -ceq $script:PilotId -and
            [string]$Receipt.authorization_run_id -ceq $runId -and
            [string]$Receipt.authorization_task_id -ceq $taskId
        )
    }
    catch { return $false }
}

function Test-EqualStepProgressReceipt {
    param([object]$Receipt)
    try {
        if (-not (Test-EqualStepBaseSafety $Receipt) -or -not (Test-EqualStepIdentity $Receipt)) { return $false }
        $status = [string]$Receipt.status
        $variant = [string]$Receipt.graph_variant
        $runId = [string]$Receipt.authorization_run_id
        $taskId = [string]$Receipt.authorization_task_id
        $event = [string]$Receipt.progress_event_id
        if ($status -ceq 'EQUAL_STEP_VARIANT_SETUP_START') {
            $expected = '{0}|{1}|{2}|{3}|none|VARIANT_SETUP|0' -f $script:PilotId, $runId, $taskId, $variant
            return (
                [string]$Receipt.arm -ceq 'none' -and [string]$Receipt.phase -ceq 'VARIANT_SETUP' -and
                [int]$Receipt.completed -eq 0 -and [int]$Receipt.total -eq 1 -and
                $Receipt.optimizer_progress_observed -eq $false -and $Receipt.training_started -eq $false -and
                $event -ceq $expected
            )
        }
        if ($status -ceq 'EQUAL_STEP_OPTIMIZER_PROGRESS') {
            $arm = [string]$Receipt.arm
            $estimator = [string]$Receipt.estimator
            $gpu = $Receipt.gpu_telemetry
            if ($arm -ceq 'reference' -and $estimator -cne 'balanced_cyclic_single_pass_v1') { return $false }
            if ($arm -ceq 'proposed' -and $estimator -cne 'balanced_group_latin_shared_encoder_v1') { return $false }
            if (@('reference', 'proposed') -cnotcontains $arm) { return $false }
            if (
                $Receipt.training_started -ne $true -or
                $null -eq $gpu -or [string]$gpu.device -cne 'cuda' -or
                $gpu.cuda_available -ne $true -or
                [int64]$gpu.memory_allocated_bytes -le 0 -or
                [int64]$gpu.memory_reserved_bytes -lt [int64]$gpu.memory_allocated_bytes
            ) { return $false }
            $completed = [int]$Receipt.completed
            $expected = '{0}|{1}|{2}|{3}|{4}|OPTIMIZER|{5}' -f $script:PilotId, $runId, $taskId, $variant, $arm, $completed
            return (
                [string]$Receipt.phase -ceq 'OPTIMIZER' -and $completed -ge 1 -and $completed -le 101 -and
                [int]$Receipt.total -eq 101 -and [int]$Receipt.optimizer_step -eq $completed -and
                [int]$Receipt.required_optimizer_steps -eq 101 -and
                $Receipt.grad_finite -eq $true -and $Receipt.parameters_finite -eq $true -and
                $Receipt.parameter_delta_positive -eq $true -and $event -ceq $expected
            )
        }
        if ($status -ceq 'EQUAL_STEP_VALIDATION_CHUNK_PROGRESS') {
            $arm = [string]$Receipt.arm
            if (@('reference', 'proposed') -cnotcontains $arm) { return $false }
            $completed = [int]$Receipt.completed
            $expected = '{0}|{1}|{2}|{3}|{4}|VALIDATION|{5}' -f $script:PilotId, $runId, $taskId, $variant, $arm, $completed
            return (
                [string]$Receipt.phase -ceq 'VALIDATION' -and $completed -ge 1 -and $completed -le 29 -and
                [int]$Receipt.total -eq 29 -and [int]$Receipt.validation_chunks_completed -eq $completed -and
                [int]$Receipt.required_runtime_chunks -eq 29 -and
                [int]$Receipt.validation_chunk_ordinal -ge 1 -and [int]$Receipt.validation_chunk_ordinal -le 29 -and
                [int]$Receipt.candidate_microbatches_completed_for_chunk -ge 1 -and $event -ceq $expected
            )
        }
        if ($status -ceq 'EQUAL_STEP_VARIANT_HEARTBEAT') {
            $expected = '{0}|{1}|{2}|{3}|both|VARIANT_COMPLETE|1' -f $script:PilotId, $runId, $taskId, $variant
            return (
                [string]$Receipt.arm -ceq 'both' -and [string]$Receipt.phase -ceq 'VARIANT_COMPLETE' -and
                [int]$Receipt.completed -eq 1 -and [int]$Receipt.total -eq 1 -and
                [int]$Receipt.reference_optimizer_steps -eq 101 -and
                [int]$Receipt.proposed_optimizer_steps -eq 101 -and $event -ceq $expected
            )
        }
        return $false
    }
    catch { return $false }
}

function Test-EqualStepTerminalReceipt {
    param([object]$Receipt, [string]$ExpectedStatus)
    try {
        if (-not (Test-EqualStepBaseSafety $Receipt)) { return $false }
        if ([string]$Receipt.status -cne $ExpectedStatus -or [string]$Receipt.pilot_id -cne $script:PilotId) { return $false }
        if ($ExpectedStatus -ceq $script:PassStatus) {
            return (
                $Receipt.scientific_pass -eq $true -and [int]$Receipt.patient_fold -eq 0 -and
                [int]$Receipt.seed -eq 20260726 -and
                (@($Receipt.graph_variants) -join ',') -ceq 'G0,G1,G2' -and
                $Receipt.cross_variant_pass -eq $true
            )
        }
        if ($script:FailStatuses -ccontains $ExpectedStatus) {
            return $Receipt.scientific_pass -eq $false
        }
        return $false
    }
    catch { return $false }
}

function Test-EqualStepCompletePassProgress {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][hashtable]$MaxCompletedByPhase
    )
    foreach ($variant in $script:Variants) {
        $required = [ordered]@{
            ("{0}|none|VARIANT_SETUP" -f $variant) = 0
            ("{0}|reference|OPTIMIZER" -f $variant) = 101
            ("{0}|proposed|OPTIMIZER" -f $variant) = 101
            ("{0}|reference|VALIDATION" -f $variant) = 29
            ("{0}|proposed|VALIDATION" -f $variant) = 29
            ("{0}|both|VARIANT_COMPLETE" -f $variant) = 1
        }
        foreach ($entry in $required.GetEnumerator()) {
            if (
                -not $MaxCompletedByPhase.ContainsKey([string]$entry.Key) -or
                [int]$MaxCompletedByPhase[[string]$entry.Key] -ne [int]$entry.Value
            ) { return $false }
        }
    }
    return $true
}

function Read-EqualStepLogWindow {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Logs,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.HashSet[string]]$SeenLineHashes,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][System.Collections.Generic.HashSet[string]]$SeenEventIds,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][hashtable]$MaxCompletedByPhase
    )
    $newProgress = 0
    $newSetup = @()
    $terminalLine = $null
    $terminalStatus = $null
    $unsafeTerminal = $false
    $newTerminalCount = 0
    $terminalHashesThisWindow = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
    foreach ($line in ($Logs -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        try { $item = $line | ConvertFrom-Json -ErrorAction Stop } catch { continue }
        if (-not (Test-EqualStepProperty $item 'schema') -or [string]$item.schema -cne $script:Schema) { continue }
        $status = [string]$item.status
        if (@(
            'EQUAL_STEP_VARIANT_SETUP_START', 'EQUAL_STEP_OPTIMIZER_PROGRESS',
            'EQUAL_STEP_VALIDATION_CHUNK_PROGRESS', 'EQUAL_STEP_VARIANT_HEARTBEAT'
        ) -ccontains $status) {
            if (-not (Test-EqualStepProgressReceipt $item)) { continue }
            $lineSha = Get-EqualStepLineSha256 -Line $line
            $eventId = [string]$item.progress_event_id
            if ($SeenLineHashes.Contains($lineSha) -or $SeenEventIds.Contains($eventId)) { continue }
            [void]$SeenLineHashes.Add($lineSha)
            [void]$SeenEventIds.Add($eventId)
            $phaseKey = '{0}|{1}|{2}' -f [string]$item.graph_variant, [string]$item.arm, [string]$item.phase
            $completed = [int]$item.completed
            $prior = -1
            if ($MaxCompletedByPhase.ContainsKey($phaseKey)) { $prior = [int]$MaxCompletedByPhase[$phaseKey] }
            if ($status -ceq 'EQUAL_STEP_VARIANT_SETUP_START') {
                if ($prior -ge 0) { continue }
                $MaxCompletedByPhase[$phaseKey] = 0
                $newSetup += [string]$item.graph_variant
                continue
            }
            if ($completed -le $prior) { continue }
            $MaxCompletedByPhase[$phaseKey] = $completed
            $newProgress += 1
            continue
        }
        if ($status -ceq $script:PassStatus -or $script:FailStatuses -ccontains $status) {
            $lineSha = Get-EqualStepLineSha256 -Line $line
            if ($terminalHashesThisWindow.Contains($lineSha)) {
                $unsafeTerminal = $true
                $terminalLine = $null
                $terminalStatus = $null
                continue
            }
            [void]$terminalHashesThisWindow.Add($lineSha)
            if ($SeenLineHashes.Contains($lineSha)) { continue }
            [void]$SeenLineHashes.Add($lineSha)
            if (
                (Test-EqualStepTerminalReceipt -Receipt $item -ExpectedStatus $status) -and
                (
                    $status -cne $script:PassStatus -or
                    (Test-EqualStepCompletePassProgress -MaxCompletedByPhase $MaxCompletedByPhase)
                )
            ) {
                $newTerminalCount += 1
                if ($newTerminalCount -eq 1) {
                    $terminalLine = $line
                    $terminalStatus = $status
                }
                else {
                    $terminalLine = $null
                    $terminalStatus = $null
                    $unsafeTerminal = $true
                }
            }
            else { $unsafeTerminal = $true }
        }
    }
    return [pscustomobject]@{
        NewProgressCount = $newProgress
        NewSetupVariants = @($newSetup)
        TerminalLine = $terminalLine
        TerminalStatus = $terminalStatus
        UnsafeTerminalSeen = $unsafeTerminal
    }
}

Export-ModuleMember -Function @(
    'Get-EqualStepLineSha256', 'Test-EqualStepProgressReceipt',
    'Test-EqualStepTerminalReceipt', 'Test-EqualStepCompletePassProgress',
    'Read-EqualStepLogWindow'
)
