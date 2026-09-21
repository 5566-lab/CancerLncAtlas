from __future__ import annotations

from pathlib import Path
import subprocess

from cc_hhgt.v32.orchestration import extract_execution_policy
from cc_hhgt.v32.training_guard import load_structured_mapping


ROOT = Path(__file__).resolve().parents[1]


def _run_probe_function_source() -> str:
    text = (
        ROOT / "scripts/server_launch_v32_g012_paid_gpu_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    start = text.index("run_probe() {")
    end = text.index("\n\nvalidate_completed_fold_for_resume()", start)
    return text[start:end]


def _probe_harness(call: str) -> subprocess.CompletedProcess[str]:
    harness = r'''set -euo pipefail
fail_wait() { printf 'FAIL_WAIT=%s\n' "$1" >&2; exit 20; }
fail_drift() { printf 'FAIL_DRIFT=%s\n' "$1" >&2; exit 22; }
timeout() { printf 'TIMEOUT_ARGS=%s|%s\n' "$1" "$2"; }
tee() { local destination="$1"; printf 'TEE_PATH=%s\n' "$destination"; cat; }
validate_probe_log() { return 0; }
python_bin=/formal/python
code_root=/formal/code
probe_config=/formal/config.yaml
probe_authorization=/formal/authorization
probe_run_id=formal-run
probe_task_id=formal-task
'''
    script = harness + _run_probe_function_source() + "\n" + call + "\n"
    result = subprocess.run(
        ["bash"],
        check=False,
        capture_output=True,
        input=script.replace("\r\n", "\n").encode("utf-8"),
    )
    return subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.decode("utf-8", errors="strict"),
        result.stderr.decode("utf-8", errors="strict"),
    )


def test_run_probe_binds_fixed_timeout_to_profile_without_sixth_argument() -> None:
    cases = (
        (
            "run_probe 4096 1 ENVIRONMENT_GATE_4096_X_1 4096 /logs/environment.log",
            "TIMEOUT_ARGS=--kill-after=30s|600s",
            "TEE_PATH=/logs/environment.log",
        ),
        (
            "run_probe 8192 4 PRIMARY_GROUP_STRESS_GATE_8192_X_4 32768 /logs/stress.log",
            "TIMEOUT_ARGS=--kill-after=30s|900s",
            "TEE_PATH=/logs/stress.log",
        ),
    )
    for call, expected_timeout, expected_log in cases:
        result = _probe_harness(call)
        assert result.returncode == 0, result.stderr
        assert expected_timeout in result.stdout
        assert expected_log in result.stdout


def test_run_probe_rejects_unknown_profile_before_timeout() -> None:
    result = _probe_harness("run_probe 1 1 UNKNOWN_PROFILE 1 /logs/unknown.log")
    assert result.returncode == 22
    assert "UNKNOWN_REAL_BACKWARD_PROBE_PROFILE=UNKNOWN_PROFILE" in result.stderr
    assert "TIMEOUT_ARGS=" not in result.stdout


def test_paid_g012_template_is_capped_and_external_core_only(tmp_path: Path) -> None:
    template = ROOT / "config/model_v3_2_g012_paid_gpu_20260831_r1.yaml"
    rendered = tmp_path / "config.yaml"
    rendered.write_text(
        template.read_text(encoding="utf-8").replace("__GRAPH_VARIANT__", "G2"),
        encoding="utf-8",
    )
    payload = load_structured_mapping(rendered)
    policy = extract_execution_policy(payload)
    assert policy.execution_mode == "TRAINING"
    assert policy.training_authorized is True
    assert policy.paid_enabled is True
    assert policy.max_paid_hours == 96
    assert policy.max_cost_cny == 210
    assert payload["task_contract"]["graph_variant"] == "G2"
    assert payload["primary_model"]["evidence_integration"]["mode"] == "external_router"
    assert payload["primary_model"]["evidence_integration"]["modalities"] == []
    assert "/G2/PATIENT_FOLD_{fold}.pt" in payload["training_io"]["prepared_fold_pattern"]


def test_paid_launcher_covers_all_arms_and_fail_closed_authority() -> None:
    text = (
        ROOT / "scripts/server_launch_v32_g012_paid_gpu_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert "for variant in G0 G1 G2" in text
    assert "for fold in 0 1 2 3 4" in text
    assert "STATIC_AUTH_READY.json" in text
    assert "validate_v32_g012_static_auth_ready.py" in text
    assert "normalize_v32_task_manifest.py" in text
    assert "IFS=$'\\x1f'" in text
    assert 'test -z "$blocked_reason"' in text
    assert 'export PYTHONPATH="$code_root"' in text
    assert 'python_bin="${V32_PYTHON_BIN:-python3}"' in text
    assert "CC_HHGT_IMPORT_DRIFT" in text
    assert "GPU_RUNTIME_CONTRACT_DRIFT" in text
    assert '"RTX 4090" not in device_name' in text
    assert "capability != (8, 9)" in text
    assert "minimum_total_memory_bytes = 23 * 1024**3" in text
    assert "GPU_FINITE_MATMUL_FAILED" in text
    assert "GPU_DISK_PREFLIGHT_FAILED" in text
    assert "ENVIRONMENT_GATE_4096_X_1) timeout_seconds=600" in text
    assert "PRIMARY_GROUP_STRESS_GATE_8192_X_4) timeout_seconds=900" in text
    assert "UNKNOWN_REAL_BACKWARD_PROBE_PROFILE" in text
    assert 'timeout --kill-after=30s "${timeout_seconds}s"' in text
    assert "4096 1 ENVIRONMENT_GATE_4096_X_1 4096" in text
    assert "8192 4 PRIMARY_GROUP_STRESS_GATE_8192_X_4 32768" in text
    assert "GPU_TRAINING_ALREADY_RUNNING" in text
    assert "GPU_TRAINING_COMPLETE_ALREADY_EXISTS_CPU_FINALIZE_REQUIRED" in text
    assert '"status":"GPU_PHASE_HEARTBEAT"' in text
    assert "cc_hhgt.v32.gpu_backward_probe:run_authorized_probe" in text
    assert "ENVIRONMENT_GATE_4096_X_1" in text
    assert "PRIMARY_GROUP_STRESS_GATE_8192_X_4" in text
    assert '"peak_reserved_limit_bytes": 23622320128' in text
    assert "peak <= 0 or peak >= limit" in text
    assert '"parameters_finite": True' in text
    assert "PROBE_PHYSICAL_HEADROOM_DRIFT" in text
    assert "00_nvidia_smi_memory.csv" in text
    assert "PROBE_LAST_LINE_NOT_JSON" in text
    assert "--endpoint-id paid_gpu" in text
    assert "--hardware-class PAID_PREEMPTIBLE_GPU" in text
    assert "GPU_TRAINING_COMPLETE.json" in text
    assert '"completed_tasks": 15' in text
    assert '"success_sha256"' in text
    assert '"best_model_state_sha256"' in text
    assert '"best_model_state_size_bytes"' in text
    assert "RESUME_BEST_CHECKPOINT_SHA_DRIFT" in text
    assert "TRAINING_BEST_CHECKPOINT_SHA_DRIFT" in text
    assert "TRAINING_SUCCESS_FOLD_OR_SEED_DRIFT" in text
    assert "TRAINING_SUCCESS_AUTHORIZATION_DRIFT" in text
    assert "TRAINING_SUCCESS_PRECISION_DRIFT" in text
    assert "best_model_state.pt" in text
    assert "prepare_v32_hierarchical_training_authorization.py" not in text
    assert "select_v32_g012_validation_winner.py" not in text
    assert "tar -" not in text
    assert "rm -rf" not in text
    assert "dscdsc@149" not in text


def test_server_input_packager_waits_for_all_15_and_preserves_canonical_root() -> None:
    text = (
        ROOT / "scripts/server_package_v32_g012_paid_gpu_input_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert "for variant in G0 G1 G2" in text
    assert "for fold in 0 1 2 3 4" in text
    assert "PREP_METRICS_NOT_CLOSED" in text
    assert "tar -cf \"$temporary\" -C / \"$prepared_rel\"" in text
    assert "sha256sum \"$archive\"" in text
    assert "rm -rf" not in text


def test_server_packaging_watcher_fails_if_materializer_dies_early() -> None:
    text = (
        ROOT / "scripts/server_watch_and_package_v32_g012_paid_gpu_input_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert "while ! test -s \"$prepared_root/PREP_METRICS.json\"" in text
    assert "kill -0 \"$materializer_pid\"" in text
    assert "G012_MATERIALIZER_EXITED_BEFORE_PREP_METRICS" in text
    assert "server_package_v32_g012_paid_gpu_input_20260831_r1.sh" in text
    assert "rm -rf" not in text


def test_legacy_gpu_bootstrap_is_hard_deprecated_before_any_wait() -> None:
    text = (
        ROOT / "scripts/cloud_bootstrap_v32_g012_paid_gpu_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert "DEPRECATED_G012_GPU_BOOTSTRAP" in text
    assert "cloud_prepare_AND_cloud_authorize_NO_GPU" in text
    assert "exit 20" in text
    assert "while !" not in text
    assert "server_launch_v32_g012_paid_gpu_20260831_r1.sh" not in text
    assert "rm -rf" not in text


def test_server_direct_transfer_uses_ephemeral_key_and_partial_rename() -> None:
    text = (
        ROOT / "scripts/server_push_v32_g012_to_compshare_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert "server_to_compshare_ed25519" in text
    assert "BatchMode=yes" in text
    assert "ServerAliveInterval=30" in text
    assert '$(basename "$input_archive").partial' in text
    assert "sha256sum \"$input_archive\"" in text
    assert "rm -rf" not in text


def test_server_transfer_watcher_waits_for_verified_input_and_uses_atomic_rename() -> None:
    text = (
        ROOT / "scripts/server_watch_and_push_v32_g012_input_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert "while ! test -s \"$input_archive\"" in text
    assert 'test "$input_receipt" -nt "$input_archive"' in text
    assert 'stat -c %s "$input_archive"' in text
    assert "cloud bootstrap performs" in text
    assert "sha256sum \"$input_archive\"" not in text
    assert "StrictHostKeyChecking=yes" in text
    assert '"/tmp/$archive_name.partial"' in text
    assert 'mv "/tmp/$archive_name.partial" "/tmp/$archive_name"' in text
    assert "rm -rf" not in text


def test_cloud_transfer_endpoint_is_exact_and_allows_only_sealed_result_pull() -> None:
    text = (
        ROOT / "scripts/cloud_restricted_transfer_endpoint_v32_g012_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert 'case "${SSH_ORIGINAL_COMMAND:-}"' in text
    assert "v32_g012_paid_gpu_training_and_winner_20260831_r1.tar.gz.sha256" in text
    assert "scp -f ./data/CancerLncAtlas/exports/" in text
    assert "REJECTED_FORCED_TRANSFER_COMMAND" in text
    assert "eval " not in text
    assert "rm -rf" not in text


def test_result_pull_watcher_waits_for_receipt_then_verifies_archive() -> None:
    text = (
        ROOT / "scripts/server_watch_and_pull_v32_g012_results_20260831_r1.sh"
    ).read_text(encoding="utf-8")
    assert "until scp" in text
    assert "remote_archive.sha256" in text
    assert "sha256sum \"$archive_partial\"" in text
    assert 'test "$observed" = "$expected"' in text
    assert 'mv "$archive_partial" "$local_archive"' in text
    assert "StrictHostKeyChecking=yes" in text
    assert "rm -rf" not in text


def test_post_lock_sealed_authority_opens_test_sources_only_after_winner_validation() -> None:
    text = (
        ROOT / "scripts/materialize_v32_g012_sealed_test_authority.py"
    ).read_text(encoding="utf-8")
    lock_gate = text.index("_validate_winner_declaration(")
    activity_open = text.index("activity = load_activity(")
    expression_open = text.index("expr_by = {")
    assert lock_gate < activity_open < expression_open
    assert "Winner declaration SHA256 drift" in text
    assert "L1 reconstruction drift" in text
    assert '"contains_train_or_validation_batches": False' in text
    assert '"payloads_opened_before_winner_lock": False' in text
    assert 'str(output / payload_path.name)' in text
    assert 'str(output / keys_path.name)' in text


def test_post_lock_server_launcher_is_hash_gated_resumable_and_variant_dynamic() -> None:
    text = (
        ROOT / "scripts/server_run_v32_g012_postlock_sealed_20260901_r1.sh"
    ).read_text(encoding="utf-8")
    assert "RESULT_ARCHIVE_SHA256_DRIFT" in text
    assert "RESULT_ARCHIVE_MEMBER_DRIFT" in text
    assert "materialize_v32_g012_sealed_test_authority.py" in text
    assert "materialize_v32_winner_locked_sealed_test.py" in text
    assert '--expected-graph-variant "$winner_variant"' in text
    assert "PASS_WINNER_LOCK_SEALED_TEST_INFERENCE" in text
    assert "primary_fold_views_independent_audit" in text
    assert 'if ! test -s "$sealed_authority/SEALED_TEST_MANIFEST.json"' in text
    assert 'if ! test -s "$primary_audit/SUCCESS.json"' in text
    assert "rm -rf" not in text


def test_post_lock_external_router_follows_winner_instead_of_assuming_g2() -> None:
    text = (
        ROOT / "scripts/server_launch_v32_external_router_postlock_20260901_r1.sh"
    ).read_text(encoding="utf-8")
    assert 'winner_variant=$("$python_bin"' in text
    assert "FAIR_INPUT_ACCEPTANCE_WINNER_BINDING_DRIFT" in text
    assert 'a["graph_variant"]==sys.argv[2]' in text
    assert 'a["winner_declaration_sha256"]==sys.argv[3]' in text
    assert "v32-postlock-equal-budget-20260901-r1" in text
    assert "PASS_V32_EXTERNAL_ROUTER_POSTLOCK" in text
    assert 'grep -Fq \'"graph_variant": "G2"\'' not in text
    assert "rm -rf" not in text


def test_post_lock_hierarchical_paid_template_is_capped_and_variant_rendered(
    tmp_path: Path,
) -> None:
    template = (
        ROOT
        / "config/model_v3_2_hierarchical_postlock_paid_gpu_20260901_r1.yaml"
    )
    rendered = tmp_path / "hierarchical.yaml"
    rendered.write_text(
        template.read_text(encoding="utf-8").replace("__GRAPH_VARIANT__", "G1"),
        encoding="utf-8",
    )
    payload = load_structured_mapping(rendered)
    policy = extract_execution_policy(payload)
    assert policy.training_authorized is True
    assert policy.paid_enabled is True
    assert policy.max_paid_hours == 96
    assert policy.max_cost_cny == 210
    assert payload["task_contract"]["graph_variant"] == "G1"
    assert payload["primary_model"]["evidence_integration"]["mode"] == "hierarchical_end_to_end"
    assert payload["primary_model"]["evidence_integration"]["modalities"] == [
        "mutation",
        "cnv",
        "atac",
    ]


def test_post_lock_hierarchical_cloud_launcher_is_resumable_and_fair() -> None:
    text = (
        ROOT / "scripts/cloud_launch_v32_hierarchical_postlock_paid_gpu_20260901_r1.sh"
    ).read_text(encoding="utf-8")
    assert 'winner_variant=$("$python_bin"' in text
    assert '--expected-graph-variant "$winner_variant"' in text
    assert "--endpoint-id paid_gpu" in text
    assert "--hardware-class PAID_PREEMPTIBLE_GPU" in text
    assert "v32-postlock-equal-budget-20260901-r1" in text
    assert "compare_v32_routing_architectures.py" in text
    assert 'if test -s "$task_output/SUCCESS.json"' in text
    assert "PASS_V32_HIERARCHICAL_POSTLOCK_PAID_GPU" in text
    assert "rm -rf" not in text


def test_post_lock_code_stager_preserves_base_snapshot_and_exact_delta() -> None:
    text = (
        ROOT / "scripts/server_stage_v32_postlock_code_20260901_r1.sh"
    ).read_text(encoding="utf-8")
    assert 'cp -a "$source_root/." "$staging_root/"' in text
    assert "POSTLOCK_DELTA_SHA256_DRIFT" in text
    assert "POSTLOCK_DELTA_MEMBER_DRIFT" in text
    assert "PASS_POSTLOCK_CODE_STAGED" in text
    assert 'mv "$staging_root" "$target_root"' in text
    assert "rm -rf" not in text


def test_hierarchical_increment_packager_excludes_the_60gb_prepared_payload() -> None:
    text = (
        ROOT
        / "scripts/server_package_v32_hierarchical_postlock_input_20260901_r1.sh"
    ).read_text(encoding="utf-8")
    assert "PRIMARY_FUSION_FRAME.PRIVATE.parquet" in text
    assert "mutation_cnv_typed_predictions.parquet" in text
    assert "atac_typed_predictions.parquet" in text
    assert "cancer_modality_oof.PRIVATE.parquet" in text
    assert "formal_prepared_20260830_r3_affine_pyg280_localtorch" not in text
    assert "sha256sum \"$archive\"" in text
    assert "rm -rf" not in text


def test_hierarchical_postlock_bootstrap_verifies_both_increment_hashes() -> None:
    text = (
        ROOT / "scripts/cloud_bootstrap_v32_hierarchical_postlock_20260901_r1.sh"
    ).read_text(encoding="utf-8")
    assert "v32_hierarchical_postlock_inputs_20260901_r1.tar" in text
    assert "57b167d93722e165b33b3dbf8e58099aa359ca4437c3255aaa90f34d2448144c" in text
    assert 'sha256sum "$input_archive"' in text
    assert 'sha256sum "$delta_archive"' in text
    assert 'cp -a "$base_code_root/." "$staging/"' in text
    assert "cloud_launch_v32_hierarchical_postlock_paid_gpu_20260901_r1.sh" in text
    assert "rm -rf" not in text


def test_postlock_forced_endpoint_allows_only_exact_increment_and_result_paths() -> None:
    text = (
        ROOT
        / "scripts/cloud_restricted_transfer_endpoint_v32_postlock_20260901_r1.sh"
    ).read_text(encoding="utf-8")
    assert 'case "${SSH_ORIGINAL_COMMAND:-}"' in text
    assert "v32_hierarchical_postlock_inputs_20260901_r1.tar.partial" in text
    assert "v32_postlock_code_delta_20260901_r1.tar.gz.partial" in text
    assert "v32_hierarchical_postlock_paid_gpu_20260901_r1.tar.gz.sha256" in text
    assert "REJECTED_FORCED_TRANSFER_COMMAND" in text
    assert "eval " not in text
    assert "rm -rf" not in text


def test_routing_validation_launcher_uses_one_inner_validation_baseline_and_no_outer() -> None:
    text = (
        ROOT
        / "scripts/cloud_launch_v32_routing_validation_paid_gpu_20260901_r3.sh"
    ).read_text(encoding="utf-8")
    g012_validation = text.index("infer_v32_g012_validation_only.py", text.index("if ! test"))
    external_validation = text.index(
        "run_v32_cancer_modality_router_validation_only.py", g012_validation
    )
    hierarchical_training = text.index("-m cc_hhgt.v32.cli run-shard", external_validation)
    hierarchical_validation = text.index(
        "infer_v32_hierarchical_validation_only.py", hierarchical_training
    )
    winner_lock = text.index(
        "select_v32_routing_validation_winner.py", hierarchical_validation
    )
    assert g012_validation < external_validation < hierarchical_training
    assert hierarchical_training < hierarchical_validation < winner_lock
    assert '--primary-fusion-frame "$primary_validation"' in text
    assert '--primary-validation-predictions "$primary_validation"' in text
    assert "compare_v32_routing_architectures.py" not in text
    assert "infer_v32_hierarchical_candidate.py" not in text
    assert "PRIMARY_FUSION_FRAME.PRIVATE.parquet" not in text
    assert "SEALED_TEST_MANIFEST.json" not in text
    assert "sealed_inference" not in text
    assert "server_run_v32_g012_postlock_sealed" not in text
    assert "PASS_V32_ROUTING_VALIDATION_WINNER" in text
    assert "rm -rf" not in text


def test_routing_validation_paid_template_is_capped_and_outer_free(
    tmp_path: Path,
) -> None:
    template = (
        ROOT
        / "config/model_v3_2_hierarchical_validation_paid_gpu_20260901_r3.yaml"
    )
    rendered = tmp_path / "routing-validation.yaml"
    rendered.write_text(
        template.read_text(encoding="utf-8").replace("__GRAPH_VARIANT__", "G1"),
        encoding="utf-8",
    )
    payload = load_structured_mapping(rendered)
    policy = extract_execution_policy(payload)
    assert policy.training_authorized is True
    assert policy.paid_enabled is True
    assert policy.max_paid_hours == 96
    assert policy.max_cost_cny == 210
    assert payload["task_contract"]["graph_variant"] == "G1"
    assert payload["fair_comparison"]["baseline"] == (
        "locked_g012_inner_validation_predictions"
    )
    assert payload["fair_comparison"]["outer_test_unavailable_during_selection"] is True


def test_routing_outer_branch_runs_only_after_hash_bound_winner() -> None:
    local = (
        ROOT
        / "scripts/server_run_v32_routing_outer_after_winner_lock_20260901_r3.sh"
    ).read_text(encoding="utf-8")
    winner_read = local.index("winner_sha=$(sha256_file")
    branch = local.index('case "$winner" in', winner_read)
    primary = local.index("infer_v32_primary_after_winner_lock.py", branch)
    external = local.index("infer_v32_external_router_after_winner_lock.py", branch)
    assert winner_read < branch < primary < external
    assert "WAITING_FOR_HIERARCHICAL_GPU_OUTER_PACKAGE" in local
    assert "compare_v32_routing_architectures.py" not in local
    assert "rm -rf" not in local

    cloud = (
        ROOT
        / "scripts/cloud_launch_v32_hierarchical_outer_after_winner_lock_20260901_r3.sh"
    ).read_text(encoding="utf-8")
    winner_check = cloud.index('test "$winner_id" = hierarchical_end_to_end')
    preparation = cloud.index(
        "prepare_v32_hierarchical_outer_after_winner_lock.py", winner_check
    )
    inference = cloud.index(
        "infer_v32_hierarchical_after_winner_lock.py", preparation
    )
    assert winner_check < preparation < inference
    assert '--primary-outer-predictions "$primary_outer"' in cloud
    assert "ablate_v32_hierarchical_modalities_after_winner_lock.py" in cloud
    assert "PASS_HIERARCHICAL_MODALITY_ABLATION_AFTER_WINNER_LOCK" in cloud
    assert "PASS_V32_HIERARCHICAL_OUTER_PAID_GPU" in cloud
    assert "rm -rf" not in cloud


def test_routing_transfer_endpoint_is_exact_for_validation_and_locked_outer() -> None:
    text = (
        ROOT
        / "scripts/cloud_restricted_transfer_endpoint_v32_routing_validation_20260901_r3.sh"
    ).read_text(encoding="utf-8")
    assert 'case "${SSH_ORIGINAL_COMMAND:-}"' in text
    assert "v32_routing_validation_inputs_20260901_r3.tar.partial" in text
    assert "v32_routing_validation_paid_gpu_20260901_r3.tar.gz.sha256" in text
    assert "v32_hierarchical_outer_inputs_20260901_r3.tar.partial" in text
    assert "v32_hierarchical_outer_paid_gpu_20260901_r3.tar.gz.sha256" in text
    assert "REJECTED_FORCED_TRANSFER_COMMAND" in text
    assert "eval " not in text
    assert "rm -rf" not in text
