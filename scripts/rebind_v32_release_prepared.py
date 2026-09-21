from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pattern", required=True)
    parser.add_argument("--target-pattern", required=True)
    parser.add_argument("--hashes", required=True)
    parser.add_argument("--fold", type=int, action="append", required=True)
    args = parser.parse_args()

    import json
    import torch

    authorized = json.loads(Path(args.hashes).read_text(encoding="utf-8"))
    required_hashes = {
        "code_sha256", "config_sha256", "input_manifest_sha256", "task_manifest_sha256"
    }
    if set(authorized) != required_hashes:
        raise RuntimeError("Authorization hash set is incomplete")

    for fold in args.fold:
        source = Path(args.source_pattern.format(fold=fold))
        target = Path(args.target_pattern.format(fold=fold))
        if target.exists():
            payload = torch.load(target, map_location="cpu", weights_only=False)
            if payload.get("artifact_hashes") != authorized:
                raise RuntimeError(f"Existing target has wrong hashes: {target}")
            print(f"REUSE fold={fold} target={target}", flush=True)
            continue
        payload = torch.load(source, map_location="cpu", weights_only=False)
        scope = payload.get("input_scope", {})
        expected = {
            "source_lnc_n": 16889,
            "shared_lnc_n": 4712,
            "discovery_features_split": "train_patients_only",
            "held_out_effects_as_features": False,
        }
        observed = {key: scope.get(key) for key in expected}
        if observed != expected:
            raise RuntimeError(f"Unsafe prepared parent fold={fold}: {observed}")
        for split in ("train_batches", "validation_batches", "test_batches"):
            if len(payload.get(split, [])) != 403:
                raise RuntimeError(f"Unexpected {split} batch count in fold={fold}")
        payload["prepared_parent_sha256"] = sha256(source)
        payload["prepared_parent_artifact_hashes"] = payload.get("artifact_hashes")
        payload["artifact_hashes"] = authorized
        payload["authorization_rebind"] = {
            "reason": "execution-only convergence extension; data tensors unchanged",
            "data_recomputed": False,
            "source": str(source.resolve()),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        print(f"WROTE fold={fold} target={target} sha256={sha256(target)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
