"""Shared-core isolation primitives for the complete V3.2 model.

Every auxiliary capability is a private head trained from a fresh random
initialisation.  It may consume a detached representation produced by the
current V3.2 core, but it cannot update that core or change the exact-pathway
ranking.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class CoreIsolationError(RuntimeError):
    """Raised when an auxiliary task changes or receives gradients into core."""


@dataclass(frozen=True)
class PrivateHeadInitialization:
    module_id: str
    seed: int
    initialization_policy: str
    source_checkpoint_sha256: None
    initial_parameter_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def module_state_sha256(module: Any) -> str:
    """Content hash a Torch module without serialisation metadata."""

    import torch

    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def freeze_v32_core(core: Any) -> str:
    """Freeze a current-generation core and return its pre-training hash."""

    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module_state_sha256(core)


def assert_v32_core_unchanged(core: Any, expected_sha256: str) -> None:
    """Fail if gradients, trainability or parameter drift reached the core."""

    trainable = [name for name, parameter in core.named_parameters() if parameter.requires_grad]
    gradients = [name for name, parameter in core.named_parameters() if parameter.grad is not None]
    if trainable or gradients:
        raise CoreIsolationError(
            f"Frozen V3.2 core isolation failed: trainable={trainable}, gradients={gradients}"
        )
    observed = module_state_sha256(core)
    if observed != str(expected_sha256):
        raise CoreIsolationError(
            f"Frozen V3.2 core parameter hash changed: expected={expected_sha256}, observed={observed}"
        )


def build_private_auxiliary_head(
    module_id: str,
    *,
    core_features: int,
    domain_features: int,
    output_features: int = 1,
    hidden_features: int = 64,
    dropout: float = 0.10,
    seed: int,
) -> tuple[Any, PrivateHeadInitialization]:
    """Build a deterministic private head with no checkpoint-loading path.

    The returned module accepts already computed V3.2 core embeddings.  Its
    forward method detaches those embeddings before concatenating private
    domain features, making the no-gradient rule structural rather than a
    trainer convention.
    """

    import torch
    from torch import nn

    if not module_id or module_id == "exact_pathway":
        raise ValueError("Private auxiliary module_id must be non-empty and non-core")
    dimensions = (core_features, domain_features, output_features, hidden_features)
    if any(int(value) < 1 for value in dimensions):
        raise ValueError("All private-head dimensions must be positive")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    class V32PrivateAuxiliaryHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module_id = str(module_id)
            self.core_features = int(core_features)
            self.domain_features = int(domain_features)
            total = int(core_features) + int(domain_features)
            self.network = nn.Sequential(
                nn.Linear(total, int(hidden_features)),
                nn.LayerNorm(int(hidden_features)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_features), int(output_features)),
            )

        def forward(self, core_embedding, domain_context, availability=None):
            if core_embedding.shape[:-1] != domain_context.shape[:-1]:
                raise ValueError("core_embedding and domain_context batch shapes differ")
            if core_embedding.shape[-1] != self.core_features:
                raise ValueError("Unexpected V3.2 core embedding width")
            if domain_context.shape[-1] != self.domain_features:
                raise ValueError("Unexpected private domain feature width")
            features = torch.cat([core_embedding.detach(), domain_context], dim=-1)
            logits = self.network(features)
            if availability is not None:
                mask = availability.to(dtype=torch.bool)
                if mask.shape == logits.shape[:-1]:
                    mask = mask.unsqueeze(-1)
                if mask.shape != logits.shape:
                    raise ValueError("availability does not align to private-head output")
                logits = torch.where(mask, logits, torch.full_like(logits, float("nan")))
            return logits

    # fork_rng prevents one component's construction from changing another
    # component's global RNG stream while still producing a recorded seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = V32PrivateAuxiliaryHead()
    record = PrivateHeadInitialization(
        module_id=str(module_id),
        seed=int(seed),
        initialization_policy="FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        source_checkpoint_sha256=None,
        initial_parameter_sha256=module_state_sha256(head),
    )
    return head, record


def validate_private_head_checkpoint(metadata: Mapping[str, Any], module: Any) -> None:
    """Validate fresh-init metadata before a private checkpoint can be saved."""

    required = {
        "module_id",
        "seed",
        "initialization_policy",
        "source_checkpoint_sha256",
        "initial_parameter_sha256",
    }
    if missing := sorted(required - set(metadata)):
        raise CoreIsolationError(f"Private-head initialization metadata missing: {missing}")
    if metadata["initialization_policy"] != "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH":
        raise CoreIsolationError("Private head is not attested as fresh V3.2 training")
    if metadata["source_checkpoint_sha256"] is not None:
        raise CoreIsolationError("Private head declares a source checkpoint")
    if not str(metadata["initial_parameter_sha256"]):
        raise CoreIsolationError("Private head lacks its random-initialization hash")
    # The current hash is expected to differ after training, so only the shape
    # and presence of a real module state are checked here.
    if not module_state_sha256(module):
        raise CoreIsolationError("Private head has no hashable model state")
