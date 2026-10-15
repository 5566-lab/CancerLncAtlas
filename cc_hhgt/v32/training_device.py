"""Explicit CPU execution on 149; never infer CPU permission from absent CUDA."""
import socket


def resolve_training_device(torch, runtime, context):
    requested = runtime.get("device", "cuda")
    if requested == "cpu":
        if (socket.gethostname() != "149" or runtime.get("target_host") != "149"
                or context.get("endpoint_id") != "server149"
                or context.get("hardware_class") != "SERVER149_CPU"):
            raise RuntimeError("CPU training requires the explicit server149 host/endpoint contract")
        if runtime.get("mixed_precision", False) not in (False, "fp32", "none"):
            raise RuntimeError("Server149 CPU training requires FP32")
        return "cpu"
    if requested != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Authorized CC-HHGT training requires CUDA; CPU fallback is forbidden")
    return "cuda"


def cuda_memory_event(torch, device, event):
    if event not in {"reset_peak_memory_stats", "empty_cache", "max_memory_reserved"}:
        raise ValueError("Unknown CUDA memory event")
    if device == "cpu": return 0
    return getattr(torch.cuda, event)()
