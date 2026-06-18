import torch


def resolve_device(device=None):
    if device is None:
        device = "auto"

    device = str(device).strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"

    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            return "cpu"
        raise ValueError(
            "device='mps' is not supported because UnIPH uses torch.float64, "
            "which PyTorch MPS does not support. Use device='cpu'."
        )

    return device
