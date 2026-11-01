"""Physical-device routing, independent of compiler product aliases.

The 950 path is an experimental Triton-Ascend port. Never load a dav-2201
extension on it, and never infer the physical model from an environment label.
"""
from functools import lru_cache
import os
import re


def classify_device(name):
    compact = re.sub(r"[\s_-]", "", name).lower()
    if re.fullmatch(r"ascend950[a-z0-9]*", compact):
        return "npu950"
    if compact == "ascend9109382":
        return "npu"
    raise RuntimeError(f"Unreviewed Ascend device {name!r}; expected Ascend910_9382 or Ascend950 series")


@lru_cache(None)
def _device_backend(index):
    import torch
    return classify_device(torch.npu.get_device_name(index))


def device_backend(device):
    import torch
    device = torch.device(device)
    if device.type != "npu":
        raise ValueError("An NPU device is required")
    index = torch.npu.current_device() if device.index is None else device.index
    return _device_backend(index)


def require_950(device="npu"):
    if device_backend(device) != "npu950":
        raise RuntimeError("This experiment requires a physical Ascend950; no substitute device is allowed")


def variant():
    value = os.environ.get('FDR_NPU950_VARIANT', 'combined')
    if value not in ('residual', 'matrix', 'combined'):
        raise ValueError('FDR_NPU950_VARIANT must be residual, matrix or combined')
    return value
