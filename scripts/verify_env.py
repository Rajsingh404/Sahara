"""Small, dependency-only verification for the SAHARA development environment."""
from __future__ import annotations

import importlib
import sys


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        checks.append(("tensorflow", True, f"GPU devices: {gpus}"))
    except Exception as exc:
        checks.append(("tensorflow", False, repr(exc)))
        gpus = []
    for name in ("librosa", "tensorflow_hub", "desed", "yt_dlp"):
        try:
            module = importlib.import_module(name)
            checks.append((name, True, getattr(module, "__version__", "imported")))
        except Exception as exc:
            checks.append((name, False, repr(exc)))
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    gpu_ok = any(device.device_type == "GPU" for device in gpus)
    print(f"{'PASS' if all(ok for _, ok, _ in checks) else 'FAIL'}  imports")
    print(f"{'PASS' if gpu_ok else 'WARN'}  Metal GPU {'detected' if gpu_ok else 'not detected'}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
