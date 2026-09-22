"""Environment fingerprint attached to every benchmark result file.

Research reproducibility requires recording *what* ran, not just the numbers:
platform, CPU, core counts, library versions, and thread/dtype configuration.
``Environment.snapshot()`` is embedded verbatim in runner output. All probing
is pure filesystem parsing — no shell execution.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"arch": platform.machine()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        models = [line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                  if line.startswith("model name")]
        info["model"] = models[0] if models else platform.processor()
        info["logical_cpus"] = sum(1 for line in cpuinfo.splitlines()
                                   if line.startswith("processor"))
        core_ids = set()
        cpu_root = Path("/sys/devices/system/cpu")
        for core_id_file in sorted(cpu_root.glob("cpu[0-9]*/topology/core_id")):
            core_ids.add(core_id_file.read_text().strip())
        if core_ids:
            info["physical_cores"] = len(core_ids)
    except Exception:  # pragma: no cover - best effort only
        pass
    return info


def _lib_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("numpy", "soundfile", "sherpa-onnx", "piper-tts", "fastapi",
                "torch", "qwen-asr", "qwen-tts", "transformers", "onnxruntime"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def _runtime_state() -> dict[str, Any]:
    """Isolation evidence: which CPUs this process may run on, and ambient load.

    Benchmarks on a shared machine are only comparable when the affinity mask
    and load average at measurement time are recorded (see
    docs/METHODOLOGY.md §6 Run-to-run stability).
    """
    state: dict[str, Any] = {}
    try:
        state["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        state["cpu_affinity"] = None
    try:
        load1, _, _ = os.getloadavg()
        state["loadavg_1min"] = round(load1, 2)
    except OSError:
        pass
    return state


def snapshot() -> dict[str, Any]:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_note": "see docs/METHODOLOGY.md",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_info(),
        "libraries": _lib_versions(),
        "runtime": _runtime_state(),
    }
