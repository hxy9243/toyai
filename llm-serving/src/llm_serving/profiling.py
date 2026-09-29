"""Read saved worker traces without requiring PyTorch on the analysis machine."""

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


def trace_rows(run_dir: str | Path) -> list[dict]:
    """Inventory real trace files, retaining case, retry attempt and workload."""
    root = Path(run_dir)
    rows = []
    for path in sorted((root / "cases").glob("*/profiling/attempt-*/*/*.pt.trace.json*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rows.append({
            "case_id": relative.parts[1],
            "attempt": relative.parts[3],
            "workload": relative.parts[4],
            "trace": path.name,
            "size_mb": path.stat().st_size / 1_000_000,
            "path": str(path),
        })
    return rows


def kernel_rows(trace_path: str | Path, *, limit: int = 20) -> list[dict]:
    """Rank GPU kernels by summed duration within ONE worker trace.

    Durations can overlap across streams: totals are neither wall time nor GPU
    utilization. PyTorch operator self time is a different measurement. Loading
    JSON decompresses the whole trace, so callers should select a small capture.
    """
    path = Path(trace_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    totals = defaultdict(lambda: [0, 0.0])
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        duration = event.get("dur")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            continue
        key = (str(event.get("args", {}).get("device", event.get("pid", "unknown"))), event.get("name", "unknown"))
        totals[key][0] += 1
        totals[key][1] += duration
    by_device = defaultdict(float)
    for (device, _), (_, duration) in totals.items():
        by_device[device] += duration
    ordered = sorted(totals.items(), key=lambda item: item[1][1], reverse=True)
    return [{
        "device": device,
        "kernel": name,
        "calls": count,
        "total_ms": duration / 1000,
        "mean_us": duration / count,
        "device_kernel_time_pct": 100 * duration / by_device[device] if by_device[device] else 0,
    } for (device, name), (count, duration) in ordered[:limit]]
