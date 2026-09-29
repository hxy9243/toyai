"""Per-invocation prefix-cache counter deltas from vLLM Prometheus snapshots."""

import math
import re


def _series(text, metric):
    result = {}
    pattern = re.compile(r'^' + re.escape(metric) + r'(?:_total)?(\{.*\})?\s+(\S+)(?:\s+\S+)?$')
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid counter")
            result[match[1] or ""] = value
    return result


def cache_counter_delta(before: str, after: str) -> dict:
    """Never turn absent, reset, or invalid counters into a zero hit rate."""
    result = {"hit_tokens": None, "query_tokens": None, "hit_rate": None,
              "status": "unavailable", "scope": "benchmark invocation including warmups and probe"}
    try:
        deltas = []
        for metric in ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries"):
            start, end = _series(before, metric), _series(after, metric)
            if not start or start.keys() != end.keys():
                return result
            values = [end[key] - start[key] for key in start]
            if any(v < 0 for v in values):
                return {**result, "status": "counter_reset"}
            deltas.append(sum(values))
        hits, queries = deltas
        if hits > queries:
            return {**result, "status": "invalid_delta"}
        return {**result, "hit_tokens": hits, "query_tokens": queries,
                "hit_rate": hits / queries if queries else None,
                "status": "ok" if queries else "no_queries"}
    except ValueError:
        return {**result, "status": "invalid_counter"}
