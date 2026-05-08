"""
Benchmark utilities for Cathey LLM quantization evaluation.

Exports:
  BENCHMARK_CASES  — 20 fixed intent-classification cases (5 per type)
  evaluate()       — compute type/command accuracy and latency metrics
  format_row()     — format one model row for the results table
  run_benchmark()  — run the full suite against an LLMParser instance
"""

from __future__ import annotations
import time
import numpy as np
from typing import Any

# ── 20 fixed benchmark cases ──────────────────────────────────────────────────
# 5 direct_command · 5 needs_clarification · 5 general_qa · 5 invalid

BENCHMARK_CASES: list[dict[str, Any]] = [
    # ── direct_command (5) ───────────────────────────────────────────────────
    {
        "input":           "Cathey, turn on the light.",
        "expected_type":   "direct_command",
        "expected_device": "light",
        "expected_action": "turn_on",
    },
    {
        "input":           "Cathey, turn off the AC.",
        "expected_type":   "direct_command",
        "expected_device": "ac",
        "expected_action": "turn_off",
    },
    {
        "input":           "Cathey, open the curtains.",
        "expected_type":   "direct_command",
        "expected_device": "curtain",
        "expected_action": "open",
    },
    {
        "input":           "Cathey, set the temperature to 24 degrees.",
        "expected_type":   "direct_command",
        "expected_device": "ac",
        "expected_action": "set_temperature",
    },
    {
        "input":           "Cathey, set the brightness to 80.",
        "expected_type":   "direct_command",
        "expected_device": "light",
        "expected_action": "set_brightness",
    },

    # ── needs_clarification (5) ──────────────────────────────────────────────
    {
        "input":           "Cathey, it's a bit dark in here.",
        "expected_type":   "needs_clarification",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, I feel cold.",
        "expected_type":   "needs_clarification",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, it's too bright.",
        "expected_type":   "needs_clarification",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, I'm not comfortable.",
        "expected_type":   "needs_clarification",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, the room feels stuffy.",
        "expected_type":   "needs_clarification",
        "expected_device": None,
        "expected_action": None,
    },

    # ── general_qa (5) ───────────────────────────────────────────────────────
    {
        "input":           "Cathey, what's the capital of France?",
        "expected_type":   "general_qa",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, how do I make pasta?",
        "expected_type":   "general_qa",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, tell me a joke.",
        "expected_type":   "general_qa",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, what's the weather like today?",
        "expected_type":   "general_qa",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Cathey, how many calories are in an apple?",
        "expected_type":   "general_qa",
        "expected_device": None,
        "expected_action": None,
    },

    # ── invalid (5) ──────────────────────────────────────────────────────────
    {
        "input":           "Turn on the light.",          # no wake word
        "expected_type":   "invalid",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Never mind.",
        "expected_type":   "invalid",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Um.",
        "expected_type":   "invalid",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Okay.",
        "expected_type":   "invalid",
        "expected_device": None,
        "expected_action": None,
    },
    {
        "input":           "Hello there.",
        "expected_type":   "invalid",
        "expected_device": None,
        "expected_action": None,
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute accuracy and latency metrics from a list of per-case result dicts.

    Each dict must contain:
      expected_type, predicted_type,
      expected_device, predicted_device,
      expected_action, predicted_action,
      latency_ms
    """
    n = len(results)
    if n == 0:
        return {"type_acc": 0.0, "cmd_acc": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}

    type_correct = sum(
        1 for r in results if r["predicted_type"] == r["expected_type"]
    )

    direct_cases = [r for r in results if r["expected_type"] == "direct_command"]
    if direct_cases:
        cmd_correct = sum(
            1 for r in direct_cases
            if r["predicted_device"] == r["expected_device"]
            and r["predicted_action"] == r["expected_action"]
        )
        cmd_acc = cmd_correct / len(direct_cases)
    else:
        cmd_acc = 0.0

    latencies = [r["latency_ms"] for r in results]
    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))

    return {
        "type_acc": type_correct / n,
        "cmd_acc":  cmd_acc,
        "avg_ms":   avg_ms,
        "p95_ms":   p95_ms,
    }


def format_row(model_name: str, metrics: dict[str, float], size_gb: float) -> str:
    """Format one model's results as a markdown table row."""
    return (
        f"| {model_name} "
        f"| {metrics['type_acc'] * 100:.0f}% "
        f"| {metrics['cmd_acc'] * 100:.0f}% "
        f"| {metrics['avg_ms']:.0f} "
        f"| {metrics['p95_ms']:.0f} "
        f"| {size_gb} |"
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(llm, n_runs: int = 3) -> list[dict[str, Any]]:
    """Run BENCHMARK_CASES against an LLMParser instance.

    Each case is run n_runs times; median latency is recorded.
    Returns a list of result dicts suitable for evaluate().
    """
    results = []
    for case in BENCHMARK_CASES:
        latencies = []
        last_parsed = {}
        for _ in range(n_runs):
            t0 = time.perf_counter()
            parsed = llm.parse_unified(case["input"])
            latencies.append((time.perf_counter() - t0) * 1000)
            last_parsed = parsed or {}

        latencies.sort()
        median_ms = latencies[len(latencies) // 2]

        results.append({
            "input":            case["input"],
            "expected_type":    case["expected_type"],
            "expected_device":  case["expected_device"],
            "expected_action":  case["expected_action"],
            "predicted_type":   last_parsed.get("type"),
            "predicted_device": last_parsed.get("device"),
            "predicted_action": last_parsed.get("action"),
            "latency_ms":       median_ms,
        })
    return results
