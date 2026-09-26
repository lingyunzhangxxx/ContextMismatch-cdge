#!/usr/bin/env python3
"""Audit official-code-derived ContextMismatch baseline adaptations."""

from scripts.benchmark_v19 import analyze_external_baselines as analyzer


analyzer.STAGE = "qwen3_8b_external_baseline_official_evaluation"
analyzer.IMPLEMENTATION_PROVENANCE = {
    "method_faithful_adapter": True,
    "official_code_derived_task_adaptation": True,
    "adapter_only": False,
    "unmodified_official_implementation": False,
}


if __name__ == "__main__":
    analyzer.main()
