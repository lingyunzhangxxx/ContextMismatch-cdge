#!/usr/bin/env python3
"""Run official-code-derived ContextMismatch baseline adaptations."""

from scripts.benchmark_v19 import run_external_baselines as runner
from scripts.benchmark_v20.official_adapters import adapter_from_official_checkpoint, finite_checkpoint


runner.STAGE = "qwen3_8b_external_baseline_official_evaluation"
runner.IMPLEMENTATION_PROVENANCE = {
    "method_faithful_adapter": True,
    "official_code_derived_task_adaptation": True,
    "adapter_only": False,
    "unmodified_official_implementation": False,
}
runner.ADAPTER_FACTORY = adapter_from_official_checkpoint
runner.finite_checkpoint = finite_checkpoint


if __name__ == "__main__":
    runner.main()
