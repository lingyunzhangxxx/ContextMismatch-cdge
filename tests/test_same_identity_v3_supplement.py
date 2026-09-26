from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 3072
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
METHOD_SHARDS = {
    "operators": ("fixed_negative_vector", "symmetric_rank_one"),
    "experts": ("shared_single_expert", "positive_only_expert"),
    "routing": ("negative_only_expert", "dge_without_structural_routing"),
}

try:
    import torch

    from scripts.benchmark_v4.directional_governance import (
        DirectionalGovernanceSite,
        DirectionalGovernanceSiteEditor,
        DirectionalSingleSiteGovernanceEditor,
    )
    from scripts.benchmark_v1.operators import FixedDirectionTranslation
    from scripts.benchmark_v7.same_identity_variants import DGEAblationSiteEditor

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


class SameIdentitySupplementContractTests(unittest.TestCase):
    def test_frozen_identity_and_complete_method_partition(self) -> None:
        contract = json.loads(
            (ROOT / "protocol/QWEN3_8B_DGE_SAME_IDENTITY_SUPPLEMENT_V1.json").read_text()
        )
        self.assertEqual(contract["scope"]["expected_rows_per_method"], EXPECTED_ROWS)
        self.assertEqual(contract["scope"]["expected_job_key_sha256"], EXPECTED_KEY_SHA256)
        runnable = {
            row["method_id"]
            for row in contract["method_matrix"]
            if row["new_npu_forward_required"]
        }
        self.assertEqual(runnable, {method for shard in METHOD_SHARDS.values() for method in shard})
        self.assertEqual(sum(len(shard) for shard in METHOD_SHARDS.values()), 6)

    def test_consensus_v5_behavior_is_frozen_on_the_same_identity(self) -> None:
        contract = json.loads(
            (
                ROOT
                / "protocol/QWEN3_8B_CONSENSUS_V5_BEHAVIOR_SUPPLEMENT_V1.json"
            ).read_text()
        )
        self.assertEqual(
            contract["status"],
            "frozen_before_any_consensus_v5_behavior_forward",
        )
        self.assertEqual(contract["scope"]["methods"], ["full_dge", "consensus_v5"])
        self.assertEqual(contract["scope"]["expected_rows_per_method"], EXPECTED_ROWS)
        self.assertEqual(contract["scope"]["expected_job_key_sha256"], EXPECTED_KEY_SHA256)
        self.assertTrue(contract["scope"]["shared_baseline_forward_within_each_job"])
        self.assertFalse(contract["audit_gates"]["final_test_open"])
        self.assertEqual(contract["audit_gates"]["final_test_open_count"], 0)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for DGE ablation tests")
class DGEAblationTests(unittest.TestCase):
    def _v3(self) -> DirectionalSingleSiteGovernanceEditor:
        hidden = 8
        editor = DirectionalGovernanceSiteEditor(
            boundary_basis=torch.eye(hidden)[:, :2],
            context_basis=torch.eye(hidden)[:, 2:4],
            positive_output_basis=torch.eye(hidden)[:, :2],
            negative_output_basis=torch.eye(hidden)[:, 6:8],
            boundary_center=torch.zeros(hidden),
            context_center=torch.zeros(hidden),
            boundary_scale=torch.ones(2),
            context_scale=torch.ones(2),
            maximum_relative_correction=0.05,
            activation_threshold=0.5,
            hidden_width=4,
        )
        with torch.no_grad():
            editor.history_head.weight.zero_()
            editor.task_head.weight.zero_()
            for expert in (editor.positive_expert, editor.negative_expert):
                expert.network[-1].weight.zero_()
                expert.network[-1].bias.fill_(1.0)
        return DirectionalSingleSiteGovernanceEditor(
            DirectionalGovernanceSite(27, "mlp", 2, 2), editor
        )

    def test_external_zero_gate_is_exact_identity_for_every_ablation(self) -> None:
        for method in (
            "shared_single_expert",
            "positive_only_expert",
            "negative_only_expert",
            "dge_without_structural_routing",
        ):
            with self.subTest(method=method):
                wrapper = DGEAblationSiteEditor(self._v3(), method)
                value = torch.randn(2, 3, 8)
                result = wrapper(
                    value,
                    0.0,
                    boundary_state=torch.zeros(2, 8),
                    context_state=torch.zeros(2, 3, 8),
                )
                self.assertIs(result, value)

    def test_legacy_baseline_diagnostics_are_optional(self) -> None:
        operator = FixedDirectionTranslation(torch.tensor([1.0, 0.0]), strength=1.0)
        self.assertEqual(getattr(operator, "last_diagnostics", {}), {})

    def test_one_sided_experts_abstain_on_the_opposite_route(self) -> None:
        cases = (
            ("positive_only_expert", -8.0, 8.0),
            ("negative_only_expert", 8.0, -8.0),
        )
        for method, history_bias, task_bias in cases:
            with self.subTest(method=method):
                v3 = self._v3()
                with torch.no_grad():
                    v3.site_editor.history_head.bias.fill_(history_bias)
                    v3.site_editor.task_head.bias.fill_(task_bias)
                wrapper = DGEAblationSiteEditor(v3, method)
                value = torch.randn(2, 3, 8)
                result = wrapper(
                    value,
                    1.0,
                    boundary_state=torch.zeros(2, 8),
                    context_state=torch.zeros(2, 3, 8),
                )
                self.assertTrue(torch.equal(result, value))

    def test_no_structural_router_edits_matched_state(self) -> None:
        v3 = self._v3()
        with torch.no_grad():
            v3.site_editor.history_head.bias.fill_(8.0)
            v3.site_editor.task_head.bias.fill_(8.0)
        wrapper = DGEAblationSiteEditor(v3, "dge_without_structural_routing")
        value = torch.ones(2, 3, 8)
        result = wrapper(
            value,
            1.0,
            boundary_state=torch.zeros(2, 8),
            context_state=torch.zeros(2, 3, 8),
        )
        self.assertFalse(torch.equal(result, value))


if __name__ == "__main__":
    unittest.main()
