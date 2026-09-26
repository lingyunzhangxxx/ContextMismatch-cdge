from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch

    from scripts.benchmark_v4.directional_governance import (
        DirectionalGovernanceSite,
        DirectionalGovernanceSiteEditor,
        DirectionalSingleSiteGovernanceEditor,
    )
    from scripts.benchmark_v5.abstaining_directional_governance import (
        APPLICATION_FEATURE_WIDTH,
        ApplicationVetoHead,
    )
    from scripts.benchmark_v6.consensus_directional_governance import (
        ConsensusDirectionalSiteEditor,
        editor_from_checkpoint,
    )
    from scripts.benchmark_v6.fit_consensus_router import (
        Dataset,
        _checkpoint_contains_no_base_weights,
        _train_head,
    )

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


class ConsensusRouterV5ContractTests(unittest.TestCase):
    def test_fit_contract_is_output_blind_and_forbids_consumed_folds(self) -> None:
        router_path = (
            ROOT / "protocol/QWEN3_8B_GROUP_ROBUST_CONSENSUS_GOVERNANCE_EDITOR_V5.json"
        )
        fit = json.loads(
            (
                ROOT
                / "protocol/QWEN3_8B_GROUP_ROBUST_CONSENSUS_GOVERNANCE_FIT_V5.json"
            ).read_text()
        )
        import hashlib

        self.assertEqual(
            fit["router_contract_sha256"],
            hashlib.sha256(router_path.read_bytes()).hexdigest(),
        )
        self.assertFalse(fit["capture_outputs_examined_before_freeze"])
        self.assertEqual(fit["training"]["allowed_subspace_hash_folds"], list(range(6)))
        self.assertEqual(fit["training"]["forbidden_subspace_hash_folds"], [6, 7])
        self.assertEqual(fit["cross_fit"]["safety_margin_standardized_units"], 0.25)
        for field in (
            "architecture_search_forbidden",
            "seed_search_forbidden",
            "epoch_search_forbidden",
            "threshold_search_forbidden",
        ):
            self.assertTrue(fit[field])


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for V5 editor tests")
class ConsensusDirectionalGovernanceV5Tests(unittest.TestCase):
    def _v3(self) -> DirectionalSingleSiteGovernanceEditor:
        hidden = 64
        site_editor = DirectionalGovernanceSiteEditor(
            boundary_basis=torch.eye(hidden)[:, :16],
            context_basis=torch.eye(hidden)[:, 16:48],
            positive_output_basis=torch.eye(hidden)[:, :16],
            negative_output_basis=torch.eye(hidden)[:, 48:64],
            boundary_center=torch.zeros(hidden),
            context_center=torch.zeros(hidden),
            boundary_scale=torch.ones(16),
            context_scale=torch.ones(32),
            maximum_relative_correction=0.1,
            activation_threshold=0.5,
            hidden_width=16,
        )
        with torch.no_grad():
            site_editor.history_head.weight.zero_()
            site_editor.task_head.weight.zero_()
            site_editor.history_head.bias.fill_(8.0)
            site_editor.task_head.bias.fill_(-8.0)
            site_editor.positive_expert.network[-1].weight.zero_()
            site_editor.positive_expert.network[-1].bias.fill_(1.0)
            site_editor.negative_expert.network[-1].weight.zero_()
            site_editor.negative_expert.network[-1].bias.fill_(-1.0)
        return DirectionalSingleSiteGovernanceEditor(
            DirectionalGovernanceSite(27, "mlp", 16, 16), site_editor
        )

    def _head(self, bias: float) -> ApplicationVetoHead:
        head = ApplicationVetoHead(
            feature_center=torch.zeros(APPLICATION_FEATURE_WIDTH),
            feature_scale=torch.ones(APPLICATION_FEATURE_WIDTH),
            hidden_widths=[],
            threshold_logit=0.0,
        )
        with torch.no_grad():
            head.network[-1].weight.zero_()
            head.network[-1].bias.fill_(bias)
        return head

    def _v3_checkpoint(self, v3: DirectionalSingleSiteGovernanceEditor) -> dict:
        editor = v3.site_editor
        return {
            "schema_version": 2,
            "kind": "directional_structural_governance_editor",
            "method": "DSGE-V3",
            "base_model_weights_included": False,
            "trainable_parameter_count": v3.trainable_parameter_count,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
            "sites": [
                {
                    "layer": 27,
                    "component": "mlp",
                    "positive_output_rank": 16,
                    "negative_output_rank": 16,
                    "maximum_relative_correction": editor.maximum_relative_correction,
                    "activation_threshold": editor.activation_threshold,
                    "hidden_width": editor.hidden_width,
                    "head_input_mode": editor.head_input_mode,
                    "constructor_tensors": {
                        "boundary_basis": editor.boundary_basis.detach().clone(),
                        "context_basis": editor.context_basis.detach().clone(),
                        "positive_output_basis": editor.positive_output_basis.detach().clone(),
                        "negative_output_basis": editor.negative_output_basis.detach().clone(),
                        "boundary_center": editor.boundary_center.detach().clone(),
                        "context_center": editor.context_center.detach().clone(),
                        "boundary_scale": editor.boundary_scale.detach().clone(),
                        "context_scale": editor.context_scale.detach().clone(),
                        "boundary_head_scale": editor.boundary_head_scale,
                        "context_head_scale": editor.context_head_scale,
                        "history_logit_location": float(editor.history_logit_location),
                        "history_logit_scale": float(editor.history_logit_scale),
                        "task_logit_location": float(editor.task_logit_location),
                        "task_logit_scale": float(editor.task_logit_scale),
                    },
                    "state_dict": {
                        name: value.detach().clone()
                        for name, value in editor.state_dict().items()
                    },
                }
            ],
        }

    def _checkpoint(self) -> dict:
        v3_checkpoint = self._v3_checkpoint(self._v3())
        records = {}
        for name, bias in (("all_negative", 8.0), ("matched_state", 8.0)):
            head = self._head(bias)
            records[name] = {
                "architecture": "linear",
                "seed": 1,
                "feature_center": head.feature_center.detach().clone(),
                "feature_scale": head.feature_scale.detach().clone(),
                "threshold_logit": float(head.threshold_logit),
                "state_dict": {
                    key: value.detach().clone() for key, value in head.state_dict().items()
                },
            }
        return {
            "schema_version": 1,
            "kind": "group_robust_consensus_directional_governance_editor",
            "method": "GRC-DGE-V5",
            "v3_checkpoint": v3_checkpoint,
            "v3_expert_frozen": True,
            "application_heads": records,
            "application_trainable_parameter_count": 2 * (APPLICATION_FEATURE_WIDTH + 1),
            "base_model_weights_included": False,
            "operator_dev_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }

    def test_zero_external_gate_returns_original_object(self) -> None:
        wrapper = ConsensusDirectionalSiteEditor(
            self._v3(), self._head(8.0), self._head(8.0)
        )
        value = torch.randn(2, 3, 64)
        result = wrapper(
            value,
            0.0,
            boundary_state=torch.zeros(2, 64),
            context_state=torch.zeros(2, 3, 64),
        )
        self.assertIs(result, value)
        self.assertEqual(wrapper.last_diagnostics, {})

    def test_either_head_veto_is_bitwise_identity(self) -> None:
        for biases in ((-8.0, 8.0), (8.0, -8.0), (-8.0, -8.0)):
            with self.subTest(biases=biases):
                wrapper = ConsensusDirectionalSiteEditor(
                    self._v3(), self._head(biases[0]), self._head(biases[1])
                )
                value = torch.randn(2, 3, 64)
                result = wrapper(
                    value,
                    1.0,
                    boundary_state=torch.zeros(2, 64),
                    context_state=torch.zeros(2, 3, 64),
                )
                self.assertTrue(torch.equal(result, value))
                self.assertFalse(
                    bool(wrapper.last_diagnostics["consensus_application_active"].any())
                )

    def test_both_heads_must_accept_before_v3_edit_is_applied(self) -> None:
        wrapper = ConsensusDirectionalSiteEditor(
            self._v3(), self._head(8.0), self._head(8.0)
        )
        value = torch.ones(2, 3, 64)
        result = wrapper(
            value,
            1.0,
            boundary_state=torch.zeros(2, 64),
            context_state=torch.zeros(2, 3, 64),
        )
        self.assertFalse(torch.equal(result, value))
        self.assertTrue(bool(wrapper.last_diagnostics["v5_gate_active"].all()))

    def test_checkpoint_reload_has_two_linear_heads_and_frozen_v3(self) -> None:
        editor = editor_from_checkpoint(self._checkpoint())
        self.assertEqual(editor.trainable_parameter_count, 2 * (APPLICATION_FEATURE_WIDTH + 1))
        self.assertTrue(all(not value.requires_grad for value in editor.frozen_v3.parameters()))
        self.assertEqual(
            tuple(editor.site_editor.all_negative_head.hidden_widths), ()
        )
        self.assertEqual(
            tuple(editor.site_editor.matched_state_head.hidden_widths), ()
        )

    def test_nested_v3_checkpoint_base_weight_scan_fails_closed(self) -> None:
        checkpoint = self._v3_checkpoint(self._v3())
        self.assertTrue(_checkpoint_contains_no_base_weights(checkpoint))
        checkpoint["sites"][0]["state_dict"]["model.embed_tokens.weight"] = torch.zeros(1)
        self.assertFalse(_checkpoint_contains_no_base_weights(checkpoint))

    def test_held_fold_features_do_not_change_trained_head(self) -> None:
        generator = torch.Generator().manual_seed(19)
        features = torch.randn(36, APPLICATION_FEATURE_WIDTH, generator=generator)
        target = torch.tensor(([0.0, 1.0] * 18), dtype=torch.float32)
        folds = torch.arange(36, dtype=torch.long) % 6
        sources = tuple(
            "governance_mismatch" if value else "governance_matched:matched_verification"
            for value in target.tolist()
        )
        data = Dataset(
            features=features,
            target=target,
            weights=torch.ones(36),
            folds=folds,
            sources=sources,
            directions=tuple("positive" if value else "none" for value in target.tolist()),
            label_swaps=tuple(index % 2 for index in range(36)),
        )
        held = folds == 5
        changed = Dataset(
            features=features.clone(),
            target=data.target,
            weights=data.weights,
            folds=data.folds,
            sources=data.sources,
            directions=data.directions,
            label_swaps=data.label_swaps,
        )
        changed.features[held] += 100.0
        indices = torch.where(~held)[0]
        config = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 8,
            "epochs": 3,
            "gradient_norm_clip": 1.0,
        }
        first, first_report, _ = _train_head(
            data=data,
            indices=indices,
            seed=57721,
            generator_seed=57721 + 1009 * 5,
            config=config,
        )
        second, second_report, _ = _train_head(
            data=changed,
            indices=indices,
            seed=57721,
            generator_seed=57721 + 1009 * 5,
            config=config,
        )
        self.assertEqual(first_report, second_report)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
