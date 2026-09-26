from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.benchmark_v7.materialize_pair_ge_fit_authorization_v55 import (
    FIT_SHA256,
    ROUTER_SHA256,
    SELECTION_CONTRACT,
    _validate_contracts,
    _validate_selection,
    _validate_v5_1,
)


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch

    from scripts.benchmark_v4.directional_governance import (
        DirectionalGovernanceSite,
        DirectionalGovernanceSiteEditor,
        DirectionalSingleSiteGovernanceEditor,
    )
    from scripts.benchmark_v7.fit_paired_interaction_router import (
        PairDataset,
        _train_router,
    )
    from scripts.benchmark_v7.paired_interaction_governance import (
        FEATURE_SCHEMA,
        PAIR_INTERACTION_FEATURE_WIDTH,
        PairedInteractionGovernanceEditor,
        PairedInteractionRouter,
        PairedInteractionSite,
        PairedInteractionSiteEditor,
        editor_from_checkpoint,
        expand_pair_interaction_features,
    )

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


class PairGEV52ContractTests(unittest.TestCase):
    def _contracts(self):
        protocol = ROOT / "protocol"
        router_path = protocol / "QWEN3_8B_PAIRED_INTERACTION_GOVERNANCE_EDITOR_V5_2.json"
        fit_path = protocol / "QWEN3_8B_PAIRED_INTERACTION_GOVERNANCE_FIT_V5_2.json"
        return json.loads(router_path.read_text()), json.loads(fit_path.read_text())

    def test_frozen_contract_hashes_and_method(self):
        import hashlib

        router, fit = self._contracts()
        router_path = (
            ROOT / "protocol/QWEN3_8B_PAIRED_INTERACTION_GOVERNANCE_EDITOR_V5_2.json"
        )
        fit_path = ROOT / "protocol/QWEN3_8B_PAIRED_INTERACTION_GOVERNANCE_FIT_V5_2.json"
        self.assertEqual(hashlib.sha256(router_path.read_bytes()).hexdigest(), ROUTER_SHA256)
        self.assertEqual(hashlib.sha256(fit_path.read_bytes()).hexdigest(), FIT_SHA256)
        _validate_contracts(router, fit)
        self.assertEqual(router["application_features"]["expanded_input_width"], 662)
        self.assertEqual(router["router"]["shared_trunk_seed"], 58121)
        self.assertTrue(fit["component_discovery_is_developmental_only"])
        self.assertTrue(fit["operator_dev_untouched_confirmatory_selection_required"])

    def test_contract_mutation_is_rejected(self):
        router, fit = self._contracts()
        fit = copy.deepcopy(fit)
        fit["training"]["epochs"] += 1
        with self.assertRaisesRegex(ValueError, "epoch contract"):
            _validate_contracts(router, fit)

    def test_cpu_selector_contract_is_exact(self):
        selection = {
            "node": "a06",
            "state": "IDLE",
            "cpu_free": 192,
            "mem_free_mib": 2_000_000,
            "selection_contract": copy.deepcopy(SELECTION_CONTRACT),
        }
        _validate_selection(selection, "a06")
        selection["selection_contract"]["npus"] = 1
        with self.assertRaisesRegex(ValueError, "resource contract"):
            _validate_selection(selection, "a06")

    def test_v5_1_must_be_a_falsifier(self):
        report = {"method": "GRC-DGE-V5", "fit_eligible": False}
        receipt = {
            "job_id": 9205,
            "archive_sha256": "6e038c155b88410a0065d3d504faf505e706751c7f4083956729b252582f0778",
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
        }
        _validate_v5_1(report, receipt)
        report["fit_eligible"] = True
        with self.assertRaisesRegex(ValueError, "falsifier"):
            _validate_v5_1(report, receipt)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for PAIR-GE tests")
class PairGEV52TorchTests(unittest.TestCase):
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

    def _router(self, biases=(8.0, 8.0)) -> PairedInteractionRouter:
        router = PairedInteractionRouter(
            feature_center=torch.zeros(PAIR_INTERACTION_FEATURE_WIDTH),
            feature_scale=torch.ones(PAIR_INTERACTION_FEATURE_WIDTH),
            hidden_widths=[96, 48],
        )
        with torch.no_grad():
            for parameter in router.parameters():
                parameter.zero_()
            for index, name in enumerate(("all_negative", "matched_state")):
                router.output_heads[name].bias.fill_(biases[index])
        return router

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
        v3 = self._v3()
        router = self._router()
        return {
            "schema_version": 1,
            "kind": "paired_interaction_governance_editor",
            "method": "PAIR-GE-V5.2",
            "feature_schema": FEATURE_SCHEMA,
            "v3_checkpoint": self._v3_checkpoint(v3),
            "v3_expert_frozen": True,
            "router": {
                "feature_center": router.feature_center.detach().clone(),
                "feature_scale": router.feature_scale.detach().clone(),
                "threshold_logits": router.threshold_logits.detach().clone(),
                "hidden_widths": list(router.hidden_widths),
                "state_dict": {
                    name: value.detach().clone()
                    for name, value in router.state_dict().items()
                },
            },
            "application_trainable_parameter_count": sum(
                parameter.numel() for parameter in router.parameters()
            ),
            "base_model_weights_included": False,
            "operator_dev_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }

    def test_fixed_interaction_width_and_values(self):
        base = torch.arange(2 * 54, dtype=torch.float32).reshape(2, 54) / 100.0
        expanded = expand_pair_interaction_features(base)
        self.assertEqual(expanded.shape, (2, 662))
        boundary = base[:, :16].repeat(1, 2)
        context = base[:, 16:48]
        self.assertTrue(torch.equal(expanded[:, 54:86], boundary * context))
        self.assertTrue(torch.equal(expanded[:, 86:118], torch.abs(boundary - context)))

    def test_zero_external_gate_is_exact_identity(self):
        wrapper = PairedInteractionSiteEditor(self._v3(), self._router())
        value = torch.randn(2, 3, 64)
        result = wrapper(
            value,
            0.0,
            boundary_state=torch.zeros(2, 64),
            context_state=torch.zeros(2, 3, 64),
        )
        self.assertIs(result, value)
        self.assertEqual(wrapper.last_diagnostics, {})

    def test_either_output_head_veto_is_exact_identity(self):
        for biases in ((-8.0, 8.0), (8.0, -8.0), (-8.0, -8.0)):
            with self.subTest(biases=biases):
                wrapper = PairedInteractionSiteEditor(self._v3(), self._router(biases))
                value = torch.randn(2, 3, 64)
                result = wrapper(
                    value,
                    1.0,
                    boundary_state=torch.zeros(2, 64),
                    context_state=torch.zeros(2, 3, 64),
                )
                self.assertTrue(torch.equal(result, value))

    def test_checkpoint_reload_has_one_trunk_two_heads_and_frozen_v3(self):
        editor = editor_from_checkpoint(self._checkpoint())
        self.assertIsInstance(editor, PairedInteractionGovernanceEditor)
        self.assertEqual(tuple(editor.site_editor.router.hidden_widths), (96, 48))
        self.assertEqual(
            set(editor.site_editor.router.output_heads),
            {"all_negative", "matched_state"},
        )
        self.assertTrue(all(not value.requires_grad for value in editor.frozen_v3.parameters()))

    def test_held_fold_features_do_not_change_training(self):
        generator = torch.Generator().manual_seed(17)
        features = torch.randn(
            24, PAIR_INTERACTION_FEATURE_WIDTH, generator=generator
        )
        target = torch.tensor([value for _ in range(12) for value in (0.0, 1.0)])
        folds = torch.tensor([index % 6 for index in range(12) for _ in range(2)])
        sources = tuple(
            "governance_mismatch"
            if value > 0.5
            else "governance_matched:matched_verification"
            for value in target.tolist()
        )
        directions = tuple("positive" if value > 0.5 else "none" for value in target)
        swaps = tuple(index % 2 for index in range(12) for _ in range(2))
        pairs = tuple(f"pair-{index}" for index in range(12) for _ in range(2))
        data = PairDataset(
            features=features,
            target=target,
            weights=torch.ones(24),
            folds=folds,
            sources=sources,
            directions=directions,
            label_swaps=swaps,
            pair_keys=pairs,
        )
        config = {
            "hidden_widths": [96, 48],
            "learning_rate": 0.0015,
            "weight_decay": 0.0001,
            "epochs": 2,
            "gradient_norm_clip": 1.0,
            "worst_group_temperature": 0.25,
            "row_bce_weight": 1.0,
            "worst_group_weight": 0.5,
            "paired_ranking_weight": 1.0,
            "paired_ranking_margin_logit": 2.0,
            "negative_tail_separation_weight": 0.5,
            "negative_tail_top_k": 4,
            "negative_tail_margin_logit": 1.0,
        }
        indices = torch.where(folds != 5)[0]
        first, _, _ = _train_router(data=data, indices=indices, seed=99, config=config)
        modified = PairDataset(
            features=features.clone(),
            target=data.target,
            weights=data.weights,
            folds=data.folds,
            sources=data.sources,
            directions=data.directions,
            label_swaps=data.label_swaps,
            pair_keys=data.pair_keys,
        )
        modified.features[folds == 5] += 10_000.0
        second, _, _ = _train_router(
            data=modified, indices=indices, seed=99, config=config
        )
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
