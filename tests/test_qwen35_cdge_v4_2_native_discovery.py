import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_v14.native_discovery import (
    build_scan_plan,
    materialize_candidate_grid,
    require_sha_bound_path,
    scan_factorial_axes,
    select_native_sites,
    smoke_scan_axes,
    validate_checkpoint_against_candidate,
    value_sha256,
)


class NativeDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            Path("protocol/QWEN3_5_9B_CDGE_V4_2_NATIVE_DISCOVERY_V1.json").read_text()
        )
        self.model = {"architecture": {"num_hidden_layers": 32}}

    def _rows(self, components):
        rows = []
        for layer, component in components:
            score = 0.1 + layer / 100.0 + (0.03 if component == "mlp" else 0.0)
            for benchmark in ("a", "b", "c", "d", "e", "f"):
                for direction, sign in (("obedience", 1.0), ("verification", -1.0)):
                    for swap in (0, 1):
                        rows.append({
                            "job_key": f"{layer}-{component}-{benchmark}-{direction}-{swap}-1",
                            "layer": layer,
                            "component": component,
                            "benchmark": benchmark,
                            "source_regime": direction,
                            "label_swap": swap,
                            "patch_coefficient": 1.0,
                            "patch_effect": sign * score,
                            "target_gap": sign,
                            "source_cached_margin": 0.0,
                            "correct_logit_margin": sign * score,
                        })
            rows.append({
                "job_key": f"{layer}-{component}-zero",
                "layer": layer,
                "component": component,
                "benchmark": "a",
                "source_regime": "obedience",
                "label_swap": 0,
                "patch_coefficient": 0.0,
                "patch_effect": 0.0,
                "target_gap": 1.0,
                "source_cached_margin": 0.25,
                "correct_logit_margin": 0.25,
            })
        return rows

    def test_plan_is_model_derived_and_deterministic(self):
        first = build_scan_plan(self.protocol, self.model)
        second = build_scan_plan(self.protocol, self.model)
        self.assertEqual(first, second)
        self.assertEqual(first["residual_layers"], list(range(32)))
        self.assertNotIn(32, first["residual_layers"])
        self.assertFalse(first["final_test_open"])

    def test_smoke_uses_model_endpoints_and_all_hook_types(self):
        plan = build_scan_plan(self.protocol, self.model)
        layers, components = smoke_scan_axes(self.protocol, plan)
        self.assertEqual(layers, [0, 31])
        self.assertEqual(components, ["residual", "self_attn", "mlp"])
        axes = scan_factorial_axes(self.protocol, "smoke")
        self.assertEqual(axes["declared_roles"], ["collaborator"])
        self.assertEqual(axes["label_swaps"], [0])
        self.assertEqual(axes["source_regimes"], ["verification", "obedience"])

    def test_site_and_candidate_selection_are_deterministic(self):
        residual = self._rows((layer, "residual") for layer in range(32))
        nominated = (31, 30, 29, 28)
        component = self._rows(
            (layer, name) for layer in nominated for name in ("self_attn", "mlp")
        )
        first = select_native_sites(
            residual_rows=residual, component_rows=component, protocol=self.protocol
        )
        second = select_native_sites(
            residual_rows=list(reversed(residual)),
            component_rows=list(reversed(component)),
            protocol=self.protocol,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["nominated_layers"], [31, 30, 29, 28])
        grid = materialize_candidate_grid(first, self.protocol)
        self.assertEqual(grid["candidate_count"], 27)
        self.assertEqual(len({row["candidate_id"] for row in grid["candidates"]}), 27)
        self.assertFalse(grid["operator_dev_accessed"])
        self.assertFalse(grid["final_test_open"])

    def test_sha_binding_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{}\n")
            record = {"manifest": {"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}}
            require_sha_bound_path(record, "manifest", path)
            path.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, "SHA-bound"):
                require_sha_bound_path(record, "manifest", path)

    def test_dynamic_checkpoint_resolution_and_no_base_weights(self):
        site = {
            "layer": 11, "component": "self_attn", "boundary_rank": 8,
            "context_rank": 16, "positive_output_rank": 8,
            "negative_output_rank": 8, "protected_rank": 32,
            "max_relative_correction": 0.025, "rank_profile": "compact",
        }
        candidate = {"candidate_id": "CDGE42-test", "config": site}
        checkpoint = {
            "base_model_weights_included": False,
            "candidate_grid_sha256": "a" * 64,
            "sites": [{
                **{key: site[key] for key in (
                    "layer", "component", "boundary_rank", "context_rank",
                    "positive_output_rank", "negative_output_rank",
                )},
                "maximum_relative_correction": 0.025,
            }],
            "state_dict": {"editor.history_head.weight": "synthetic-editor-tensor"},
        }
        validate_checkpoint_against_candidate(
            checkpoint, candidate, candidate_grid_sha256="a" * 64
        )
        bad = copy.deepcopy(checkpoint)
        bad["sites"][0]["layer"] = 27
        with self.assertRaisesRegex(ValueError, "dynamic checkpoint/site"):
            validate_checkpoint_against_candidate(
                bad, candidate, candidate_grid_sha256="a" * 64
            )
        bad = copy.deepcopy(checkpoint)
        bad["state_dict"]["model.layers.0.weight"] = "synthetic-base-tensor"
        with self.assertRaisesRegex(ValueError, "base-weight"):
            validate_checkpoint_against_candidate(
                bad, candidate, candidate_grid_sha256="a" * 64
            )

    def test_matched_structural_routes_are_exact_zero_identity(self):
        try:
            import torch
            from scripts.benchmark_v4.directional_governance import structural_directional_routes
        except ImportError:
            self.skipTest("local Torch runtime is unavailable; this test runs in the cluster bundle")
        probabilities = torch.tensor([0.0, 1.0], dtype=torch.float32)
        routes = structural_directional_routes(probabilities, probabilities)
        self.assertTrue(torch.equal(routes["positive_route"], torch.zeros(2)))
        self.assertTrue(torch.equal(routes["negative_route"], torch.zeros(2)))
        original = torch.tensor([[1.25, -3.5]], dtype=torch.float32)
        correction = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
        edited = original + routes["positive_route"][0] * correction
        self.assertTrue(torch.equal(original, edited))

    def test_protocol_hash_changes_if_layer_is_injected(self):
        plan = build_scan_plan(self.protocol, self.model)
        changed = copy.deepcopy(plan)
        changed["residual_layers"] = [27]
        self.assertNotEqual(value_sha256(plan), value_sha256(changed))


if __name__ == "__main__":
    unittest.main()
