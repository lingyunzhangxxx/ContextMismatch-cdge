from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    import torch.optim.optimizer as optimizer_module

    from scripts.benchmark_v1.operators import LayerOperatorSpec, MultiLayerContextOperator
    from scripts.benchmark_v3.adaptive_governance import (
        AdaptiveGovernanceSiteEditor,
        AdaptiveMultiSiteGovernanceEditor,
        GovernanceSite,
        clip_relative_correction,
        editor_from_checkpoint,
        protected_residual_output_basis,
        signed_governance_factor,
    )
    from scripts.benchmark_v3.fit_governance_editor import _cpu_adamw

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for activation-editor tests")
class AdaptiveGovernanceEditorTests(unittest.TestCase):
    def _site(self, hidden: int = 8, maximum: float = 0.1):
        boundary_basis = torch.eye(hidden)[:, :2]
        context_basis = torch.eye(hidden)[:, 2:5]
        output_basis = torch.eye(hidden)[:, 5:7]
        editor = AdaptiveGovernanceSiteEditor(
            boundary_basis=boundary_basis,
            context_basis=context_basis,
            output_basis=output_basis,
            boundary_center=torch.zeros(hidden),
            context_center=torch.zeros(hidden),
            boundary_scale=torch.ones(2),
            context_scale=torch.ones(3),
            maximum_relative_correction=maximum,
            hidden_width=12,
        )
        with torch.no_grad():
            editor.history_head.weight.zero_()
            editor.history_head.bias.fill_(2.0)
            editor.task_head.weight.zero_()
            editor.task_head.bias.fill_(-2.0)
            editor.output_head.weight.zero_()
            editor.output_head.bias.fill_(4.0)
        return editor

    def test_zero_gate_returns_bitwise_identical_tensor(self):
        editor = self._site()
        value = torch.randn(2, 4, 8, dtype=torch.float32)
        result = editor(
            value,
            0.0,
            boundary_state=torch.randn(2, 8),
            context_state=torch.randn(2, 4, 8),
        )
        self.assertTrue(torch.equal(result, value))
        self.assertEqual(editor.last_diagnostics, {})

    def test_signed_factor_reverses_with_history_task_order(self):
        positive = signed_governance_factor(torch.tensor([2.0]), torch.tensor([-2.0]))
        negative = signed_governance_factor(torch.tensor([-2.0]), torch.tensor([2.0]))
        self.assertGreater(float(positive), 0.0)
        self.assertLess(float(negative), 0.0)
        self.assertAlmostEqual(float(positive), -float(negative), places=6)

    def test_full_state_heads_use_all_hidden_coordinates_within_budget(self):
        hidden = 4096
        identity = torch.eye(hidden)
        editor = AdaptiveGovernanceSiteEditor(
            boundary_basis=identity[:, :16],
            context_basis=identity[:, :32],
            output_basis=identity[:, :32],
            boundary_center=torch.zeros(hidden),
            context_center=torch.zeros(hidden),
            boundary_scale=torch.ones(16),
            context_scale=torch.ones(32),
            maximum_relative_correction=0.1,
            hidden_width=96,
            head_input_mode="full_state",
            boundary_head_scale=2.0,
            context_head_scale=3.0,
        )
        self.assertEqual(editor.history_head.in_features, hidden)
        self.assertEqual(editor.task_head.in_features, hidden)
        boundary = torch.zeros(2, hidden)
        context = torch.zeros(2, hidden)
        boundary[:, -1] = 4.0
        context[:, -2] = -6.0
        with torch.no_grad():
            editor.history_head.weight.zero_()
            editor.history_head.bias.zero_()
            editor.task_head.weight.zero_()
            editor.task_head.bias.zero_()
            editor.history_head.weight[0, -1] = 1.0
            editor.task_head.weight[0, -2] = 1.0
            history_logit, task_logit, boundary_code, context_code = (
                editor.governance_logits(boundary, context)
            )
        self.assertTrue(torch.equal(boundary_code, torch.zeros_like(boundary_code)))
        self.assertTrue(torch.equal(context_code, torch.zeros_like(context_code)))
        self.assertTrue(torch.allclose(history_logit, torch.full((2,), 2.0)))
        self.assertTrue(torch.allclose(task_logit, torch.full((2,), -2.0)))
        self.assertLess(editor.trainable_parameter_count * 3, 100000)

    def test_relative_trust_region_clips_every_row(self):
        reference = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        correction = torch.tensor([[30.0, 40.0], [10.0, 0.0]])
        clipped, _ = clip_relative_correction(correction, reference, 0.1)
        relative = torch.linalg.vector_norm(clipped, dim=-1) / torch.linalg.vector_norm(
            reference, dim=-1
        )
        self.assertTrue(torch.all(relative <= 0.100001))

    def test_output_basis_is_residualized_against_protected_space(self):
        torch.manual_seed(7)
        teacher = torch.randn(128, 12)
        protected = torch.randn(128, 12)
        protected[:, :2] += 4.0 * torch.randn(128, 2)
        basis = protected_residual_output_basis(teacher, protected, 3, 2)
        centered = protected - protected.mean(dim=0)
        _, _, protected_basis = torch.pca_lowrank(centered, q=4, niter=4)
        overlap = protected_basis[:, :2].transpose(0, 1) @ basis
        self.assertLess(float(overlap.abs().max()), 1e-4)

    def test_cpu_adamw_step_does_not_probe_npu_device(self):
        model = torch.nn.Linear(4, 2)
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    optimizer_module,
                    "_get_foreach_kernels_supported_devices",
                    side_effect=AssertionError(
                        "CPU optimizer step performed backend device discovery"
                    ),
                )
            )
            try:
                import torch_npu
            except (ImportError, OSError):
                torch_npu = None
            if torch_npu is not None:
                self.assertFalse(torch_npu.npu.is_initialized())
                stack.enter_context(
                    mock.patch.object(
                        torch_npu.npu,
                        "current_device",
                        side_effect=AssertionError(
                            "CPU optimizer step queried the current NPU device"
                        ),
                    )
                )
            optimizer = _cpu_adamw(
                model.parameters(), learning_rate=1e-3, weight_decay=1e-2
            )
            loss = model(torch.ones(3, 4)).square().mean()
            loss.backward()
            optimizer.step()
        self.assertIs(optimizer.defaults["foreach"], False)
        self.assertIs(optimizer.defaults["fused"], False)
        if torch_npu is not None:
            self.assertFalse(torch_npu.npu.is_initialized())

    def test_hooks_are_reversible_and_checkpoint_contains_no_base_weights(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.site = torch.nn.Linear(8, 8, bias=False)
                with torch.no_grad():
                    self.site.weight.copy_(torch.eye(8))

        site_editor = self._site()
        spec = GovernanceSite(23, "self_attn", 2)
        adaptive = AdaptiveMultiSiteGovernanceEditor({spec: site_editor})
        manager = adaptive.hook_manager()
        model = DummyModel()
        value = torch.randn(1, 3, 8)
        baseline = model.site(value)
        layer_spec = next(iter(manager.operators))
        manager.install(
            model,
            gate={layer_spec: 1.0},
            module_resolver=lambda model_value, _layer, _component: model_value.site,
            boundary_state={layer_spec: torch.randn(8)},
        )
        edited = model.site(value)
        manager.remove()
        restored = model.site(value)
        self.assertFalse(torch.equal(edited, baseline))
        self.assertTrue(torch.equal(restored, baseline))
        checkpoint = {
            "schema_version": 1,
            "kind": "adaptive_governance_editor",
            "trainable_parameter_count": adaptive.trainable_parameter_count,
            "base_model_weights_included": False,
            "sites": [
                {
                    "layer": 23,
                    "component": "self_attn",
                    "output_rank": 2,
                    "maximum_relative_correction": 0.1,
                    "hidden_width": 12,
                    "temperature": 1.0,
                    "constructor_tensors": {
                        "boundary_basis": site_editor.boundary_basis,
                        "context_basis": site_editor.context_basis,
                        "output_basis": site_editor.output_basis,
                        "boundary_center": site_editor.boundary_center,
                        "context_center": site_editor.context_center,
                        "boundary_scale": site_editor.boundary_scale,
                        "context_scale": site_editor.context_scale,
                    },
                    "state_dict": site_editor.state_dict(),
                }
            ],
        }
        rebuilt = editor_from_checkpoint(checkpoint)
        self.assertEqual(rebuilt.trainable_parameter_count, adaptive.trainable_parameter_count)
        self.assertFalse(checkpoint["base_model_weights_included"])
        self.assertEqual(
            set(checkpoint),
            {
                "schema_version",
                "kind",
                "trainable_parameter_count",
                "base_model_weights_included",
                "sites",
            },
        )
        self.assertFalse(
            any(isinstance(value, torch.nn.Module) for value in checkpoint.values())
        )
        self.assertEqual(
            set(checkpoint["sites"][0]["state_dict"]),
            set(site_editor.state_dict()),
        )
        self.assertFalse(
            any(
                key.startswith(("model.", "base_model."))
                for key in checkpoint["sites"][0]["state_dict"]
            )
        )


class GovernanceAuthorizationTests(unittest.TestCase):
    def _command(self, stage: str, output: Path) -> list[str]:
        design = (
            "governance_crossover_operator_dev_design_audit_v2.json"
            if stage.endswith("selection")
            else "governance_crossover_replication_design_audit_v2.json"
        )
        return [
            sys.executable,
            "-m",
            "scripts.benchmark_v3.materialize_governance_authorization",
            "--stage",
            stage,
            "--code-version",
            "22",
            "--bundle-manifest-sha256",
            "a" * 64,
            "--editor-contract",
            str(ROOT / "protocol/QWEN3_8B_ADAPTIVE_GOVERNANCE_EDITOR_V1.json"),
            "--crossover-contract",
            str(ROOT / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json"),
            "--operator-site-manifest",
            str(ROOT / "artifacts/qwen3-8b-v1/QWEN3_8B_OPERATOR_SITE_MANIFEST_V1.json"),
            "--manifest",
            str(ROOT / "artifacts/qwen3-8b-v1/benchmark_manifest.jsonl"),
            "--design-audit",
            str(ROOT / f"artifacts/qwen3-8b-v1/{design}"),
            "--model-manifest-remote",
            "/workspace/context-mismatch-qwen3-8b/runs/model-verification.json",
            "--model-manifest-sha256",
            "b" * 64,
            "--created-utc",
            "2026-07-28T03:30:00Z",
            "--output",
            str(output),
        ]

    def test_capture_smoke_authorization_is_sha_bound_and_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "authorization.json"
            subprocess.run(
                self._command("governance_capture_smoke", output),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            value = json.loads(output.read_text())
            self.assertEqual(value["expected_rows"], 192)
            self.assertEqual(len(value["expected_key_sha256"]), 64)
            self.assertFalse(value["final_test_open"])
            self.assertEqual(value["final_test_open_count"], 0)
            self.assertFalse(value["production_rollout_approved"])

    def test_code_v21_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "authorization.json"
            command = self._command("governance_capture_smoke", output)
            command[command.index("--code-version") + 1] = "21"
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("code-v22 or newer", result.stderr)
            self.assertFalse(output.exists())

    def test_v2_was_frozen_before_v1_post_failure_characterization(self):
        v2_path = ROOT / "protocol/QWEN3_8B_ADAPTIVE_GOVERNANCE_EDITOR_V2.json"
        diagnostic_path = (
            ROOT / "protocol/QWEN3_8B_AMSGE_V1_POST_FAILURE_CHARACTERIZATION.json"
        )
        v2 = json.loads(v2_path.read_text())
        diagnostic = json.loads(diagnostic_path.read_text())
        self.assertEqual(
            v2["status"], "frozen_before_any_v1_post_failure_operator_dev_forward"
        )
        self.assertEqual(
            diagnostic["pre_registration_order"]["v2_contract_sha256"],
            sha256(v2_path),
        )
        self.assertFalse(
            diagnostic["pre_registration_order"][
                "v1_operator_dev_results_available_at_freeze"
            ]
        )
        self.assertFalse(diagnostic["candidate_eligible"])
        self.assertFalse(diagnostic["candidate_may_be_locked"])


if __name__ == "__main__":
    unittest.main()
