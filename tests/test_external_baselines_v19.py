import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from scripts.benchmark_v19.external_baselines import (
        CAAAdapter, CASTAdapter, LoReFTAdapter, RePSAdapter, finite_checkpoint,
    )
    from scripts.benchmark_v19.fit_external_baselines import _fit_reps


@unittest.skipIf(torch is None, "Torch is required")
class ExternalBaselineTests(unittest.TestCase):
    def test_caa_is_task_signed_and_zero_gate_is_identity(self):
        x = torch.ones(2, 4)
        editor = CAAAdapter(torch.tensor([1.0, 0, 0, 0]), 0.5, 0.5).to("cpu")
        editor.set_task(1.0)
        self.assertTrue(torch.equal(editor(x, 0.0), x))
        positive = editor(x, 1.0)
        editor.set_task(0.0)
        negative = editor(x, 1.0)
        self.assertGreater(float(positive[0, 0]), 1.0)
        self.assertLess(float(negative[0, 0]), 1.0)

    def test_cast_abstains_on_matched_history(self):
        editor = CASTAdapter(
            torch.tensor([1.0, 0.0]), 0.2,
            torch.tensor([1.0, 0.0]), 0.0, 0.5,
        ).to("cpu")
        value = torch.ones(1, 2)
        editor.set_task(1.0)
        matched = editor(value, 1.0, boundary_state=torch.tensor([1.0, 0.0]))
        mismatch = editor(value, 1.0, boundary_state=torch.tensor([-1.0, 0.0]))
        self.assertTrue(torch.equal(matched, value))
        self.assertFalse(torch.equal(mismatch, value))

    def test_loreft_formula_and_reps_are_bounded(self):
        basis = torch.tensor([[1.0], [0.0]])
        weights = torch.zeros(2, 1, 2)
        bias = torch.tensor([[1.0], [-1.0]])
        editor = LoReFTAdapter(basis, weights, bias, 0.1).to("cpu")
        value = torch.tensor([[2.0, 0.0]])
        editor.set_task(1.0)
        changed = editor(value, 1.0)
        self.assertLessEqual(float(torch.linalg.vector_norm(changed - value)), 0.200001)
        reps = RePSAdapter(
            torch.tensor([[10.0, 0.0], [-10.0, 0.0]]), 0.1,
        ).to("cpu")
        reps.set_task(0.0)
        changed = reps(value, 1.0)
        self.assertLessEqual(float(torch.linalg.vector_norm(changed - value)), 0.200001)

    def test_finite_checkpoint(self):
        self.assertTrue(finite_checkpoint({"x": torch.ones(2), "y": 1.0}))
        self.assertFalse(finite_checkpoint({"x": torch.tensor([float("nan")])}))
        vectors, report = _fit_reps(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0.1, 0.2]), torch.tensor([0.0, 1.0]),
            torch.tensor([0, 1]), beta=0.1, steps=2, learning_rate=0.01,
        )
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertEqual(report["0"]["optimizer"], "deterministic_cpu_adamw")


if __name__ == "__main__":
    unittest.main()
