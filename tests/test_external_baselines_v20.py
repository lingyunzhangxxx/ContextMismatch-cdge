import hashlib
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from scripts.benchmark_v20.fit_official_baselines import _cast_scores, _pca_pairwise, _select_threshold
    from scripts.benchmark_v20.official_adapters import OfficialCASTAdapter, finite_checkpoint
    from scripts.benchmark_v20.official_source import load_loreft_class, load_preference_loss, load_reps_class


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "third_party" / "upstream"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipIf(torch is None, "Torch is required")
class OfficialDerivedBaselineTests(unittest.TestCase):
    def test_official_loreft_class_is_executed(self):
        path = SOURCE_ROOT / "pyreft/pyreft/interventions.py"
        cls = load_loreft_class(SOURCE_ROOT, _sha(path))
        module = cls(embed_dim=4, low_rank_dimension=2, dtype=torch.float32, dropout=0.0, act_fn=None)
        value = torch.randn(3, 4)
        output = module(value)
        self.assertEqual(output.shape, value.shape)
        gram = module.rotate_layer.weight.T @ module.rotate_layer.weight
        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-5, rtol=1e-5))

    def test_official_reps_class_and_loss_are_executed(self):
        intervention_path = SOURCE_ROOT / "axbench/axbench/models/interventions.py"
        loss_path = SOURCE_ROOT / "axbench/axbench/models/preference_model.py"
        cls = load_reps_class(SOURCE_ROOT, _sha(intervention_path))
        loss_fn = load_preference_loss(SOURCE_ROOT, _sha(loss_path))
        modules = [
            cls(embed_dim=4, low_rank_dimension=1, dropout=0.0, intervention_positions_dropout=0.0)
            for _ in range(2)
        ]
        with torch.no_grad():
            modules[0].proj.weight.copy_(torch.tensor([[1.0, 0, 0, 0]]))
            modules[1].proj.weight.copy_(torch.tensor([[0.0, 1, 0, 0]]))
            for module in modules:
                module.proj.bias.zero_()
                module.proj.bias.requires_grad_(False)
        self.assertEqual(
            sum(parameter.numel() for module in modules for parameter in module.parameters() if parameter.requires_grad),
            2 * 4,
        )
        value = torch.zeros(1, 1, 4)
        first = modules[0](value, subspaces={"subspaces": [0], "steering_factor": torch.ones(1)})
        second = modules[1](value, subspaces={"subspaces": [0], "steering_factor": torch.ones(1)})
        self.assertEqual(float(first.output[0, 0, 0]), 1.0)
        self.assertEqual(float(second.output[0, 0, 1]), 1.0)
        chosen = torch.tensor([1.0, -1.0])
        rejected = -chosen
        zeros = torch.zeros(2)
        ones = torch.ones(2, dtype=torch.long)
        losses, _, _ = loss_fn(chosen, rejected, zeros, zeros, 0.1, 0.0, 1.0, ones, ones, reference_free=True)
        self.assertTrue(torch.isfinite(losses).all())
        self.assertLess(float(losses[0]), float(losses[1]))

    def test_cast_uses_pairwise_pca_and_projector_cosine(self):
        positive = torch.tensor([[2.0, 1.0], [3.0, -1.0], [4.0, 0.5]])
        negative = torch.tensor([[-2.0, 1.0], [-3.0, -1.0], [-4.0, 0.5]])
        direction, variance = _pca_pairwise(positive, negative)
        self.assertGreater(float(direction[0]), 0.99)
        self.assertGreater(variance, 0.99)
        boundary = torch.tensor([[2.0, 0.1], [1.0, 3.0], [-2.0, 0.1], [-1.0, 3.0]])
        scores = _cast_scores(boundary, direction)
        threshold, polarity, accuracy = _select_threshold(scores, torch.tensor([1.0, 1.0, 0.0, 0.0]))
        self.assertIn(polarity, (-1, 1))
        self.assertGreaterEqual(accuracy, 0.5)
        editor = OfficialCASTAdapter(direction, 0.2, direction, threshold, polarity, 0.5)
        value = torch.ones(1, 2)
        editor.set_task(1.0)
        self.assertTrue(torch.equal(editor(value, 0.0, boundary_state=boundary[0]), value))

    def test_nested_checkpoint_finiteness(self):
        self.assertTrue(finite_checkpoint({"state": [{"x": torch.ones(2)}]}))
        self.assertFalse(finite_checkpoint({"state": [{"x": torch.tensor([float("nan")])}]}))


if __name__ == "__main__":
    unittest.main()
