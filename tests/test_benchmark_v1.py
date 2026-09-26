from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ImportError:  # The lightweight local audit environment need not ship PyTorch.
    torch = None
    nn = None

from scripts.benchmark_v1.adapters import adapt_arc, adapt_bbh, adapt_gsm8k, adapt_math500
from scripts.benchmark_v1.common import (
    PARTITIONS,
    assign_partitions,
    effective_mechanism_items_per_benchmark,
    expected_full_mechanism_rows,
)
from scripts.benchmark_v1.controls import (
    factual_memory_messages,
    factual_memory_prompt,
    supported_authority_prompt,
    verification_requirement,
)
from scripts.benchmark_v1.histories import (
    pairwise_no_pressure_prompt,
    pairwise_prompt,
    prefix_messages,
)
from scripts.benchmark_v1.lock_operator_sites import _sign_diagnostic, _site_score
if torch is not None:
    from scripts.benchmark_v1.operators import (
        ConditionalLowRankTransport,
        ContextConditionedLowRankTransport,
        FixedDirectionTranslation,
        LayerOperatorSpec,
        MultiLayerContextOperator,
        OneSidedProtectedProjection,
        SignedGovernanceAlignmentGate,
        bilinear_transport_features,
        fit_ridge_transport,
        orthonormalize,
        remove_protected_subspace,
    )
    from scripts.benchmark_v1.run_subspace_capture import _module_hidden_input


class AdapterTests(unittest.TestCase):
    def test_arc_deterministic_foil(self):
        row = {
            "id": "x",
            "question": "Which?",
            "choices": {"label": ["A", "B", "C"], "text": ["one", "two", "three"]},
            "answerKey": "B",
        }
        item = adapt_arc(row)
        self.assertEqual(item["correct_answer"], "two")
        self.assertNotEqual(item["foil_answer"], "two")

    def test_bbh_parser(self):
        row = {"input": "Question text\nOptions:\n(A) alpha\n(B) beta", "target": "(A)"}
        item = adapt_bbh(row, "demo", 0)
        self.assertEqual(item["question"], "Question text")
        self.assertEqual(item["correct_answer"], "alpha")
        self.assertEqual(item["foil_answer"], "beta")

    def test_bbh_boolean_adapter(self):
        item = adapt_bbh({"input": "Does A tell the truth?", "target": "No"}, "web_of_lies", 2)
        self.assertEqual(item["correct_answer"], "No")
        self.assertEqual(item["foil_answer"], "Yes")

    def test_numeric_mutations_change_answer(self):
        gsm = adapt_gsm8k({"question": "q", "answer": "work\n#### 18"}, 0)
        math = adapt_math500({"unique_id": "u", "problem": "p", "answer": r"\frac{3}{4}", "subject": "x", "level": 1})
        self.assertNotEqual(gsm["correct_answer"], gsm["foil_answer"])
        self.assertNotEqual(math["correct_answer"], math["foil_answer"])


class DesignTests(unittest.TestCase):
    def test_partitions_are_exact(self):
        rows = [{"native_id": str(i), "item_id": f"b:{i}"} for i in range(128)]
        assigned = assign_partitions(rows, 7, "b")
        counts = {partition: sum(row["partition"] == partition for row in assigned) for partition in PARTITIONS}
        self.assertEqual(set(counts.values()), {32})
        self.assertTrue(all(0 <= row["history_realization"] <= 2 for row in assigned))

    def test_mechanism_erratum_resolves_row_arithmetic_without_expanding_items(self):
        contract = {
            "data_split": {"benchmarks": list("abcdef"), "items_per_benchmark": 32},
            "factorial": {
                "declared_roles": ["collaborator", "assistant"],
                "history_styles": ["lexical_matched", "natural"],
                "label_swaps": [0, 1],
                "source_regimes": ["verification", "obedience"],
            },
            "residual_scan": {
                "layers": [3, 7, 11, 15, 19, 23, 27, 31, 35],
                "coefficients": [0.0, 1.0],
                "expected_full_rows": 27648,
                "component_nomination_rule": {"select_top_k": 3},
            },
            "component_scan": {
                "components": ["self_attn", "mlp"],
                "coefficients": [0.0, 1.0],
                "expected_full_rows": 18432,
            },
        }
        erratum = {
            "production_rollout_approved": False,
            "mechanism_contract_sha256": "contract-sha",
            "affected_field": "data_split.items_per_benchmark",
            "audit_finding": {
                "detected_before_affected_forward": True,
                "failed_job_scientific_rows": 0,
            },
            "replacement": {
                "items_per_benchmark": 16,
                "residual_expected_full_rows": 27648,
                "component_expected_full_rows": 18432,
            },
        }
        self.assertEqual(
            effective_mechanism_items_per_benchmark(contract, erratum, "contract-sha"),
            16,
        )
        self.assertEqual(expected_full_mechanism_rows(contract, "residual", 16), 27648)
        self.assertEqual(expected_full_mechanism_rows(contract, "component", 16), 18432)
        self.assertEqual(expected_full_mechanism_rows(contract, "residual", 32), 55296)
        with self.assertRaises(ValueError):
            effective_mechanism_items_per_benchmark(contract, erratum, "wrong-sha")

    def test_label_swap_preserves_prompt_pressure_pair(self):
        item = {"question": "q", "correct_answer": "right", "foil_answer": "wrong"}
        prompt0, correct0, foil0 = pairwise_prompt(item, 0)
        prompt1, correct1, foil1 = pairwise_prompt(item, 1)
        self.assertEqual((correct0, foil0), ("A", "B"))
        self.assertEqual((correct1, foil1), ("B", "A"))
        self.assertIn("Candidate A: right", prompt0)
        self.assertIn("Candidate A: wrong", prompt1)

    def test_pressure_free_control_removes_user_preference(self):
        item = {"question": "q", "correct_answer": "right", "foil_answer": "wrong"}
        prompt, correct, foil = pairwise_no_pressure_prompt(item, 0)
        self.assertEqual((correct, foil), ("A", "B"))
        self.assertIn("There is no user preference", prompt)
        self.assertNotIn("I prefer candidate", prompt)

    def test_role_and_governance_are_separate(self):
        collaborator = prefix_messages("collaborator", "verification", "natural", 1, 0)
        assistant = prefix_messages("assistant", "verification", "natural", 1, 0)
        self.assertNotEqual(collaborator[0], assistant[0])
        self.assertEqual(collaborator[1:], assistant[1:])

    def test_supported_authority_and_memory_controls_are_label_balanced(self):
        authority = {
            "rule": "Use the owner's choice.",
            "target": "east",
            "foil": "west",
            "evidence": ["Both work.", "The owner chose east."],
        }
        memory = {"fact": "QUARTZ-17", "foil": "QUARTZ-71"}
        prompt0, correct0, foil0 = supported_authority_prompt(authority, 0)
        prompt1, correct1, foil1 = supported_authority_prompt(authority, 1)
        self.assertEqual((correct0, foil0), ("A", "B"))
        self.assertEqual((correct1, foil1), ("B", "A"))
        self.assertIn("Candidate A: east", prompt0)
        self.assertIn("Candidate A: west", prompt1)
        memory_prompt, memory_correct, _ = factual_memory_prompt(memory, 1)
        self.assertEqual(memory_correct, "B")
        self.assertIn("QUARTZ-17", memory_prompt)
        self.assertIn("QUARTZ-17", factual_memory_messages(memory)[0]["content"])
        self.assertEqual(verification_requirement("supported_user_authority"), 0.0)
        self.assertEqual(verification_requirement("independent_verification"), 1.0)

    def test_operator_site_score_uses_all_benchmark_direction_cells(self):
        rows = []
        for benchmark in ("a", "b", "c", "d", "e", "f"):
            for source_regime, effect, gap in (
                ("obedience", 0.5, 1.0),
                ("verification", -0.25, -1.0),
            ):
                for label_swap in (0, 1):
                    rows.append(
                        {
                            "layer": 7,
                            "component": "self_attn",
                            "history_style": "lexical_matched",
                            "patch_coefficient": 1.0,
                            "benchmark": benchmark,
                            "source_regime": source_regime,
                            "label_swap": label_swap,
                            "patch_effect": effect,
                            "target_gap": gap,
                        }
                    )
        score, cells = _site_score(rows, 7, "self_attn")
        self.assertEqual(len(cells), 12)
        self.assertAlmostEqual(score, 0.375)
        self.assertTrue(
            all(value["expected_direction"] for value in _sign_diagnostic(rows, 7, "self_attn"))
        )


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local audit environment")
class OperatorTests(unittest.TestCase):
    def test_component_capture_resolves_positional_and_keyword_hidden_states(self):
        value = torch.tensor([[[1.0, 2.0]]])
        self.assertIs(_module_hidden_input((value,), {}), value)
        self.assertIs(_module_hidden_input((), {"hidden_states": value}), value)
        with self.assertRaises(RuntimeError):
            _module_hidden_input((), {})

    def test_orthonormalize_preserves_one_sided_column_orientation(self):
        basis = torch.tensor([[1.0, 0.0], [0.0, -2.0], [0.0, 0.0]])
        oriented = orthonormalize(basis)
        self.assertGreater(float(oriented[:, 0] @ basis[:, 0]), 0.0)
        self.assertGreater(float(oriented[:, 1] @ basis[:, 1]), 0.0)

    def test_harmful_basis_is_residualized_against_protected_basis(self):
        harmful = torch.tensor([[1.0], [1.0], [0.0]])
        protected = torch.tensor([[1.0], [0.0], [0.0]])
        basis = remove_protected_subspace(harmful, protected)
        self.assertTrue(torch.allclose(protected.T @ basis, torch.zeros(1, 1), atol=1e-6))

    def test_zero_gate_is_exact_identity_and_negative_side_is_preserved(self):
        operator = OneSidedProtectedProjection(
            harmful_basis=torch.tensor([[1.0], [0.0]]),
            threshold=torch.tensor([0.0]),
            strength=torch.tensor([1.0]),
        )
        value = torch.tensor([[2.0, 3.0], [-2.0, 3.0]])
        self.assertTrue(torch.equal(operator(value, 0.0), value))
        edited = operator(value, 1.0)
        self.assertTrue(torch.allclose(edited[0], torch.tensor([0.0, 3.0])))
        self.assertTrue(torch.equal(edited[1], value[1]))

    def test_hooks_are_reversible(self):
        model = nn.Sequential(nn.Identity())
        spec = LayerOperatorSpec(layer=0, component="identity", rank=1)
        operator = OneSidedProtectedProjection(
            harmful_basis=torch.tensor([[1.0], [0.0]]),
            threshold=torch.tensor([0.0]),
            strength=torch.tensor([1.0]),
        )
        hooks = MultiLayerContextOperator({spec: operator})
        resolver = lambda current_model, _layer, _component: current_model[0]
        value = torch.tensor([[2.0, 3.0]])
        hooks.install(model, 1.0, resolver)
        self.assertTrue(torch.allclose(model(value), torch.tensor([[0.0, 3.0]])))
        hooks.remove()
        self.assertTrue(torch.equal(model(value), value))

    def test_operator_hooks_accept_keyword_only_hidden_states(self):
        class KeywordIdentity(nn.Module):
            def forward(self, *, hidden_states):
                return hidden_states

        model = KeywordIdentity()
        spec = LayerOperatorSpec(layer=0, component="identity", rank=1)
        operator = FixedDirectionTranslation(torch.tensor([1.0, 0.0]), strength=1.0)
        hooks = MultiLayerContextOperator({spec: operator})
        hooks.install(model, 1.0, lambda current, _layer, _component: current)
        value = torch.tensor([[2.0, 3.0]])
        self.assertTrue(
            torch.equal(model(hidden_states=value), torch.tensor([[1.0, 3.0]]))
        )
        hooks.remove()
        self.assertTrue(torch.equal(model(hidden_states=value), value))

    def test_low_rank_transport_separates_trigger_and_output_and_respects_budget(self):
        operator = ConditionalLowRankTransport(
            trigger_basis=torch.tensor([[1.0], [0.0]]),
            output_basis=torch.tensor([[0.0], [1.0]]),
            transport=torch.tensor([[10.0]]),
            threshold=torch.tensor([0.0]),
            max_relative_correction=0.5,
        )
        value = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
        edited = operator(value, 1.0)
        self.assertTrue(torch.allclose(edited[0], torch.tensor([2.0, -1.0]), atol=1e-6))
        self.assertTrue(torch.equal(edited[1], value[1]))
        self.assertTrue(torch.equal(operator(value, 0.0), value))

    def test_fixed_translation_is_exact_at_zero_gate(self):
        operator = FixedDirectionTranslation(torch.tensor([1.0, 0.0]), strength=2.0)
        value = torch.tensor([[3.0, 4.0]])
        self.assertTrue(torch.equal(operator(value, 0.0), value))
        self.assertTrue(torch.equal(operator(value, 1.0), torch.tensor([[1.0, 4.0]])))

    def test_legacy_operators_obey_relative_trust_budget(self):
        value = torch.tensor([[3.0, 4.0]])
        translation = FixedDirectionTranslation(
            torch.tensor([1.0, 0.0]),
            strength=2.0,
            max_relative_correction=0.1,
        )
        translated = translation(value, 1.0)
        self.assertLessEqual(
            float(torch.linalg.vector_norm(translated - value)),
            0.5 + 1e-6,
        )
        projection = OneSidedProtectedProjection(
            harmful_basis=torch.tensor([[1.0], [0.0]]),
            threshold=torch.tensor([0.0]),
            strength=torch.tensor([1.0]),
            one_sided=False,
            max_relative_correction=0.1,
        )
        projected = projection(value, 1.0)
        self.assertLessEqual(
            float(torch.linalg.vector_norm(projected - value)),
            0.5 + 1e-6,
        )

    def test_v4_uses_task_context_to_reverse_the_correction_without_answer_labels(self):
        operator = ContextConditionedLowRankTransport(
            history_trigger_basis=torch.tensor([[1.0], [0.0]]),
            context_basis=torch.tensor([[0.0], [1.0]]),
            output_basis=torch.tensor([[1.0], [0.0]]),
            # Features are [history, history*context]. The interaction term
            # makes the correction change sign with the task-state coordinate.
            transport=torch.tensor([[0.0], [2.0]]),
            threshold=torch.tensor([0.0]),
            context_scale=torch.tensor([1.0]),
            edit_last_token_only=False,
        )
        boundary = torch.tensor([[1.0, 0.0]])
        values = torch.tensor([[[0.0, 2.0], [0.0, -2.0]]])
        edited = operator(values, 1.0, boundary_state=boundary)
        self.assertLess(edited[0, 0, 0], 0.0)
        self.assertGreater(edited[0, 1, 0], 0.0)

    def test_v4_last_token_mask_and_zero_gate_are_exact(self):
        operator = ContextConditionedLowRankTransport(
            history_trigger_basis=torch.tensor([[1.0], [0.0]]),
            output_basis=torch.tensor([[0.0], [1.0]]),
            transport=torch.tensor([[1.0]]),
            threshold=torch.tensor([0.0]),
            edit_last_token_only=True,
        )
        boundary = torch.tensor([2.0, 0.0])
        value = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
        edited = operator(value, 1.0, boundary_state=boundary)
        self.assertTrue(torch.equal(edited[:, 0], value[:, 0]))
        self.assertTrue(torch.allclose(edited[:, 1], torch.tensor([[1.0, -1.0]])))
        self.assertTrue(torch.equal(operator(value, 0.0, boundary_state=boundary), value))

    def test_v5_reference_trigger_allows_reverse_edit_from_verification_state(self):
        operator = ContextConditionedLowRankTransport(
            history_trigger_basis=torch.tensor([[1.0], [0.0]]),
            output_basis=torch.tensor([[0.0], [1.0]]),
            transport=torch.tensor([[2.0]]),
            threshold=torch.tensor([0.5]),
            history_reference=torch.tensor([1.0]),
            edit_last_token_only=False,
        )
        verification_boundary = torch.tensor([[-2.0, 0.0]])
        value = torch.tensor([[1.0, 1.0]])
        edited = operator(value, -1.0, boundary_state=verification_boundary)
        self.assertGreater(float(edited[0, 1]), float(value[0, 1]))

    def test_signed_gate_repairs_both_mismatch_directions_and_closes_for_memory(self):
        gate = SignedGovernanceAlignmentGate(
            history_weight=torch.tensor([4.0, 0.0]),
            history_bias=0.0,
            deadzone=0.0,
        )
        obedience_state = torch.tensor([[1.0, 0.0]])
        verification_state = torch.tensor([[-1.0, 0.0]])
        too_obedient = gate(obedience_state, target_obedience=0.0)
        too_verifying = gate(verification_state, target_obedience=1.0)
        memory_closed = gate(obedience_state, target_obedience=0.0, applicable=0.0)
        self.assertGreater(float(too_obedient), 0.0)
        self.assertLess(float(too_verifying), 0.0)
        self.assertEqual(float(memory_closed), 0.0)

    def test_signed_gate_deadzone_suppresses_small_alignment_error(self):
        gate = SignedGovernanceAlignmentGate(
            history_weight=torch.tensor([0.0, 0.0]),
            history_bias=0.0,
            deadzone=0.2,
        )
        state = torch.tensor([[0.0, 0.0]])
        self.assertEqual(float(gate(state, target_obedience=0.5)), 0.0)

    def test_multilayer_hooks_accept_layer_specific_boundary_states(self):
        model = nn.Sequential(nn.Identity(), nn.Identity())
        spec0 = LayerOperatorSpec(layer=0, component="identity", rank=1)
        spec1 = LayerOperatorSpec(layer=1, component="identity", rank=1)
        make_operator = lambda: ContextConditionedLowRankTransport(
            history_trigger_basis=torch.tensor([[1.0], [0.0]]),
            output_basis=torch.tensor([[0.0], [1.0]]),
            transport=torch.tensor([[1.0]]),
            threshold=torch.tensor([0.0]),
            edit_last_token_only=False,
        )
        hooks = MultiLayerContextOperator({spec0: make_operator(), spec1: make_operator()})
        resolver = lambda current_model, layer, _component: current_model[layer]
        hooks.install(
            model,
            1.0,
            resolver,
            boundary_state={
                spec0: torch.tensor([1.0, 0.0]),
                spec1: torch.tensor([2.0, 0.0]),
            },
        )
        value = torch.tensor([[1.0, 4.0]])
        self.assertTrue(torch.allclose(model(value), torch.tensor([[1.0, 1.0]])))
        hooks.remove()

    def test_multilayer_hooks_accept_signed_site_specific_gates(self):
        model = nn.Sequential(nn.Identity(), nn.Identity())
        spec0 = LayerOperatorSpec(layer=0, component="identity", rank=1)
        spec1 = LayerOperatorSpec(layer=1, component="identity", rank=1)
        # Use an activation-independent correction so this test isolates signed
        # gate routing.  Sequential activation-dependent projections do not
        # generally cancel: the first site changes the feature seen by the
        # second site.
        make_operator = lambda: FixedDirectionTranslation(
            direction=torch.tensor([1.0, 0.0]),
            strength=1.0,
        )
        hooks = MultiLayerContextOperator({spec0: make_operator(), spec1: make_operator()})
        resolver = lambda current_model, layer, _component: current_model[layer]
        value = torch.tensor([[1.0, 2.0]])
        hooks.install(
            model,
            gate={spec0: 1.0, spec1: -1.0},
            module_resolver=resolver,
            collect_diagnostics=True,
        )
        self.assertTrue(torch.equal(model(value), value))
        self.assertEqual(set(hooks.last_stats), {spec0, spec1})
        hooks.remove()

    def test_bilinear_feature_order_and_ridge_fit(self):
        history = torch.tensor(
            [[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [3.0, -1.0], [-2.0, 3.0]]
        )
        context = torch.tensor([[3.0], [-1.0], [2.0], [0.5], [-2.0]])
        features = bilinear_transport_features(
            history,
            threshold=torch.zeros(2),
            context_scores=context,
            one_sided=False,
        )
        self.assertTrue(
            torch.equal(
                features[0],
                torch.tensor([1.0, 2.0, 3.0, 6.0]),
            )
        )
        true_transport = torch.tensor([[1.0], [2.0], [-1.0], [0.5]])
        targets = features @ true_transport
        fitted = fit_ridge_transport(features, targets, penalty=1e-6)
        self.assertTrue(torch.allclose(features @ fitted, targets, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
