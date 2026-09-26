#!/usr/bin/env python3
"""Build paper-facing summaries and vector figures from frozen evidence.

This script does not fit or select a method. It reads immutable/derived evidence,
checks the expected row counts, and emits a compact claims JSON plus PDF figures.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import reportlab


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
GENERATED = ROOT / "generated"
FIGURES = GENERATED / "figures"
FIGURE_FONT = "FigureSans"
FIGURE_FONT_BOLD = "FigureSansBold"
pdfmetrics.registerFont(
    TTFont(FIGURE_FONT, str(Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"))
)
pdfmetrics.registerFont(
    TTFont(FIGURE_FONT_BOLD, str(Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf"))
)


def load_json(name: str):
    with (DATA / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(name: str):
    with (DATA / name).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float) -> float:
    return 100.0 * value


BLUE = colors.HexColor("#3B6FB6")
ORANGE = colors.HexColor("#D77A2B")
GREEN = colors.HexColor("#2A9D8F")
GRAY = colors.HexColor("#5C6670")
LIGHT_GRAY = colors.HexColor("#D8DADD")


def new_canvas(name: str, width_in: float, height_in: float) -> tuple[canvas.Canvas, float, float]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    width, height = width_in * 72, height_in * 72
    return canvas.Canvas(
        str(FIGURES / name), pagesize=(width, height), initialFontName=FIGURE_FONT
    ), width, height


def finish(c: canvas.Canvas) -> None:
    c.showPage()
    c.save()


def draw_multiline(c: canvas.Canvas, x: float, y: float, text: str, size: float = 7.2,
                   color=colors.black, align: str = "center", leading: float | None = None) -> None:
    leading = leading or size * 1.16
    lines = text.split("\n")
    c.setFillColor(color)
    c.setFont(FIGURE_FONT, size)
    for index, line in enumerate(lines):
        yy = y - index * leading
        if align == "center":
            c.drawCentredString(x, yy, line)
        elif align == "right":
            c.drawRightString(x, yy, line)
        else:
            c.drawString(x, yy, line)


def draw_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, text: str,
             stroke, fill) -> None:
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.setLineWidth(1.0)
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    lines = text.split("\n")
    draw_multiline(c, x + w / 2, y + h / 2 + (len(lines) - 1) * 4.0,
                   text, size=6.5, color=stroke)


def draw_arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=GRAY) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.0)
    c.line(x1, y1, x2, y2)
    angle = __import__("math").atan2(y2 - y1, x2 - x1)
    size = 4.0
    for delta in (2.55, -2.55):
        c.line(x2, y2, x2 + size * __import__("math").cos(angle + delta),
               y2 + size * __import__("math").sin(angle + delta))


def build_overview() -> None:
    c, width, height = new_canvas("overview.pdf", 7.0, 3.05)
    c.setFont(FIGURE_FONT_BOLD, 10.2)
    c.setFillColor(colors.HexColor("#263238"))
    c.drawCentredString(width / 2, height - 14,
                        "Retain the evidence; expire the obsolete policy")

    # Panel (a): paired identification as a two-by-two crossover.
    x0, y0, cw, ch = 57, 73, 53, 38
    draw_multiline(c, 82, 190, "(a) Failure: policy outlives task", size=8.2, color=colors.black)
    draw_multiline(c, 30, 165, "Prior\nhistory", size=6.8, color=GRAY)
    draw_multiline(c, x0 + cw / 2, 165, "Independent\nverification", size=6.6, color=BLUE)
    draw_multiline(c, x0 + 1.5 * cw, 165, "Delegated\nchoice", size=6.6, color=ORANGE)
    draw_multiline(c, 49, y0 + 1.5 * ch + 5, "Verify", size=6.6, color=BLUE, align="right")
    draw_multiline(c, 49, y0 + 0.5 * ch + 5, "Obey", size=6.6, color=ORANGE, align="right")
    cells = [
        (x0, y0 + ch, "MATCH", GREEN, colors.HexColor("#E8F5F1")),
        (x0 + cw, y0 + ch, "MISMATCH", colors.HexColor("#C84C4C"), colors.HexColor("#FCEBEC")),
        (x0, y0, "MISMATCH", colors.HexColor("#C84C4C"), colors.HexColor("#FCEBEC")),
        (x0 + cw, y0, "MATCH", GREEN, colors.HexColor("#E8F5F1")),
    ]
    for x, y, label, stroke, fill in cells:
        c.setStrokeColor(stroke); c.setFillColor(fill); c.setLineWidth(0.9)
        c.roundRect(x, y, cw - 3, ch - 3, 4, stroke=1, fill=1)
        draw_multiline(c, x + (cw - 3) / 2, y + 13, label, size=6.2, color=stroke)
    draw_multiline(c, 82, 62, "Same evidence, candidates, outcomes, and A/B swaps",
                   size=5.9, color=GRAY)

    # Panel (b): the mechanism-level insight.
    draw_multiline(c, 255, 190, "(b) Insight: stale policy consumes margin", size=8.2, color=colors.black)
    draw_box(c, 183, 129, 74, 34, "completed interaction\nsets policy state z_H", BLUE,
             colors.HexColor("#EEF4FB"))
    draw_box(c, 276, 129, 66, 34, "state persists\npast boundary", ORANGE,
             colors.HexColor("#FFF2E8"))
    draw_box(c, 276, 77, 66, 34, "task-conditioned\nreadout r_T", GRAY,
             colors.HexColor("#F3F4F5"))
    draw_box(c, 183, 77, 74, 34, "current evidence\nremains available", GREEN,
             colors.HexColor("#EAF7F4"))
    draw_arrow(c, 258, 146, 275, 146)
    draw_arrow(c, 309, 128, 309, 112)
    draw_arrow(c, 258, 94, 275, 94)
    c.setStrokeColor(colors.HexColor("#C84C4C")); c.setFillColor(colors.HexColor("#C84C4C"))
    c.setLineWidth(1.1); c.line(309, 76, 309, 61)
    draw_multiline(c, 263, 51, "stale shift / original margin >= 1", size=5.8,
                   color=colors.HexColor("#C84C4C"))
    draw_multiline(c, 263, 39, "decision flips near the boundary", size=6.1,
                   color=colors.HexColor("#C84C4C"))

    # Panel (c): mechanism-derived editor and claim-bearing outcomes.
    draw_multiline(c, 424, 190, "(c) Method: correct only the stale readout", size=8.2, color=colors.black)
    draw_box(c, 363, 136, 59, 31, "detect\nmismatch", BLUE, colors.HexColor("#EEF4FB"))
    draw_box(c, 437, 136, 59, 31, "choose +/-\nexpert", ORANGE, colors.HexColor("#FFF2E8"))
    draw_box(c, 400, 93, 59, 31, "protected-state\nveto", GREEN, colors.HexColor("#EAF7F4"))
    draw_arrow(c, 423, 151, 436, 151)
    draw_arrow(c, 466, 135, 448, 125, GREEN)
    c.setStrokeColor(GREEN); c.setFillColor(colors.HexColor("#F0FAF7")); c.setLineWidth(1.1)
    c.roundRect(359, 35, 141, 44, 5, stroke=1, fill=1)
    c.setFont(FIGURE_FONT_BOLD, 7.1); c.setFillColor(GREEN)
    c.drawCentredString(429.5, 65, "30.36% margin gap recovered")
    c.setFont(FIGURE_FONT, 6.3); c.setFillColor(colors.HexColor("#263238"))
    c.drawCentredString(429.5, 54, "+1.20 pp accuracy; 32 / 135 flips rescued")
    c.drawCentredString(429.5, 42, "0 / 2,856 protected controls changed")

    c.setStrokeColor(LIGHT_GRAY); c.setLineWidth(0.7)
    c.line(170, 32, 170, 194); c.line(352, 32, 352, 194)
    finish(c)


def build_behavior_boundary(discovery, replication, boundary) -> None:
    c, _, height = new_canvas("behavior_boundary.pdf", 7.0, 2.65)
    left = (34, 31, 206, 124); right = (292, 31, 196, 124)
    draw_multiline(c, 137, 176, "(a) Effect reverses with task requirement", size=8.3)
    draw_multiline(c, 390, 176, "(b) Failures concentrate near the boundary", size=8.3)
    xvals = [86, 190]
    ymin, ymax = -1.25, 1.85
    def ly(v): return left[1] + (v - ymin) / (ymax - ymin) * left[3]
    c.setStrokeColor(GRAY); c.line(left[0], ly(0), left[0] + left[2], ly(0)); c.line(left[0], left[1], left[0], left[1] + left[3])
    for offset, source, label, color in [(-3, discovery, "Discovery", BLUE), (3, replication, "Replication", ORANGE)]:
        est, boot = source["estimands"], source["cluster_bootstrap"]
        vals = [est["verification_history_effect"]["equal_weight_benchmark_mean"], est["deference_history_effect"]["equal_weight_benchmark_mean"]]
        cis = [boot["verification_history_effect"]["ci95"], boot["deference_history_effect"]["ci95"]]
        c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.2)
        c.line(xvals[0] + offset, ly(vals[0]), xvals[1] + offset, ly(vals[1]))
        for x, value, ci in zip(xvals, vals, cis):
            xx = x + offset; c.line(xx, ly(ci[0]), xx, ly(ci[1])); c.line(xx - 2.5, ly(ci[0]), xx + 2.5, ly(ci[0])); c.line(xx - 2.5, ly(ci[1]), xx + 2.5, ly(ci[1])); c.circle(xx, ly(value), 2.3, stroke=1, fill=1)
        draw_multiline(c, 47 + (0 if offset < 0 else 53), 163, label, size=6.8, color=color, align="left")
    draw_multiline(c, xvals[0], 21, "Verification\nrequired", size=6.8)
    draw_multiline(c, xvals[1], 21, "Delegation\nrequired", size=6.8)
    draw_multiline(c, 8, 109, "Obedience - verification margin", size=6.5, align="left")

    reqs = ["independent_verification", "delegated_choice"]
    bands = ["boundary", "middle", "robust"]
    centers = [325, 389, 453]
    def ry(v): return right[1] + v / 35.0 * right[3]
    c.setStrokeColor(GRAY); c.line(right[0], right[1], right[0], right[1] + right[3]); c.line(right[0], right[1], right[0] + right[2], right[1])
    for j, (req, color) in enumerate(zip(reqs, [BLUE, ORANGE])):
        vals = [pct(boundary["requirements"][req]["replication"][band]["correct_to_wrong_flip_rate"]) for band in bands]
        for center, value in zip(centers, vals):
            x = center + (-8 if j == 0 else 8); w = 14
            c.setFillColor(color); c.rect(x - w / 2, right[1], w, ry(value) - right[1], stroke=0, fill=1)
            draw_multiline(c, x, ry(value) + 4, f"{value:.1f}", size=6.2, color=color)
    for x, label in zip(centers, ["Low margin", "Middle", "Robust"]): draw_multiline(c, x, 19, label, size=6.8)
    draw_multiline(c, 267, 108, "Correct-to-wrong flips (%)", size=6.5, align="left")
    draw_multiline(c, 305, 163, "Verification", size=6.8, color=BLUE, align="left"); draw_multiline(c, 365, 163, "Delegation", size=6.8, color=ORANGE, align="left")
    finish(c)


def build_mechanism() -> None:
    c, _, _ = new_canvas("mechanism.pdf", 7.0, 2.55)
    draw_multiline(c, 137, 169, "(a) Qwen3.5-9B exact cache exchange", size=8.3)
    draw_multiline(c, 389, 169, "(b) Late readout replicates on Ascend", size=8.3)
    zero = 139; scale = 62
    c.setStrokeColor(GRAY); c.line(zero, 37, zero, 150)
    labels = ["Recurrent", "Full-attn KV", "Recurrent + KV"]
    rescue = [0.453, 1.184, 1.547]; reverse = [-0.349, -1.076, -1.542]
    ys = [129, 95, 61]
    for label, pos, p, n in zip(labels, ys, rescue, reverse):
        draw_multiline(c, 82, pos + 1, label, size=6.8, align="right")
        c.setFillColor(BLUE); c.rect(zero, pos + 2, p * scale, 8, stroke=0, fill=1)
        c.setFillColor(ORANGE); c.rect(zero + n * scale, pos - 8, -n * scale, 8, stroke=0, fill=1)
    draw_multiline(c, 136, 21, "Change in correct-vs-foil margin", size=6.8)
    draw_multiline(c, 42, 151, "Rescue", size=6.5, color=BLUE, align="left"); draw_multiline(c, 87, 151, "Reverse induction", size=6.5, color=ORANGE, align="left")

    x0, y0, w, h = 302, 40, 175, 107
    c.setStrokeColor(GRAY); c.line(x0, y0 + h / 2, x0 + w, y0 + h / 2); c.line(x0, y0, x0, y0 + h)
    layers = [19, 23, 27]; xs = [x0 + 20, x0 + w / 2, x0 + w - 20]
    rescue = [0.102, 0.307, 0.352]; reverse = [-0.165, -0.408, -0.470]
    def yy(v): return y0 + h / 2 + v * 95
    for vals, color in [(rescue, BLUE), (reverse, ORANGE)]:
        c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.2)
        for i in range(2): c.line(xs[i], yy(vals[i]), xs[i + 1], yy(vals[i + 1]))
        for x, value in zip(xs, vals): c.circle(x, yy(value), 2.3, stroke=1, fill=1)
    for x, layer in zip(xs, layers): draw_multiline(c, x, 29, str(layer), size=6.8)
    draw_multiline(c, x0 + w / 2, 18, "Patched residual layer", size=6.8)
    draw_multiline(c, 307, 151, "Rescue", size=6.5, color=BLUE, align="left"); draw_multiline(c, 350, 151, "Reverse induction", size=6.5, color=ORANGE, align="left")
    finish(c)


def build_dge_results(same_identity, controls_summary, v4_controls) -> None:
    c, _, _ = new_canvas("dge_results.pdf", 7.0, 2.75)
    draw_multiline(c, 137, 184, "(a) Exact same-identity mitigation", size=8.3)
    draw_multiline(c, 390, 184, "(b) Applicability veto removes protected edits", size=8.3)
    methods = [
        "fixed_negative_vector", "symmetric_rank_one", "shared_single_expert",
        "positive_only_expert", "negative_only_expert",
        "dge_without_structural_routing", "full_dge",
    ]
    names = ["Neg.\nvector", "Rank-1", "Shared", "Pos.\nonly", "Neg.\nonly", "No\nrouter", "Dir.\ncore"]
    reports = [same_identity["method_reports"][method] for method in methods]
    vals = [pct(report["normalized_gap_reduction"]["equal_weight_benchmark_mean"])
            for report in reports]
    cis = [report["normalized_gap_reduction_bootstrap"]["ci95"] for report in reports]
    baseline_y, scale = 54, 3.45
    c.setStrokeColor(GRAY); c.line(27, baseline_y, 251, baseline_y)
    xs = [43, 75, 107, 139, 171, 203, 235]
    bar_colors = [GRAY, GRAY, BLUE, BLUE, BLUE, BLUE, GREEN]
    for x, name, value, ci, color in zip(xs, names, vals, cis, bar_colors):
        c.setFillColor(color)
        if value >= 0: c.rect(x - 9, baseline_y, 18, value * scale, stroke=0, fill=1)
        else: c.rect(x - 9, baseline_y + value * scale, 18, -value * scale, stroke=0, fill=1)
        c.setStrokeColor(colors.black); c.line(x, baseline_y + pct(ci[0]) * scale, x, baseline_y + pct(ci[1]) * scale); c.line(x - 3, baseline_y + pct(ci[0]) * scale, x + 3, baseline_y + pct(ci[0]) * scale); c.line(x - 3, baseline_y + pct(ci[1]) * scale, x + 3, baseline_y + pct(ci[1]) * scale)
        draw_multiline(c, x, 40, name, size=5.5)
        draw_multiline(c, x, baseline_y + value * scale + 6, f"{value:.1f}", size=5.7, color=color)
    draw_multiline(c, 7, 135, "Normalized gap reduction (%)", size=6.3, align="left")

    order = ["matched_verification", "matched_delegated_choice", "supported_user_authority", "factual_boundary_memory", "fresh_verification", "explicit_governance_reset"]
    labels = ["Matched-V", "Matched-D", "Authority", "Memory", "Fresh", "Reset"]
    v3_changed = [controls_summary[key]["changed_rate"] for key in order]
    v4_changed = [pct(v4_controls["family_reports"][key]
                      ["application_gated_identity"]["changed_fraction"])
                  for key in order]
    x0, y0, h = 285, 47, 119
    c.setStrokeColor(GRAY); c.line(x0, y0, 493, y0); c.line(x0, y0, x0, y0 + h)
    xs2 = [305, 339, 373, 407, 441, 475]
    for x, label, v3_value, v4_value in zip(xs2, labels, v3_changed, v4_changed):
        c.setFillColor(ORANGE)
        c.rect(x - 10, y0, 8, v3_value / 100 * h, stroke=0, fill=1)
        c.setStrokeColor(GREEN)
        c.setLineWidth(2.0)
        v4_y = y0 + v4_value / 100 * h
        c.line(x + 2, v4_y, x + 10, v4_y)
        draw_multiline(c, x, 36, label, size=5.9)
    draw_multiline(c, 291, 174, "Directional core", size=6.4, color=ORANGE, align="left")
    draw_multiline(c, 365, 174, "Full C-DGE", size=6.4, color=GREEN, align="left")
    draw_multiline(c, 426, 174, "963 -> 0 edits", size=6.4, color=colors.black, align="left")
    draw_multiline(c, 264, 116, "Rows edited (%)", size=6.5, align="left")
    finish(c)


def main() -> None:
    full = load_json("qwen3_8b_full_behavior_analysis.json")
    discovery = load_json("qwen3_8b_crossover_discovery.json")
    replication = load_json("qwen3_8b_crossover_replication.json")
    boundary = load_json("qwen3_8b_boundary_susceptibility.json")
    negative = load_json("negative_vector_behavior_analysis.json")
    v1 = load_json("amsge_v1_behavior_analysis.json")
    v2 = load_json("amsge_v2_behavior_analysis.json")
    dge = load_json("dge_operator_dev_analysis.json")
    same_identity = load_json("dge_same_identity_comparison.json")
    v4_behavior = load_json("adsge_v4_behavior_operator_dev_diagnostic.json")
    v4_controls = load_json("adsge_v4_protected_controls_diagnostic.json")
    systems = load_json("dge_systems_metrics.json")
    dge_rows = load_jsonl("dge_operator_dev_rows.jsonl")
    control_rows = load_jsonl("dge_controls_rows.jsonl")

    assert full["audit"]["row_count"] == 27648 and full["audit"]["success"]
    assert discovery["audit"]["row_count"] == 6144 and discovery["audit"]["success"]
    assert replication["audit"]["row_count"] == 6144 and replication["audit"]["success"]
    assert dge["audit"]["row_count"] == 3072 and dge["audit"]["success"]
    assert same_identity["success"] is True
    assert same_identity["expected_rows_per_method"] == 3072
    assert len(same_identity["methods"]) == 7
    assert same_identity["all_cross_shard_baseline_logits_identical"] is True
    assert v4_behavior["audit"]["row_count"] == 3072 and v4_behavior["audit"]["success"]
    v4_v3_comparison = v4_behavior["comparison_to_v3_same_identity"]
    assert v4_v3_comparison["normalized_gap_reduction"]["v4_minus_v3"] == 0.0
    assert v4_v3_comparison["match_advantage_reduction"]["v4_minus_v3"] == 0.0
    assert v4_v3_comparison["matched_minus_mismatched_reduction"]["v4_minus_v3"] == 0.0
    assert all(report["v4_minus_v3"] == 0.0 for report in
               v4_v3_comparison["gap_reduction_by_benchmark"].values())
    assert v4_controls["audit"]["row_count"] == 2856 and v4_controls["audit"]["success"]
    assert v4_controls["application_gated_identity"]["exact"] is True
    assert v4_controls["application_gated_identity"]["changed_rows"] == 0
    assert v4_controls["application_routing"]["structural_gate_active_rows"] == 1152
    assert v4_controls["application_routing"]["v4_gate_active_rows"] == 0
    assert v4_controls["all_forced_direction_reference_gates_pass"] is True
    assert v4_controls["fit_gates_passed"] is False
    assert v4_controls["candidate_eligible"] is False
    assert v4_controls["final_test_open"] is False
    assert v4_controls["production_rollout_approved"] is False
    assert len(dge_rows) == 3072 and len(control_rows) == 2856
    comparison = systems["developmental_comparison_identity"]
    assert comparison["dge"]["job_key_sha256"] == dge["audit"]["observed_key_sha256"]
    assert comparison["fixed_negative_vector"]["job_key_sha256"] == negative["audit"]["observed_key_sha256"]
    assert comparison["exact_job_identity_intersection"] == 0
    assert comparison["item_identity_intersection"] == 0
    assert comparison["same_identity_head_to_head_complete"] is False
    assert systems["checkpoint"]["sha256"] == dge["checkpoint_sha256"]
    assert systems["parameters"]["editor_trainable"] == 25586
    assert systems["parameters"]["base_model_trainable"] == 0
    assert systems["intervention_norm"]["mean_sum_site_relative_norm"] == dge["intervention_norm"]["mean_sum_site_relative_norm"]
    assert systems["intervention_norm"]["max_sum_site_relative_norm"] == dge["intervention_norm"]["max_sum_site_relative_norm"]
    assert systems["provenance"]["dge_behavior_analysis_sha256"] == sha256_file(DATA / "dge_operator_dev_analysis.json")
    assert systems["provenance"]["negative_vector_behavior_analysis_sha256"] == sha256_file(DATA / "negative_vector_behavior_analysis.json")

    cells = defaultdict(lambda: {"n": 0, "baseline_correct": 0, "edited_correct": 0})
    for row in dge_rows:
        key = "matched" if row["matched_context"] else "mismatched"
        cells[key]["n"] += 1
        cells[key]["baseline_correct"] += row["baseline"]["task_aligned_margin"] > 0
        cells[key]["edited_correct"] += row["edited"]["task_aligned_margin"] > 0
    accuracy = {}
    for key, value in cells.items():
        accuracy[key] = {
            **value,
            "baseline_accuracy": value["baseline_correct"] / value["n"],
            "edited_accuracy": value["edited_correct"] / value["n"],
        }

    controls = defaultdict(lambda: {"rows": 0, "changed": 0, "baseline_correct": 0,
                                    "edited_correct": 0, "correct_to_wrong": 0,
                                    "wrong_to_correct": 0})
    for row in control_rows:
        value = controls[row["control_family"]]
        before = row["baseline"]["correct_margin"] > 0
        after = row["application_gated"]["correct_margin"] > 0
        value["rows"] += 1
        value["changed"] += row["application_gated_selected_logit_error"] != 0
        value["baseline_correct"] += before
        value["edited_correct"] += after
        value["correct_to_wrong"] += before and not after
        value["wrong_to_correct"] += (not before) and after
    controls_summary = {}
    for key, value in controls.items():
        controls_summary[key] = {
            **value,
            "changed_rate": pct(value["changed"] / value["rows"]),
            "baseline_accuracy": pct(value["baseline_correct"] / value["rows"]),
            "edited_accuracy": pct(value["edited_correct"] / value["rows"]),
            "accuracy_delta_pp": pct((value["edited_correct"] - value["baseline_correct"]) / value["rows"]),
        }

    claims = {
        "schema_version": 1,
        "publication_method_name": "Directional Governance Editing (DGE)",
        "behavior": {
            "full_rows": full["audit"]["row_count"],
            "obedience_minus_verification": full["mismatch_obedience_minus_verification"],
            "mismatch_ci95": full["mismatch_cluster_bootstrap"]["ci95"],
            "reset_minus_obedience": full["reset_minus_obedience"],
            "reset_ci95": full["reset_cluster_bootstrap"]["ci95"],
            "crossover_discovery": discovery["estimands"],
            "crossover_replication": replication["estimands"],
            "crossover_replication_ci": replication["cluster_bootstrap"],
        },
        "boundary": boundary["requirements"],
        "mitigation": {
            "negative_vector": negative,
            "amsge_v1": v1,
            "amsge_v2": v2,
            "dge": dge,
            "same_identity": same_identity,
            "dge_accuracy": accuracy,
            "dge_controls": controls_summary,
            "v4_composite_behavior_diagnostic": v4_behavior,
            "v4_composite_controls_diagnostic": v4_controls,
        },
        "systems": systems,
        "interpretation": {
            "dge_held_out_efficacy": "strong",
            "average_accuracy_collateral_observed": "small",
            "router_selectivity": "incomplete",
            "v4_composite_router_diagnostic": "removes all 963 observed V3 protected-control edits while preserving exact V3 operator_dev behavior",
            "v4_evidence_class": "post_fit_failure_diagnostic_not_confirmatory_selection",
            "deployment_safe": False,
            "production_rollout_approved": False,
            "negative_vector_comparison": "exact same-identity paired comparison",
            "same_identity_negative_vector_rerun_required": False,
        },
    }
    GENERATED.mkdir(parents=True, exist_ok=True)
    with (GENERATED / "paper_claims.json").open("w", encoding="utf-8") as handle:
        json.dump(claims, handle, indent=2, sort_keys=True)
        handle.write("\n")

    build_overview()
    build_behavior_boundary(discovery, replication, boundary)
    build_mechanism()
    build_dge_results(same_identity, controls_summary, v4_controls)


if __name__ == "__main__":
    main()
