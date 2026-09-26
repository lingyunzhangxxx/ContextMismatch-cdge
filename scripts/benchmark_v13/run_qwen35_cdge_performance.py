#!/usr/bin/env python3
"""Run the frozen 24-cell Qwen3.5 baseline/DSGE/C-DGE performance design."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

from scripts.benchmark_v1.common import load_jsonl, sha256_file
from scripts.benchmark_v10 import run_cdge_v4_1_performance as kernel
from scripts.benchmark_v10.cdge_performance_design import selected_performance_jobs
from scripts.benchmark_v13.qwen35_eval_contract import PERFORMANCE_CELLS, PERFORMANCE_ROWS, require_authorization, require_fit, require_receipt


def main() -> None:
    p=argparse.ArgumentParser()
    for name in ("directional_checkpoint","checkpoint","directional_contract","editor_contract","composite_contract",
                 "composite_report","fit_report","fit_receipt","evaluation_contract","execution_authorization",
                 "crossover_contract","manifest","manifest_report","design_audit","model_contract","model_path",
                 "behavior_analysis","behavior_receipt","controls_analysis","controls_receipt","output","environment_output"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--device",default="npu:0"); a=p.parse_args()
    if a.output.exists() or a.environment_output.exists(): raise FileExistsError("performance outputs exist")
    require_fit(fit_report=a.fit_report,fit_receipt=a.fit_receipt,directional_checkpoint=a.directional_checkpoint,
                applicability_checkpoint=a.checkpoint,composite_report=a.composite_report,evaluation_contract=a.evaluation_contract)
    require_receipt(a.behavior_receipt,run_prefix="qwen3-5-9b-cdge-v4-1-behavior-")
    require_receipt(a.controls_receipt,run_prefix="qwen3-5-9b-cdge-v4-1-protected-controls-")
    behavior=json.loads(a.behavior_analysis.read_text()); controls=json.loads(a.controls_analysis.read_text())
    if behavior.get("candidate_eligible") is not True or controls.get("audit",{}).get("success") is not True:
        raise ValueError("performance prerequisites are not eligible/audited")
    auth=require_authorization(a.execution_authorization,{"stage":"qwen35_cdge_performance",
        "expected_measurements":PERFORMANCE_ROWS,"expected_cells":PERFORMANCE_CELLS,
        "final_test_open":False,"final_test_open_count":0})
    for field,path in {"directional_checkpoint":a.directional_checkpoint,"applicability_checkpoint":a.checkpoint,
        "directional_contract":a.directional_contract,"applicability_contract":a.editor_contract,
        "composite_contract":a.composite_contract,"composite_report":a.composite_report,"fit_report":a.fit_report,
        "fit_receipt":a.fit_receipt,"evaluation_contract":a.evaluation_contract,"crossover_contract":a.crossover_contract,
        "manifest":a.manifest,"manifest_report":a.manifest_report,"model_contract":a.model_contract,
        "behavior_analysis":a.behavior_analysis,"behavior_receipt":a.behavior_receipt,
        "controls_analysis":a.controls_analysis,"controls_receipt":a.controls_receipt,
        "operator_dev_design_audit":a.design_audit}.items():
        if auth["bound_artifacts"].get(field,{}).get("sha256")!=sha256_file(path):
            raise ValueError(f"performance authorization artifact mismatch: {field}")
    crossover=json.loads(a.crossover_contract.read_text())
    jobs=selected_performance_jobs(load_jsonl(a.manifest),crossover)
    def validated(_args): return auth,crossover,jobs
    original_validate,original_argv=kernel._validate,sys.argv
    try:
        kernel._validate=validated
        sys.argv=["run_cdge_v4_1_performance","--v3-checkpoint",str(a.directional_checkpoint),
            "--v4-checkpoint",str(a.checkpoint),"--v3-editor-contract",str(a.directional_contract),
            "--v4-editor-contract",str(a.editor_contract),"--composite-contract",str(a.composite_contract),
            "--evaluation-contract",str(a.evaluation_contract),"--composite-report",str(a.composite_report),
            "--composite-authorization",str(a.execution_authorization),"--composite-receipt",str(a.fit_receipt),
            "--execution-authorization",str(a.execution_authorization),"--crossover-contract",str(a.crossover_contract),
            "--manifest",str(a.manifest),"--manifest-report",str(a.manifest_report),"--design-audit",str(a.design_audit),
            "--model-manifest",str(a.model_contract),"--model-path",str(a.model_path),"--output",str(a.output),
            "--environment-output",str(a.environment_output),"--device",a.device,"--attn-implementation","eager"]
        kernel.main()
    finally: kernel._validate,sys.argv=original_validate,original_argv


if __name__=="__main__": main()
