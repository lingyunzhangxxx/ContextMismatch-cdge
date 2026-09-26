#!/usr/bin/env python3
"""Audit fresh Qwen3.5 application-gated and forced-expert controls."""

from __future__ import annotations

import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_controls import _bootstrap_upper
from scripts.benchmark_v8.analyze_v4_protected_controls_diagnostic import FAMILIES, FORCED_DIRECTIONS, _identity, _routing
from scripts.benchmark_v13.qwen35_eval_contract import CONTROL_KEY, CONTROL_ROWS, METHOD


def main() -> None:
    p=argparse.ArgumentParser()
    for name in ("input","environment","evaluation_contract","execution_authorization","output"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--bootstrap-replicates",type=int,default=10000); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    rows=load_jsonl(a.input); env=json.loads(a.environment.read_text()); keys=Counter(r["case_key"] for r in rows)
    observed=hashlib.sha256(("\n".join(sorted(keys))+"\n").encode()).hexdigest()
    nonfinite=[]
    for r in rows:
        nums=[*r["baseline"].values(),*r["application_gated"].values(),r["application_gated_selected_logit_error"],r["application_gated_margin_change"],r["application_gated_binary_kl"]]
        for d in FORCED_DIRECTIONS: nums += [*r[f"forced_on_{d}"].values(),r[f"forced_on_{d}_margin_change"],r[f"forced_on_{d}_binary_kl"]]
        if not all(math.isfinite(float(v)) for v in nums): nonfinite.append(r["case_key"])
    families={}; all_identity=True; all_forced=True
    for fi,family in enumerate(FAMILIES):
        selected=[r for r in rows if r["control_family"]==family]
        identity=_identity(selected); forced={}
        for di,direction in enumerate(FORCED_DIRECTIONS):
            field=f"forced_on_{direction}_margin_loss"
            for r in selected: r[field]=-float(r[f"forced_on_{direction}_margin_change"])
            loss=_bootstrap_upper(selected,field,a.bootstrap_replicates,91011+fi*10+di)
            kl=_bootstrap_upper(selected,f"forced_on_{direction}_binary_kl",a.bootstrap_replicates,92013+fi*10+di)
            passed=float(loss["upper_one_sided_95"])<=0.25 and float(kl["upper_one_sided_95"])<=0.02
            forced[direction]={"rows":len(selected),"margin_loss":loss,"margin_loss_reference_maximum":0.25,
                               "binary_kl":kl,"binary_kl_reference_maximum":0.02,"reference_gate_pass":passed,
                               "same_frozen_expert_metric_as_v3":True,
                               "equality_basis":"forced wrapper uses the exact directional expert embedded in the application checkpoint"}
            all_forced &= passed
        families[family]={"rows":len(selected),"application_gated_identity":identity,
                          "application_routing":_routing(selected),"forced_on":forced}
        all_identity &= identity["exact"]
    audit_success=(len(rows)==CONTROL_ROWS==env.get("planned_rows") and len(keys)==CONTROL_ROWS and all(v==1 for v in keys.values()) and observed==CONTROL_KEY==env.get("expected_key_sha256") and not nonfinite and set(r["control_family"] for r in rows)==set(FAMILIES))
    report={"schema_version":1,"method":METHOD,"runtime_checkpoint_method":"ADSGE-V4",
            "stage":"qwen35_cdge_protected_controls","evaluation_stage":"qwen35_cdge_protected_controls",
            "model":"Qwen3.5-9B","candidate_id":env["candidate_id"],"site":env["site"],
            "audit":{"row_count":len(rows),"planned_rows":env.get("planned_rows"),"unique_case_keys":len(keys),
                     "observed_key_sha256":observed,"expected_key_sha256":env.get("expected_key_sha256"),
                     "nonfinite_case_keys":nonfinite[:20],"success":audit_success},
            "application_gated_identity":_identity(rows),"application_routing":_routing(rows),
            "external_zero_gate_identity":_identity([r for r in rows if float(r["operator_applicable"])==0]),
            "family_reports":families,"all_six_application_gated_exact_identity":all_identity,
            "all_forced_direction_reference_gates_pass":all_forced,"forced_directions":list(FORCED_DIRECTIONS),
            "controls_evaluation_complete":audit_success,"candidate_eligible":True,"candidate_may_be_locked":False,
            "input_sha256":sha256_file(a.input),"environment_sha256":sha256_file(a.environment),
            "evaluation_contract_sha256":sha256_file(a.evaluation_contract),
            "authorization_sha256":sha256_file(a.execution_authorization),
            "final_test_open":False,"final_test_open_count":0,"production_rollout_approved":False}
    atomic_write_text(a.output,json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2,sort_keys=True))
    if not audit_success: raise SystemExit(1)


if __name__=="__main__": main()
