#!/usr/bin/env python3
"""Materialize one SHA-bound code-v84 Qwen3.5 capture authorization."""
import argparse, datetime as dt, json, re
from pathlib import Path
from scripts.benchmark_v1.common import atomic_write_text, sha256_file

def src(p): return {"path":str(p),"sha256":sha256_file(p)}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("governance","gradient","protected"),required=True)
 for n in ("code_root","selector_snapshot","replication_protocol","crossover_contract","manifest","model_contract","design_audit","controls","archive_helper","archive_coordinator","pull_helper","output"): p.add_argument("--"+n.replace("_","-"),type=Path)
 p.add_argument("--selected-node",required=True); p.add_argument("--created-utc"); a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 if str(a.code_root)!="/workspace/context-mismatch-qwen3-5-9b/code-v84": raise ValueError("code-v84 required")
 if not re.fullmatch(r"a[0-9]{2}",a.selected_node): raise ValueError("invalid node")
 protocol=json.loads(a.replication_protocol.read_text()); snap=json.loads(a.selector_snapshot.read_text())
 if protocol.get("status")!="frozen_before_any_qwen3_5_9b_cdge_forward" or snap.get("node")!=a.selected_node: raise ValueError("frozen input mismatch")
 rows={"governance":6144,"gradient":6144,"protected":4008}[a.stage]; created=a.created_utc or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
 v={"schema_version":1,"authorization_id":f"qwen3-5-9b-cdge-{a.stage}-capture-code-v84","created_utc":created,"stage":f"qwen35_cdge_{a.stage}_capture","method":"C-DGE-V4.1","code_root":str(a.code_root),"immutable_code_bundle_manifest_sha256":sha256_file(a.code_root/"bundle.sha256"),"execution_node":a.selected_node,"slurm_partition":"a01","execution_allowed":True,"replication_protocol_sha256":sha256_file(a.replication_protocol),"crossover_contract_sha256":sha256_file(a.crossover_contract),"benchmark_manifest_sha256":sha256_file(a.manifest),"model_contract_sha256":sha256_file(a.model_contract),"expected_rows":rows,"site":"27:mlp","node_selection_snapshot":src(a.selector_snapshot),"model_contract":src(a.model_contract),"monitoring_contract":{"continuous_watch_required":True,"poll_seconds":1,"archive_helper":src(a.archive_helper),"archive_coordinator":src(a.archive_coordinator),"pull_helper":src(a.pull_helper)},"resource_contract":{"partition":"a01","nodes":1,"ntasks":1,"cpus_per_task":8,"mem_mib":131072,"npu_type":"910B3","npus":1,"time_limit":"12:00:00","node":a.selected_node},"final_test_open":False,"final_test_open_count":0,"production_rollout_approved":False}
 if a.stage=="protected": v["controls_sha256"]=sha256_file(a.controls)
 else: v.update({"design_audit_sha256":sha256_file(a.design_audit),"expected_key_sha256":"488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35"})
 atomic_write_text(a.output,json.dumps(v,indent=2,sort_keys=True)+"\n"); print(json.dumps(v,sort_keys=True))
if __name__=="__main__": main()
