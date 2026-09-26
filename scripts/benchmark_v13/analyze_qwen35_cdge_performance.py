#!/usr/bin/env python3
"""Audit and label the Qwen3.5 C-DGE performance experiment."""

from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.benchmark_v1.common import atomic_write_text,load_jsonl,sha256_file
from scripts.benchmark_v10.analyze_cdge_v4_1_performance import analyze

def main():
 p=argparse.ArgumentParser()
 for n in ("input","environment","evaluation_contract","execution_authorization","output"):
  p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
 p.add_argument("--bootstrap-replicates",type=int,default=10000);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 r=analyze(load_jsonl(a.input),json.loads(a.environment.read_text()),json.loads(a.evaluation_contract.read_text()),replicates=a.bootstrap_replicates)
 r.update({"model":"Qwen3.5-9B","cross_model_stage":"qwen35_cdge_performance",
  "input_sha256":sha256_file(a.input),"environment_sha256":sha256_file(a.environment),
  "evaluation_contract_sha256":sha256_file(a.evaluation_contract),"authorization_sha256":sha256_file(a.execution_authorization)})
 atomic_write_text(a.output,json.dumps(r,indent=2,sort_keys=True)+"\n");print(json.dumps(r,indent=2,sort_keys=True))
 if not r["audit"]["success"]:raise SystemExit(1)
if __name__=="__main__":main()
