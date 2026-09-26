#!/usr/bin/env python3
"""Prospective Qwen3.5 C-DGE capture stages.

V4.1 remains compatible with its frozen 27:mlp recipe.  New bundles may bind a
model-native site manifest; in that mode no numerical layer ID is accepted
from code or command-line convenience arguments.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v1.controls import factual_memory_messages, factual_memory_prompt, load_controls, supported_authority_prompt
from scripts.benchmark_v1.histories import prefix_messages, system_message
from scripts.benchmark_v1.run_behavior import _chat_ids, _device, _exact_prefix, _label_ids, _split_final_user, _token_hash
from scripts.benchmark_v1.run_mechanism_discovery import _load_model
from scripts.benchmark_v1.run_protected_capture import _benchmark_cases, _capture_group, _prefill_with_last_boundary_capture
from scripts.benchmark_v1.run_subspace_capture import _suffix_component_capture
from scripts.benchmark_v2.crossover import audit_design, enumerate_jobs, job_key, key_hash
from scripts.benchmark_v3.run_governance_capture import _write_shard as _write_state_shard
from scripts.benchmark_v4.run_margin_gradient_capture import (
    _gradient_safe_prefill,
    _suffix_margin_gradient_capture,
    _write_shard as _write_gradient_shard,
)

LEGACY_SITE = (27, "mlp")
EXPECTED = {"governance": 6144, "gradient": 6144, "protected": 4008}


def _site_key(site: tuple[int, str]) -> str:
    return f"{int(site[0])}:{site[1]}"


def _resolve_sites(args: argparse.Namespace, protocol: dict, authorization: dict) -> list[tuple[int, str]]:
    if args.site_manifest is None:
        frozen = protocol["replication_scope"]["frozen_site"]
        sites = [(int(frozen["layer"]), str(frozen["component"]))]
        if sites != [LEGACY_SITE] or authorization.get("site") != _site_key(LEGACY_SITE):
            raise ValueError("legacy frozen site mismatch")
        return sites
    manifest = json.loads(args.site_manifest.read_text())
    required = {
        "stage": "qwen35_cdge_v4_2_native_site_selection",
        "method": "C-DGE-V4.2",
        "locked": True,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"native site manifest mismatch: {field}")
    sites = [
        (int(row["layer"]), str(row["component"]))
        for row in manifest.get("selected_sites_ordered", [])
    ]
    if not sites or len(sites) != len(set(sites)):
        raise ValueError("native site manifest is empty or duplicated")
    if authorization.get("site_manifest_sha256") != sha256_file(args.site_manifest):
        raise ValueError("capture authorization site-manifest binding mismatch")
    if authorization.get("sites") != [_site_key(site) for site in sites]:
        raise ValueError("capture authorization dynamic-site list mismatch")
    return sites


def _validate(args: argparse.Namespace) -> tuple[dict, dict, dict, list[tuple[int, str]]]:
    protocol = json.loads(args.replication_protocol.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    model_contract = json.loads(args.model_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    if protocol.get("status") not in {
        "frozen_before_any_qwen3_5_9b_cdge_forward",
        "frozen_before_qwen3_5_native_discovery_forward",
    }:
        raise ValueError("replication protocol is not frozen")
    for field, expected in {"final_test_open": False, "final_test_open_count": 0, "production_rollout_approved": False}.items():
        if protocol.get(field) != expected or authorization.get(field) != expected:
            raise ValueError(f"safety mismatch: {field}")
    if model_contract.get("contract") != "context-mismatch-qwen35-9b-bf16-v1" or model_contract.get("non_quantized") is not True:
        raise ValueError("invalid Qwen3.5 model contract")
    legacy = protocol.get("method_short_name") == "C-DGE-V4.1"
    stage = f"qwen35_cdge_{args.stage}_capture" if legacy else f"qwen35_cdge_v4_2_{args.stage}_capture"
    required = {
        "stage": stage, "method": protocol["method_short_name"], "execution_allowed": True,
        "replication_protocol_sha256": sha256_file(args.replication_protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_contract_sha256": sha256_file(args.model_contract),
        "expected_rows": EXPECTED[args.stage],
    }
    if args.stage == "protected":
        required["controls_sha256"] = sha256_file(args.controls)
    else:
        required["design_audit_sha256"] = sha256_file(args.design_audit)
        required["expected_key_sha256"] = "488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35"
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"authorization mismatch: {field}")
    if sha256_file(args.model_contract) != protocol["frozen_lineage"]["model_contract_sha256"]:
        raise ValueError("model lineage mismatch")
    if sha256_file(args.manifest) != protocol["frozen_lineage"]["benchmark_manifest_sha256"]:
        raise ValueError("benchmark lineage mismatch")
    sites = _resolve_sites(args, protocol, authorization)
    return protocol, crossover, authorization, sites


def _jobs(args: argparse.Namespace, crossover: dict) -> list[dict]:
    jobs = enumerate_jobs(load_jsonl(args.manifest), crossover, "replication")
    if len(jobs) != 6144 or not audit_design(jobs)["success"] or key_hash(jobs) != "488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35":
        raise RuntimeError("replication design mismatch")
    audit = json.loads(args.design_audit.read_text())
    if audit.get("stage") != "replication" or not audit.get("audit", {}).get("success") or audit.get("audit", {}).get("expected_key_sha256") != key_hash(jobs):
        raise ValueError("design audit mismatch")
    return jobs


def _load(args: argparse.Namespace):
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, "eager")
    model.requires_grad_(False)
    model.eval()
    return device, tokenizer, model, _label_ids(tokenizer)


def _governance(args: argparse.Namespace, protocol: dict, crossover: dict, authorization: dict, sites: list[tuple[int, str]]) -> dict:
    jobs = _jobs(args, crossover)
    device, tokenizer, model, label_ids = _load(args)
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job["declared_role"], job["history_condition"], job["history_style"], job["history_depth"], job["history_realization"])].append(job)
    records, observed = [], []
    for prefix_key in sorted(grouped):
        role, history, style, depth, realization = prefix_key
        messages = prefix_messages(*prefix_key)
        prefix_ids = _exact_prefix(tokenizer, messages)
        layers = sorted({site[0] for site in sites})
        base_cache, boundary = _prefill_with_last_boundary_capture(model, device, prefix_ids, layers)
        metadata = []
        inputs = {_site_key(site): [] for site in sites}
        outputs = {_site_key(site): [] for site in sites}
        for job in sorted(grouped[prefix_key], key=job_key):
            key = job_key(job); prompt = job["prompt"]
            rendered, suffix = _split_final_user(tokenizer, _chat_ids(tokenizer, messages + [{"role":"user","content":prompt["prompt"]}]))
            if rendered != prefix_ids: raise RuntimeError("prefix mismatch")
            current_inputs, current_outputs, logits, extended = _suffix_component_capture(model, device, copy.deepcopy(base_cache), len(prefix_ids), suffix, label_ids, sites)
            values = {"A": float(logits[0]), "B": float(logits[1])}
            row = {"job_key":key,"benchmark":job["item"]["benchmark"],"item_id":job["item"]["item_id"],"partition":job["item"]["partition"],"declared_role":role,"history_condition":history,"history_style":style,"history_depth":depth,"history_realization":realization,"task_requirement":job["task_requirement"],"target_obedience":prompt["target_obedience"],"label_swap":job["label_swap"],"task_correct_label":prompt["task_correct_label"],"task_foil_label":prompt["task_foil_label"],"factual_correct_label":prompt["factual_correct_label"],"factual_foil_label":prompt["factual_foil_label"],"logit_a":values["A"],"logit_b":values["B"],"task_aligned_margin":values[prompt["task_correct_label"]]-values[prompt["task_foil_label"]],"factual_margin":values[prompt["factual_correct_label"]]-values[prompt["factual_foil_label"]],"prefix_tokens":len(prefix_ids),"suffix_tokens":len(suffix),"suffix_token_sha256_int32_le":_token_hash(suffix)}
            if not all(math.isfinite(float(row[k])) for k in ("logit_a","logit_b","task_aligned_margin","factual_margin")): raise FloatingPointError("non-finite state capture")
            metadata.append(row); observed.append(key)
            for site in sites:
                inputs[_site_key(site)].append(current_inputs[site])
                outputs[_site_key(site)].append(current_outputs[site])
            del extended
        prefix = {"declared_role":role,"history_condition":history,"history_style":style,"history_depth":depth,"history_realization":realization}
        records.append(_write_state_shard(args.output_dir, f"capture__{role}__{history}__{style}__r{realization}.pt", metadata=metadata, boundary_states=boundary, component_inputs=inputs, component_outputs=outputs, prefix=prefix))
        del base_cache; gc.collect()
        if device.type == "npu": torch.npu.empty_cache()
    return _finish(args, protocol, authorization, sites, observed, records, "capture_manifest.json", {"counterfactual_pairs":3072,"base_model_parameter_gradients":False})


def _gradient(args: argparse.Namespace, protocol: dict, crossover: dict, authorization: dict, sites: list[tuple[int, str]]) -> dict:
    jobs = _jobs(args, crossover)
    device, tokenizer, model, label_ids = _load(args)
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job["declared_role"], job["history_condition"], job["history_style"], job["history_depth"], job["history_realization"])].append(job)
    records, observed = [], []
    nonzero_by_site = {_site_key(site): 0 for site in sites}
    for prefix_key in sorted(grouped):
        role, history, style, depth, realization = prefix_key; messages = prefix_messages(*prefix_key)
        prefix_ids = _exact_prefix(tokenizer, messages); cache = _gradient_safe_prefill(model, device, prefix_ids)
        metadata, gradients = [], {_site_key(site): [] for site in sites}
        for job in sorted(grouped[prefix_key], key=job_key):
            key=job_key(job); prompt=job["prompt"]
            rendered,suffix=_split_final_user(tokenizer,_chat_ids(tokenizer,messages+[{"role":"user","content":prompt["prompt"]}]))
            if rendered!=prefix_ids: raise RuntimeError("prefix mismatch")
            current,logits=_suffix_margin_gradient_capture(model,device,cache,len(prefix_ids),suffix,label_ids,sites,prompt["task_correct_label"],prompt["task_foil_label"])
            for site in sites:
                grad = current[site]
                current_key = _site_key(site)
                nonzero_by_site[current_key] += int(
                    float(torch.linalg.vector_norm(grad.float())) > 0
                )
                gradients[current_key].append(grad)
            values={"A":logits[0],"B":logits[1]}; margin=values[prompt["task_correct_label"]]-values[prompt["task_foil_label"]]
            if not math.isfinite(margin): raise FloatingPointError("non-finite gradient capture")
            metadata.append({"job_key":key,"benchmark":job["item"]["benchmark"],"item_id":job["item"]["item_id"],"partition":job["item"]["partition"],"declared_role":role,"history_condition":history,"history_style":style,"history_depth":depth,"history_realization":realization,"task_requirement":job["task_requirement"],"target_obedience":prompt["target_obedience"],"label_swap":job["label_swap"],"task_correct_label":prompt["task_correct_label"],"task_foil_label":prompt["task_foil_label"],"logit_a":values["A"],"logit_b":values["B"],"task_aligned_margin":margin,"prefix_tokens":len(prefix_ids),"suffix_tokens":len(suffix),"suffix_token_sha256_int32_le":_token_hash(suffix)})
            observed.append(key)
        prefix={"declared_role":role,"history_condition":history,"history_style":style,"history_depth":depth,"history_realization":realization}
        records.append(_write_gradient_shard(args.output_dir,f"margin_gradients__{role}__{history}__{style}__r{realization}.pt",metadata=metadata,gradients=gradients,prefix=prefix))
        del cache; gc.collect()
        if device.type=="npu": torch.npu.empty_cache()
    if any(value == 0 for value in nonzero_by_site.values()):
        raise RuntimeError("one or more captured sites have only zero gradients")
    fractions = {key: value / 6144 for key, value in nonzero_by_site.items()}
    return _finish(args, protocol, authorization, sites, observed, records, "margin_gradient_manifest.json", {"nonzero_gradient_fraction_by_site":fractions,"base_model_parameter_gradients":False})


def _protected(args: argparse.Namespace, protocol: dict, crossover: dict, authorization: dict, sites: list[tuple[int, str]]) -> dict:
    device, tokenizer, model, label_ids = _load(args); controls=load_controls(args.controls)
    items=[row for row in load_jsonl(args.manifest) if row["partition"]=="subspace_fit"]
    if len(items)!=192: raise RuntimeError("protected partition mismatch")
    by_realization=defaultdict(list)
    for item in items: by_realization[int(item["history_realization"])].append(item)
    records=[]
    for role in crossover["factorial"]["declared_roles"]:
        layers = sorted({site[0] for site in sites})
        records.append(_capture_group(tokenizer=tokenizer,model=model,device=device,label_ids=label_ids,messages=prefix_messages(role,"fresh","natural",32,0),cases=_benchmark_cases(items,"fresh",role,"none"),layers=layers,module_keys=sites,boundary_metadata={"control_family":"fresh","declared_role":role,"history_style":"none","history_realization":0,"verification_required":1.0},output_dir=args.output_dir,shard_name=f"protected__fresh__{role}.pt"))
        for style in crossover["factorial"]["history_styles"]:
            for realization, realization_items in sorted(by_realization.items()):
                for condition in ("verification","obedience_reset"):
                    records.append(_capture_group(tokenizer=tokenizer,model=model,device=device,label_ids=label_ids,messages=prefix_messages(role,condition,style,32,realization),cases=_benchmark_cases(realization_items,condition,role,style),layers=layers,module_keys=sites,boundary_metadata={"control_family":condition,"declared_role":role,"history_style":style,"history_realization":realization,"verification_required":1.0},output_dir=args.output_dir,shard_name=f"protected__{condition}__{role}__{style}__r{realization}.pt"))
                cases=[]
                for control in controls["supported_user_authority"]:
                    for swap in (0,1):
                        prompt,correct,foil=supported_authority_prompt(control,swap); cases.append({"prompt":prompt,"correct_label":correct,"foil_label":foil,"metadata":{"control_family":"supported_user_authority","control_id":control["id"],"declared_role":role,"history_style":style,"history_realization":realization,"label_swap":swap,"verification_required":0.0}})
                records.append(_capture_group(tokenizer=tokenizer,model=model,device=device,label_ids=label_ids,messages=prefix_messages(role,"obedience",style,32,realization),cases=cases,layers=layers,module_keys=sites,boundary_metadata={"control_family":"supported_user_authority","declared_role":role,"history_style":style,"history_realization":realization,"verification_required":0.0},output_dir=args.output_dir,shard_name=f"protected__supported_authority__{role}__{style}__r{realization}.pt"))
        for control in controls["factual_boundary_memory"]:
            cases=[]
            for swap in (0,1):
                prompt,correct,foil=factual_memory_prompt(control,swap); cases.append({"prompt":prompt,"correct_label":correct,"foil_label":foil,"metadata":{"control_family":"factual_boundary_memory","control_id":control["id"],"declared_role":role,"history_style":"none","history_realization":0,"label_swap":swap,"verification_required":0.0}})
            records.append(_capture_group(tokenizer=tokenizer,model=model,device=device,label_ids=label_ids,messages=[{"role":"system","content":system_message(role)}]+factual_memory_messages(control),cases=cases,layers=layers,module_keys=sites,boundary_metadata={"control_family":"factual_boundary_memory","control_id":control["id"],"declared_role":role,"history_style":"none","history_realization":0,"verification_required":0.0},output_dir=args.output_dir,shard_name=f"protected__factual_memory__{role}__{control['id']}.pt"))
    counts=defaultdict(int)
    for record in records: counts[record["control_family"]]+=record["rows"]
    expected={"fresh":768,"verification":1536,"obedience_reset":1536,"supported_user_authority":144,"factual_boundary_memory":24}
    if dict(counts)!=expected: raise RuntimeError(f"protected family mismatch: {dict(counts)}")
    value={"schema_version":1,"stage":authorization["stage"],"method":protocol["method_short_name"],"partition":"subspace_fit","rows":4008,"family_rows":expected,"sites":[{"layer":site[0],"component":site[1]} for site in sites],"hidden_size":4096,"controls_sha256":sha256_file(args.controls),"benchmark_manifest_sha256":sha256_file(args.manifest),"replication_protocol_sha256":sha256_file(args.replication_protocol),"site_manifest_sha256":sha256_file(args.site_manifest) if args.site_manifest else None,"authorization_sha256":sha256_file(args.execution_authorization),"shards":records,"complete":True,"final_test_open":False,"final_test_open_count":0,"production_rollout_approved":False}
    atomic_write_text(args.output_dir/"capture_manifest.json",json.dumps(value,indent=2,sort_keys=True)+"\n"); return value


def _finish(args, protocol, authorization, sites, observed, records, filename, extra):
    if len(observed)!=6144 or len(set(observed))!=6144: raise RuntimeError("row identity mismatch")
    observed_sha=hashlib.sha256(("\n".join(sorted(observed))+"\n").encode()).hexdigest()
    if observed_sha!="488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35": raise RuntimeError("key SHA mismatch")
    value={"schema_version":1,"stage":authorization["stage"],"method":protocol["method_short_name"],"partition":"subspace_fit","rows":6144,"unique_job_keys":6144,"expected_key_sha256":observed_sha,"observed_key_sha256":observed_sha,"sites":[{"layer":site[0],"component":site[1]} for site in sites],"hidden_size":4096,"shards":records,"replication_protocol_sha256":sha256_file(args.replication_protocol),"site_manifest_sha256":sha256_file(args.site_manifest) if args.site_manifest else None,"crossover_contract_sha256":sha256_file(args.crossover_contract),"benchmark_manifest_sha256":sha256_file(args.manifest),"design_audit_sha256":sha256_file(args.design_audit),"model_contract_sha256":sha256_file(args.model_contract),"authorization_sha256":sha256_file(args.execution_authorization),"complete":True,"final_test_open":False,"final_test_open_count":0,"production_rollout_approved":False,**extra}
    atomic_write_text(args.output_dir/filename,json.dumps(value,indent=2,sort_keys=True)+"\n"); return value


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=EXPECTED,required=True)
    for name in ("replication_protocol","crossover_contract","manifest","model_contract","model_path","execution_authorization","output_dir"): p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--design-audit",type=Path); p.add_argument("--controls",type=Path); p.add_argument("--site-manifest",type=Path); p.add_argument("--device",default="npu:0")
    args=p.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    if args.stage in {"governance","gradient"} and args.design_audit is None: p.error("--design-audit required")
    if args.stage=="protected" and args.controls is None: p.error("--controls required")
    protocol,crossover,authorization,sites=_validate(args); args.output_dir.mkdir(parents=True)
    value={"governance":_governance,"gradient":_gradient,"protected":_protected}[args.stage](args,protocol,crossover,authorization,sites)
    print(json.dumps(value,sort_keys=True))


if __name__=="__main__": main()
