#!/usr/bin/env bash
set -euo pipefail

expected_user=${EXPERIMENT_USER:-researcher}
expected_node=${EXPERIMENT_NODE:-compute-node}
project_root=/workspace/context-mismatch-qwen3-8b
code_root=${1:?code root required}
run_dir=${2:?run directory required}
authorization=${3:?final execution authorization required}
shared_model=$project_root/models/Qwen3-8B
node_model=/workspace/node-local/context-mismatch-qwen3-8b/models/Qwen3-8B
runtime_root=/workspace/node-local/context-mismatch-ascend/runtime-v1
runtime_base=ascend-py311-torch271-cann83rc1-bullseye
runtime_overlay=cann9-py311-overlay-v2
runtime_base_sha=af7b5881a1f73c30f710661ab4934559befbf45c5f22d7ae65e44531d8c98555
runtime_overlay_sha=9e0ce3ebe2054e56b1bb28ce8f24448a794019aef991ec44752c5d2f0a479bd9
python_bin=$runtime_root/python-3.11.13-standalone/python/bin/python3
site_packages=$runtime_root/$runtime_base/rootfs/opt/distill/lib/python3.11/site-packages
overlay_packages=$runtime_root/$runtime_overlay

crossover_contract=$code_root/protocol/GOVERNANCE_TASK_CROSSOVER_V1.json
operator_contract=$code_root/protocol/QWEN3_8B_OPERATOR_CONTRACT_V1.json
extension_contract=$code_root/protocol/QWEN3_8B_BIDIRECTIONAL_OPERATOR_EXTENSION_V1.json
manifest=$code_root/artifacts/benchmark_manifest.jsonl
manifest_report=$code_root/artifacts/benchmark_manifest.report.json
design_audit=$code_root/artifacts/governance_crossover_final_test_design_audit_v2.json
model_contract=$code_root/model_source_contract.json
model_verifier=$code_root/verify_model_source.py
tree_hasher=$code_root/hash_artifact_tree.py

[[ $(id -un) == "$expected_user" ]] || { echo "refusing unexpected user" >&2; exit 2; }
[[ "$code_root" =~ ^$project_root/code-v[0-9]+$ ]] || { echo "invalid code root" >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} && $(hostname) == "$expected_node" ]] || {
  echo "refusing: final test requires an a07 Slurm allocation" >&2; exit 2;
}
[[ ${SLURM_CPUS_PER_TASK:-} == 8 ]] || { echo "final test requires 8 CPUs" >&2; exit 2; }
case "$run_dir" in
  "$project_root"/runs/qwen3-8b-operator-final-test-*) ;;
  *) echo "refusing unexpected run directory" >&2; exit 2 ;;
esac
for required in \
  "$python_bin" "$authorization" "$crossover_contract" "$operator_contract" \
  "$extension_contract" "$manifest" "$manifest_report" "$design_audit" \
  "$model_contract" "$model_verifier" "$tree_hasher" "$code_root/bundle.sha256"; do
  [[ -r "$required" ]] || { echo "missing prerequisite: $required" >&2; exit 2; }
done
for required_dir in "$site_packages" "$overlay_packages" "$shared_model" "$node_model" "$run_dir"; do
  [[ -d "$required_dir" ]] || { echo "missing prerequisite directory: $required_dir" >&2; exit 2; }
done

umask 077
status_tmp=$run_dir/.exit_status.json.tmp-$$
record_exit() {
  rc=$?
  printf '{"job_id":"%s","exit_code":%s,"hostname":"%s","stage":"operator_final_test","final_test_open":true,"final_test_open_count":1,"production_rollout_approved":false}\n' \
    "$SLURM_JOB_ID" "$rc" "$(hostname)" >"$status_tmp"
  mv -f -- "$status_tmp" "$run_dir/exit_status.json"
}
trap record_exit EXIT

readarray -t sources < <("$python_bin" - \
  "$authorization" "$code_root" "$crossover_contract" "$operator_contract" \
  "$extension_contract" "$manifest" "$design_audit" <<'PY'
import hashlib, json, os, sys
authorization_path, code_root, crossover, operator, extension, manifest, design = sys.argv[1:]

def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

value=json.load(open(authorization_path))
required={
    "stage":"operator_final_test",
    "code_root":code_root,
    "immutable_code_bundle_manifest_sha256":sha256(os.path.join(code_root,"bundle.sha256")),
    "execution_allowed":True,
    "final_test_open":True,
    "final_test_open_count":1,
    "production_rollout_approved":False,
    "crossover_contract_sha256":sha256(crossover),
    "operator_contract_sha256":sha256(operator),
    "extension_contract_sha256":sha256(extension),
    "benchmark_manifest_sha256":sha256(manifest),
    "design_audit_sha256":sha256(design),
    "expected_rows":6144,
}
for field, expected in required.items():
    if value.get(field) != expected:
        raise SystemExit(f"final authorization mismatch: {field}")
for name in ("operator_lock","candidate_manifest","candidate_tensor","model_manifest"):
    artifact=value.get(name,{})
    path=artifact.get("path","")
    if not path.startswith("/workspace/context-mismatch-qwen3-8b/"):
        raise SystemExit(f"invalid final artifact path: {name}")
    if sha256(path) != artifact.get("sha256"):
        raise SystemExit(f"final artifact SHA mismatch: {name}")
if value["operator_lock"]["sha256"] != value.get("operator_lock_sha256"):
    raise SystemExit("operator-lock duplicate binding mismatch")
if value["candidate_manifest"]["sha256"] != value.get("candidate_manifest_sha256"):
    raise SystemExit("candidate-manifest duplicate binding mismatch")
if value["candidate_tensor"]["sha256"] != value.get("candidate_tensor_sha256"):
    raise SystemExit("candidate-tensor duplicate binding mismatch")
print(value["operator_lock"]["path"])
print(value["candidate_manifest"]["path"])
print(value["candidate_tensor"]["path"])
print(value["model_manifest"]["path"])
PY
)
operator_lock_source=${sources[0]}
candidate_manifest_source=${sources[1]}
candidate_tensor_source=${sources[2]}
model_manifest=${sources[3]}

sha256sum \
  "$code_root/operator_final_job.sh" "$authorization" "$crossover_contract" \
  "$operator_contract" "$extension_contract" "$manifest" "$manifest_report" \
  "$design_audit" "$model_contract" "$model_verifier" "$tree_hasher" \
  "$code_root/scripts/benchmark_v1/"*.py "$code_root/scripts/benchmark_v2/"*.py \
  "$code_root/tests/test_benchmark_v1.py" "$code_root/tests/test_governance_crossover.py" \
  "$code_root/tests/test_operator_execution_plumbing.py" \
  >"$run_dir/executed_code.sha256"
cp "$authorization" "$crossover_contract" "$operator_contract" "$extension_contract" \
  "$manifest_report" "$design_audit" "$model_contract" "$run_dir/"
mkdir "$run_dir/source_candidate"
cp "$operator_lock_source" "$run_dir/QWEN3_8B_OPERATOR_MANIFEST_V1.json"
cp "$candidate_manifest_source" "$run_dir/source_candidate/"
cp "$candidate_tensor_source" "$run_dir/source_candidate/"
operator_lock=$run_dir/QWEN3_8B_OPERATOR_MANIFEST_V1.json
candidate_manifest=$run_dir/source_candidate/$(basename "$candidate_manifest_source")
candidate_tensor=$run_dir/source_candidate/$(basename "$candidate_tensor_source")

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  uname -a
  printf 'user=%s\njob_id=%s\nstage=operator_final_test\n' "$(id -un)" "$SLURM_JOB_ID"
  printf 'job_nodelist=%s\nvisible_devices=%s\n' \
    "${SLURM_JOB_NODELIST:-}" "${ASCEND_RT_VISIBLE_DEVICES:-}"
} >"$run_dir/environment.txt"
/usr/local/bin/npu-smi info >"$run_dir/npu-smi-before.txt" 2>&1
observed_base_sha=$("$python_bin" "$tree_hasher" "$runtime_root/$runtime_base")
observed_overlay_sha=$("$python_bin" "$tree_hasher" "$runtime_root/$runtime_overlay")
[[ "$observed_base_sha" == "$runtime_base_sha" ]] || { echo "runtime base tree mismatch" >&2; exit 3; }
[[ "$observed_overlay_sha" == "$runtime_overlay_sha" ]] || { echo "runtime overlay tree mismatch" >&2; exit 3; }
printf 'base_tree_sha256=%s\noverlay_tree_sha256=%s\n' \
  "$observed_base_sha" "$observed_overlay_sha" >"$run_dir/runtime_tree.sha256"
"$python_bin" "$model_verifier" --contract "$model_contract" --root "$shared_model" \
  --output "$run_dir/shared_model_verification.json"
"$python_bin" "$model_verifier" --contract "$model_contract" --root "$node_model" \
  --output "$run_dir/node_model_verification.json"

set +u
# shellcheck disable=SC1091
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONPATH="$code_root:$overlay_packages:$site_packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" -m unittest \
  "$code_root/tests/test_benchmark_v1.py" \
  "$code_root/tests/test_governance_crossover.py" \
  "$code_root/tests/test_operator_execution_plumbing.py" \
  >"$run_dir/tests.txt" 2>&1

rows=$run_dir/operator_final_test.jsonl
operator_environment=$run_dir/operator_final_test_environment.json
identity=$run_dir/operator_final_test_identity.json
operator_analysis=$run_dir/operator_final_test_analysis.json
"$python_bin" -m scripts.benchmark_v2.run_operator_candidate \
  --candidate-tensor "$candidate_tensor" --candidate-manifest "$candidate_manifest" \
  --execution-authorization "$authorization" --crossover-contract "$crossover_contract" \
  --manifest "$manifest" --manifest-report "$manifest_report" \
  --design-audit "$design_audit" --model-manifest "$model_manifest" \
  --model-path "$node_model" --evaluation-split final_test --operator-lock "$operator_lock" \
  --output "$rows" --environment-output "$operator_environment" \
  --identity-output "$identity" --device npu:0 --attn-implementation eager
"$python_bin" -m scripts.benchmark_v2.analyze_operator_candidate \
  --input "$rows" --environment "$operator_environment" --identity-report "$identity" \
  --output "$operator_analysis"

baseline_rows=$run_dir/governance_crossover_final_test.jsonl
baseline_environment=$run_dir/governance_crossover_final_test_environment.json
baseline_analysis=$run_dir/governance_crossover_final_test_analysis.json
"$python_bin" -m scripts.benchmark_v2.extract_crossover_from_operator_run \
  --input "$rows" --operator-environment "$operator_environment" \
  --output "$baseline_rows" --environment-output "$baseline_environment"
"$python_bin" -m scripts.benchmark_v2.analyze_crossover \
  --input "$baseline_rows" --environment "$baseline_environment" \
  --contract "$crossover_contract" --output "$baseline_analysis"

"$python_bin" - "$authorization" "$operator_lock" "$candidate_manifest" \
  "$candidate_tensor" "$operator_analysis" "$baseline_analysis" <<'PY'
import hashlib, json, pathlib, sys
authorization, lock, manifest, tensor, operator_analysis, baseline_analysis = map(pathlib.Path, sys.argv[1:])
def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()
a=json.loads(authorization.read_text())
o=json.loads(operator_analysis.read_text())
b=json.loads(baseline_analysis.read_text())
i=json.loads((operator_analysis.parent/"operator_final_test_identity.json").read_text())
if o.get("audit",{}).get("success") is not True or o["audit"].get("row_count") != 6144:
    raise SystemExit("operator final-test audit failed")
if b.get("audit",{}).get("success") is not True or b["audit"].get("row_count") != 6144:
    raise SystemExit("crossover final-test baseline audit failed")
if i.get("success") is not True or float(i.get("max_error",1.0)) != 0.0:
    raise SystemExit("operator final-test identity audit failed")
for name, analysis in (("operator",o),("baseline",b)):
    audit=analysis.get("audit",{})
    if audit.get("unique_job_keys") != 6144:
        raise SystemExit(f"{name} final-test unique-key mismatch")
    if audit.get("observed_key_sha256") != a.get("expected_key_sha256"):
        raise SystemExit(f"{name} final-test key hash mismatch")
    if audit.get("expected_key_sha256") != a.get("expected_key_sha256"):
        raise SystemExit(f"{name} final-test expected-key binding mismatch")
    if audit.get("nonfinite_job_keys"):
        raise SystemExit(f"{name} final-test contains non-finite rows")
if o.get("final_test_open") is not True or b.get("audit",{}).get("success") is not True:
    raise SystemExit("final-test stage metadata mismatch")
summary={
    "schema_version":1,
    "stage":"operator_final_test",
    "candidate_id":a["candidate_id"],
    "rows":6144,
    "expected_key_sha256":a["expected_key_sha256"],
    "operator_lock_sha256":sha256(lock),
    "candidate_manifest_sha256":sha256(manifest),
    "candidate_tensor_sha256":sha256(tensor),
    "operator_analysis_sha256":sha256(operator_analysis),
    "crossover_analysis_sha256":sha256(baseline_analysis),
    "final_test_open":True,
    "final_test_open_count":1,
    "production_rollout_approved":False,
}
(operator_analysis.parent/"final_test_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
PY
/usr/local/bin/npu-smi info >"$run_dir/npu-smi-after.txt" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >"$run_dir/COMPLETE"
