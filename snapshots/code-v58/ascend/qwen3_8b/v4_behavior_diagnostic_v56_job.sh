#!/usr/bin/env bash
set -euo pipefail

expected_user=researcher
home_root=/workspace/context-mismatch-qwen3-8b
work_root=/workspace/context-mismatch-qwen3-8b
code_root=${1:?code root required}
run_dir=${2:?run directory required}
authorization=${3:?authorization required}
shared_model=$home_root/models/Qwen3-8B
node_model=/workspace/node-local/context-mismatch-qwen3-8b/models/Qwen3-8B
runtime_root=/workspace/node-local/context-mismatch-ascend/runtime-v1
runtime_base=ascend-py311-torch271-cann83rc1-bullseye
runtime_overlay=cann9-py311-overlay-v2
runtime_base_sha=af7b5881a1f73c30f710661ab4934559befbf45c5f22d7ae65e44531d8c98555
runtime_overlay_sha=9e0ce3ebe2054e56b1bb28ce8f24448a794019aef991ec44752c5d2f0a479bd9
python_bin=$runtime_root/python-3.11.13-standalone/python/bin/python3
site_packages=$runtime_root/$runtime_base/rootfs/opt/distill/lib/python3.11/site-packages
overlay_packages=$runtime_root/$runtime_overlay
editor_contract=$code_root/protocol/QWEN3_8B_ABSTAINING_DIRECTIONAL_GOVERNANCE_EDITOR_V4.json
diagnostic_contract=$code_root/protocol/QWEN3_8B_ADSGE_V4_POST_FIT_BEHAVIOR_DIAGNOSTIC_V1.json
crossover_contract=$code_root/protocol/GOVERNANCE_TASK_CROSSOVER_V1.json
site_manifest=$code_root/artifacts/QWEN3_8B_OPERATOR_SITE_MANIFEST_V1.json
manifest=$code_root/artifacts/benchmark_manifest.jsonl
manifest_report=$code_root/artifacts/benchmark_manifest.report.json
design_audit=$code_root/artifacts/governance_crossover_operator_dev_design_audit_v2.json
v3_analysis=$code_root/artifacts/v3_directional_behavior_operator_dev.analysis.json
model_contract=$code_root/model_source_contract.json
model_verifier=$code_root/verify_model_source.py
tree_hasher=$code_root/hash_artifact_tree.py

[[ $(id -un) == "$expected_user" ]] || { echo "refusing unexpected user" >&2; exit 2; }
[[ "$code_root" == "$work_root/code-v57" ]] || { echo "V4 diagnostic requires code-v57" >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} ]] || { echo "refusing non-Slurm execution" >&2; exit 2; }
[[ ${SLURM_CPUS_PER_TASK:-} == 8 ]] || { echo "V4 diagnostic requires 8 CPUs" >&2; exit 2; }
[[ "$run_dir" =~ ^$work_root/runs/qwen3-8b-adsge-v4-behavior-diagnostic-[0-9]{8}T[0-9]{6}Z$ ]] || {
  echo "invalid V4 diagnostic run directory" >&2; exit 2;
}
for required in "$authorization" "$editor_contract" "$diagnostic_contract" \
  "$crossover_contract" "$site_manifest" "$manifest" "$manifest_report" \
  "$design_audit" "$v3_analysis" "$model_contract" "$model_verifier" "$tree_hasher" \
  "$code_root/bundle.sha256"; do
  [[ -r "$required" ]] || { echo "missing prerequisite: $required" >&2; exit 2; }
done

umask 077
status_tmp=$run_dir/.exit_status.json.tmp-$$
record_exit() {
  rc=$?
  printf '{"job_id":"%s","exit_code":%s,"hostname":"%s","stage":"governance_failed_fit_selection","final_test_open":false,"final_test_open_count":0,"production_rollout_approved":false}\n' \
    "$SLURM_JOB_ID" "$rc" "$(hostname)" >"$status_tmp"
  mv -f -- "$status_tmp" "$run_dir/exit_status.json"
}
trap record_exit EXIT

readarray -t source_values < <(python3 - "$authorization" "$code_root" \
  "$editor_contract" "$diagnostic_contract" "$crossover_contract" "$site_manifest" \
  "$manifest" "$design_audit" <<'PY'
import hashlib,json,os,sys
auth_path,code_root,editor,diagnostic,crossover,sites,manifest,design=sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
a=json.load(open(auth_path))
required={
 'stage':'governance_failed_fit_selection','code_root':code_root,
 'immutable_code_bundle_manifest_sha256':sha(os.path.join(code_root,'bundle.sha256')),
 'execution_allowed':True,'editor_contract_sha256':sha(editor),
 'diagnostic_contract_sha256':sha(diagnostic),'crossover_contract_sha256':sha(crossover),
 'operator_site_manifest_sha256':sha(sites),'benchmark_manifest_sha256':sha(manifest),
 'design_audit_sha256':sha(design),'expected_rows':3072,
 'expected_key_sha256':'74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e',
 'post_failure_characterization':True,'fit_gates_passed':False,
 'candidate_eligible':False,'candidate_may_be_locked':False,
 'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False,
}
for key,value in required.items():
    if a.get(key)!=value: raise SystemExit(f'authorization mismatch: {key}')
node=a.get('execution_node')
contract=a.get('resource_contract',{})
expected={'partition':'a01','nodes':1,'ntasks':1,'cpus_per_task':8,'mem_mib':131072,
          'npu_type':'910B3','npus':1,'time_limit':'08:00:00','node':node}
if contract!=expected: raise SystemExit('resource contract mismatch')
for name in ('node_selection_snapshot','checkpoint','fit_report','v3_operator_dev_analysis','model_manifest'):
    source=a.get(name,{})
    if sha(source.get('path',''))!=source.get('sha256'):
        raise SystemExit(f'{name} source mismatch')
print(a['checkpoint']['path']); print(a['fit_report']['path'])
print(a['model_manifest']['path']); print(node); print(a['node_selection_snapshot']['path'])
PY
)
checkpoint=${source_values[0]}
fit_report=${source_values[1]}
model_manifest=${source_values[2]}
expected_node=${source_values[3]}
selection_snapshot=${source_values[4]}
[[ $(hostname) == "$expected_node" && ${SLURM_JOB_NODELIST:-} == "$expected_node" ]] || {
  echo "allocation differs from authorization node" >&2; exit 2;
}

[[ -x "$python_bin" ]] || { echo "missing runtime prerequisite: $python_bin" >&2; exit 2; }
for required_dir in "$site_packages" "$overlay_packages" "$shared_model" "$run_dir"; do
  [[ -d "$required_dir" ]] || { echo "missing directory: $required_dir" >&2; exit 2; }
done

sha256sum "$code_root/v4_behavior_diagnostic_v56_job.sh" "$authorization" \
  "$selection_snapshot" "$editor_contract" "$diagnostic_contract" "$crossover_contract" \
  "$site_manifest" "$manifest" "$manifest_report" "$design_audit" \
  "$code_root/scripts/benchmark_v1/"*.py "$code_root/scripts/benchmark_v2/"*.py \
  "$code_root/scripts/benchmark_v3/"*.py "$code_root/scripts/benchmark_v4/"*.py \
  "$code_root/scripts/benchmark_v5/"*.py "$code_root/scripts/benchmark_v8/"*.py \
  "$code_root/tests/test_abstaining_directional_governance_v4.py" \
  "$code_root/tests/test_v4_behavior_diagnostic.py" >"$run_dir/executed_code.sha256"
cp "$authorization" "$run_dir/execution_authorization.json"
cp "$selection_snapshot" "$editor_contract" "$diagnostic_contract" "$crossover_contract" \
  "$site_manifest" "$manifest_report" "$design_audit" "$v3_analysis" "$run_dir/"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  uname -a
  printf 'user=%s\njob_id=%s\nstage=governance_failed_fit_selection\n' \
    "$(id -un)" "$SLURM_JOB_ID"
  printf 'authorized_node=%s\njob_nodelist=%s\nvisible_devices=%s\n' \
    "$expected_node" "${SLURM_JOB_NODELIST:-}" "${ASCEND_RT_VISIBLE_DEVICES:-}"
} >"$run_dir/environment.txt"

set +u
# shellcheck disable=SC1091
source /usr/local/Ascend/cann/set_env.sh
set -u
export PYTHONPATH="$code_root:$overlay_packages:$site_packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
(
  cd "$code_root"
  "$python_bin" -m unittest \
    tests.test_directional_governance_v3 \
    tests.test_abstaining_directional_governance_v4 \
    tests.test_v4_behavior_diagnostic
) >"$run_dir/tests.txt" 2>&1

/usr/local/bin/npu-smi info >"$run_dir/npu-smi-before.txt" 2>&1
observed_base_sha=$("$python_bin" "$tree_hasher" "$runtime_root/$runtime_base")
observed_overlay_sha=$("$python_bin" "$tree_hasher" "$runtime_root/$runtime_overlay")
[[ "$observed_base_sha" == "$runtime_base_sha" ]] || { echo "runtime base mismatch" >&2; exit 3; }
[[ "$observed_overlay_sha" == "$runtime_overlay_sha" ]] || { echo "runtime overlay mismatch" >&2; exit 3; }
"$python_bin" "$model_verifier" --contract "$model_contract" --root "$shared_model" \
  --output "$run_dir/shared_model_verification.json"

if [[ ! -d "$node_model" ]]; then
  model_parent=$(dirname "$node_model")
  mkdir -p "$model_parent"
  exec 9>"$model_parent/.qwen3-8b-stage.lock"
  flock -x 9
  if [[ ! -d "$node_model" ]]; then
    model_incoming=$model_parent/.Qwen3-8B.incoming-$SLURM_JOB_ID
    [[ ! -e "$model_incoming" ]] || { echo "model staging target exists" >&2; exit 3; }
    cp -a "$shared_model" "$model_incoming"
    "$python_bin" "$model_verifier" --contract "$model_contract" --root "$model_incoming" \
      --output "$run_dir/staged_model_verification.json"
    mv "$model_incoming" "$node_model"
  fi
  flock -u 9
fi
"$python_bin" "$model_verifier" --contract "$model_contract" --root "$node_model" \
  --output "$run_dir/node_model_verification.json"

rows=$run_dir/v4_behavior_operator_dev.jsonl
environment=$run_dir/v4_behavior_operator_dev.environment.json
identity=$run_dir/v4_behavior_operator_dev.identity.json
analysis=$run_dir/v4_behavior_operator_dev.analysis.json
"$python_bin" -m scripts.benchmark_v3.run_governance_editor \
  --checkpoint "$checkpoint" --fit-report "$fit_report" \
  --editor-contract "$editor_contract" --diagnostic-contract "$diagnostic_contract" \
  --execution-authorization "$authorization" --crossover-contract "$crossover_contract" \
  --manifest "$manifest" --manifest-report "$manifest_report" \
  --design-audit "$design_audit" --model-manifest "$model_manifest" \
  --model-path "$node_model" --evaluation-split failed_fit_selection \
  --output "$rows" --environment-output "$environment" --identity-output "$identity" \
  --device npu:0 --attn-implementation eager
"$python_bin" -m scripts.benchmark_v8.analyze_v4_behavior_diagnostic \
  --input "$rows" --environment "$environment" --identity-report "$identity" \
  --editor-contract "$editor_contract" --diagnostic-contract "$diagnostic_contract" \
  --fit-report "$fit_report" --v3-analysis "$v3_analysis" \
  --output "$analysis" --bootstrap-replicates 10000

"$python_bin" - "$run_dir" "$authorization" <<'PY'
import json,math,pathlib,sys
root=pathlib.Path(sys.argv[1]); auth=json.load(open(sys.argv[2]))
status=json.loads((root/'v4_behavior_operator_dev.analysis.json').read_text())
audit=status.get('audit',{})
if audit.get('success') is not True or audit.get('row_count')!=3072 or audit.get('unique_job_keys')!=3072:
    raise SystemExit('V4 behavior audit failed')
if audit.get('observed_key_sha256')!=auth.get('expected_key_sha256') or audit.get('nonfinite_job_keys'):
    raise SystemExit('V4 behavior key/finite audit failed')
if audit.get('zero_gate_max_error')!=0.0: raise SystemExit('zero-gate identity failed')
if status.get('method')!='ADSGE-V4' or status.get('post_failure_characterization') is not True:
    raise SystemExit('V4 diagnostic identity mismatch')
if status.get('fit_gates_passed') is not False or status.get('candidate_eligible') is not False:
    raise SystemExit('V4 failed fit gate changed')
if status.get('candidate_may_be_locked') is not False: raise SystemExit('V4 lock boundary changed')
if set(status.get('application_routing',{}))!={'matched','mismatched'}:
    raise SystemExit('V4 routing summary incomplete')
comparison=status.get('comparison_to_v3_same_identity',{})
if set(comparison.get('gap_reduction_by_benchmark',{}))!={'arc_challenge','bbh','gsm8k','math_500','mmlu_pro','musr'}:
    raise SystemExit('V3/V4 benchmark comparison incomplete')
def finite(value):
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,dict): return all(finite(v) for v in value.values())
    if isinstance(value,list): return all(finite(v) for v in value)
    return True
if not finite(status): raise SystemExit('non-finite V4 terminal metrics')
for value in (auth,status):
    if value.get('final_test_open') is not False or value.get('final_test_open_count')!=0:
        raise SystemExit('final-test boundary changed')
    if value.get('production_rollout_approved') is not False:
        raise SystemExit('production boundary changed')
PY

/usr/local/bin/npu-smi info >"$run_dir/npu-smi-after.txt" 2>&1
printf 'COMPLETE\n' >"$run_dir/COMPLETE"
