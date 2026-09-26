#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?V4 protected-controls run id required}
[[ "$run_id" =~ ^qwen3-8b-adsge-v4-protected-controls-diagnostic-[0-9]{8}T[0-9]{6}Z$ ]] || {
  echo "invalid V4 protected-controls run id" >&2; exit 2;
}
ssh_target=${CLUSTER_SSH_TARGET:-archive-host}
tailscale_container=${CLUSTER_TAILSCALE_CONTAINER:-tunnel-container}
cluster_host=${CLUSTER_CLUSTER_HOST:-compute-cluster}

remote_script=$(command cat <<'REMOTE_SCRIPT'
set -euo pipefail
run_id=$1
home_root=/workspace/context-mismatch-qwen3-8b
work_root=/workspace/context-mismatch-qwen3-8b
run_dir=$work_root/runs/$run_id
archive_dir=$home_root/evidence
archive_final=$archive_dir/$run_id.tar.gz
archive_incoming=$archive_dir/.$run_id.tar.gz.incoming
staging=$home_root/.archive-$run_id.incoming
slurm_root=/service/slurm/24.11.5/bin
[[ $(id -un) == researcher ]] || { echo "refusing unexpected user" >&2; exit 2; }
for required in "$run_dir/COMPLETE" "$run_dir/exit_status.json" \
  "$run_dir/execution_authorization.json" "$run_dir/executed_code.sha256" \
  "$run_dir/tests.txt" "$run_dir/v4_protected_controls.jsonl" \
  "$run_dir/v4_protected_controls.environment.json" \
  "$run_dir/v4_protected_controls.analysis.json"; do
  [[ -r "$required" ]] || { echo "terminal artifact missing: $required" >&2; exit 2; }
done
[[ ! -e "$archive_final" && ! -e "$archive_incoming" && ! -e "$staging" ]] || {
  echo "refusing existing archive or staging target" >&2; exit 2;
}
(cd / && sha256sum -c "$run_dir/executed_code.sha256")
readarray -t audit_values < <(python3 - "$run_dir" <<'PY'
import json,math,pathlib,sys
root=pathlib.Path(sys.argv[1])
status=json.loads((root/'exit_status.json').read_text())
auth=json.loads((root/'execution_authorization.json').read_text())
analysis=json.loads((root/'v4_protected_controls.analysis.json').read_text())
environment=json.loads((root/'v4_protected_controls.environment.json').read_text())
tests=(root/'tests.txt').read_text()
if (root/'COMPLETE').read_text().strip()!='COMPLETE': raise SystemExit('invalid COMPLETE marker')
if 'Ran 33 tests' not in tests or not tests.rstrip().endswith('OK') or 'skipped=' in tests:
 raise SystemExit('V4 protected-controls Torch tests failed or skipped')
for field,expected in {'stage':'governance_failed_fit_protected_controls','exit_code':0,
 'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False}.items():
 if status.get(field)!=expected: raise SystemExit(f'exit status mismatch: {field}')
for field,expected in {'stage':'governance_failed_fit_protected_controls','execution_allowed':True,
 'expected_rows':2856,
 'expected_key_sha256':'220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77',
 'forced_directions':['positive','negative'],'post_failure_characterization':True,
 'fit_gates_passed':False,'candidate_eligible':False,'candidate_may_be_locked':False,
 'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False}.items():
 if auth.get(field)!=expected: raise SystemExit(f'authorization mismatch: {field}')
audit=analysis.get('audit',{})
for field,expected in {'success':True,'row_count':2856,'unique_case_keys':2856,
 'observed_key_sha256':'220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77'}.items():
 if audit.get(field)!=expected: raise SystemExit(f'analysis audit mismatch: {field}')
if audit.get('nonfinite_case_keys'): raise SystemExit('non-finite V4 protected-control keys')
for field,expected in {'method':'ADSGE-V4','post_failure_characterization':True,
 'fit_gates_passed':False,'controls_diagnostic_complete':True,'candidate_eligible':False,
 'candidate_may_be_locked':False,'final_test_open':False,'final_test_open_count':0,
 'production_rollout_approved':False}.items():
 if analysis.get(field)!=expected: raise SystemExit(f'analysis mismatch: {field}')
if analysis.get('external_zero_gate_identity',{}).get('exact') is not True:
 raise SystemExit('V4 external zero-gate identity failed')
families={'fresh_verification','matched_verification','matched_delegated_choice',
 'explicit_governance_reset','supported_user_authority','factual_boundary_memory'}
observed=analysis.get('family_reports',{})
if set(observed)!=families: raise SystemExit('V4 control family coverage incomplete')
for family,value in observed.items():
 if value.get('rows',0)<=0: raise SystemExit(f'empty family: {family}')
 if set(value.get('forced_on',{}))!={'positive','negative'}:
  raise SystemExit(f'forced directions incomplete: {family}')
 for direction in ('positive','negative'):
  forced=value['forced_on'][direction]
  if forced.get('rows')!=value.get('rows'):
   raise SystemExit(f'forced row mismatch: {family}/{direction}')
comparison=analysis.get('comparison_to_v3_same_identity',{})
if comparison.get('same_identity_key_sha256') is not True:
 raise SystemExit('V3/V4 control key mismatch')
if environment.get('fit_gates_passed') is not False or environment.get('expected_rows')!=2856:
 raise SystemExit('environment fit/row boundary mismatch')
def finite(item):
 if isinstance(item,float): return math.isfinite(item)
 if isinstance(item,dict): return all(finite(v) for v in item.values())
 if isinstance(item,list): return all(finite(v) for v in item)
 return True
if not finite(analysis): raise SystemExit('analysis contains non-finite metrics')
for value in (auth,status,analysis,environment):
 if value.get('final_test_open') is not False or value.get('final_test_open_count',0)!=0:
  raise SystemExit('final-test boundary changed')
 if value.get('production_rollout_approved') is not False:
  raise SystemExit('production boundary changed')
job_id=status.get('job_id')
if not isinstance(job_id,str) or not job_id.isdigit(): raise SystemExit('invalid job id')
node=auth.get('execution_node','')
if not (len(node)==3 and node[0]=='a' and node[1:].isdigit()): raise SystemExit('invalid execution node')
print(job_id); print(node)
PY
)
job_id=${audit_values[0]}
execution_node=${audit_values[1]}
slurm_record=$("$slurm_root/scontrol" show job -o "$job_id")
for required in "JobId=$job_id" "UserId=researcher(" "JobState=COMPLETED" "ExitCode=0:0" \
  "Partition=a01" "NodeList=$execution_node" "NumCPUs=8" "TresPerNode=gres/npu:910B3:1"; do
  [[ "$slurm_record" == *"$required"* ]] || { echo "terminal record missing: $required" >&2; exit 3; }
done

mkdir -p "$archive_dir" "$staging/runs"
cp -a "$run_dir" "$staging/runs/"
copied=$staging/runs/$run_id
printf '%s\n' "$slurm_record" >"$staging/SLURM_TERMINAL_RECORD.txt"
printf '{"schema_version":1,"run_id":"%s","job_id":%s,"stage":"governance_failed_fit_protected_controls","method":"ADSGE-V4","rows":2856,"post_failure_characterization":true,"fit_gates_passed":false,"candidate_eligible":false,"candidate_may_be_locked":false,"scientific_audit_complete":true,"slurm_completed_verified":true,"final_test_open":false,"final_test_open_count":0,"production_rollout_approved":false}\n' \
  "$run_id" "$job_id" >"$staging/TRANSFER_STATUS.json"
(
  cd "$copied"
  find . -type f ! -name artifact_tree.sha256 -print | LC_ALL=C sort | xargs sha256sum >artifact_tree.sha256
  sha256sum -c artifact_tree.sha256
)
COPYFILE_DISABLE=1 tar --no-xattrs -C "$staging" -czf "$archive_incoming" .
archive_sha=$(sha256sum "$archive_incoming" | awk '{print $1}')
mv "$archive_incoming" "$archive_final"
rm -rf -- "$staging"
printf 'archive_name=%s\narchive_sha256=%s\narchive_path=%s\n' \
  "$run_id.tar.gz" "$archive_sha" "$archive_final"
REMOTE_SCRIPT
)
printf -v cluster_command '%q ' bash -c "$remote_script" -- "$run_id"
printf -v outer_command '%q ' docker exec -i "$tailscale_container" ssh "$cluster_host" "$cluster_command"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" "$outer_command"
