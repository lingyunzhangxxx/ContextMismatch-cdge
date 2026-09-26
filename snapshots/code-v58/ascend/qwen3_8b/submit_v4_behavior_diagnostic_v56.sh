#!/usr/bin/env bash
set -euo pipefail

slurm_root=/service/slurm/24.11.5/bin
account_user=researcher
home_root=/workspace/context-mismatch-qwen3-8b
work_root=/workspace/context-mismatch-qwen3-8b
queue_limit=7
code_root=${1:?code root required}
authorization=${2:?authorization path required}
timestamp=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
target_node=${CLUSTER_TARGET_NODE:?CLUSTER_TARGET_NODE is required}
job_script=$code_root/v4_behavior_diagnostic_v56_job.sh
run_id=qwen3-8b-adsge-v4-behavior-diagnostic-$timestamp
job_name=shang-cm-$run_id
run_dir=$work_root/runs/$run_id

[[ $(id -un) == "$account_user" ]] || { echo "refusing unexpected user" >&2; exit 2; }
[[ "$code_root" == "$work_root/code-v57" ]] || { echo "submitter requires code-v57" >&2; exit 2; }
[[ "$target_node" =~ ^a[0-9]{2}$ ]] || { echo "invalid target node" >&2; exit 2; }
[[ "$authorization" =~ ^$home_root/authorizations/[A-Za-z0-9._-]+\.json$ ]] || {
  echo "invalid authorization path" >&2; exit 2;
}
[[ "$timestamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid timestamp" >&2; exit 2; }
for required in "$job_script" "$authorization" "$code_root/bundle.sha256"; do
  [[ -r "$required" ]] || { echo "missing prerequisite: $required" >&2; exit 2; }
done
python3 - "$authorization" "$code_root" "$target_node" <<'PY'
import hashlib,json,os,sys
path,code_root,target=sys.argv[1:]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
a=json.load(open(path))
required={
 'stage':'governance_failed_fit_selection','code_root':code_root,
 'immutable_code_bundle_manifest_sha256':sha(os.path.join(code_root,'bundle.sha256')),
 'execution_node':target,'slurm_partition':'a01','execution_allowed':True,
 'expected_rows':3072,
 'expected_key_sha256':'74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e',
 'post_failure_characterization':True,'fit_gates_passed':False,
 'candidate_eligible':False,'candidate_may_be_locked':False,
 'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False,
}
for key,value in required.items():
    if a.get(key)!=value: raise SystemExit(f'authorization mismatch: {key}')
expected={'partition':'a01','nodes':1,'ntasks':1,'cpus_per_task':8,'mem_mib':131072,
          'npu_type':'910B3','npus':1,'time_limit':'08:00:00','node':target}
if a.get('resource_contract')!=expected: raise SystemExit('resource contract mismatch')
PY
[[ ! -e "$run_dir" ]] || { echo "stale run directory exists" >&2; exit 2; }
queue_count=$("$slurm_root/squeue" -h -u "$account_user" -o '%i' | awk 'NF {n++} END {print n+0}')
(( queue_count < queue_limit )) || {
  echo "refusing: queue already has $queue_count jobs (limit $queue_limit)" >&2; exit 2;
}
if "$slurm_root/squeue" -h -u "$account_user" -n "$job_name" -o '%i' | grep -q .; then
  echo "refusing duplicate job name" >&2; exit 2
fi

mkdir -p "$work_root/runs"
mkdir "$run_dir"
printf '{"run_id":"%s","stage":"governance_failed_fit_selection","target_node":"%s","queue_before":%s,"final_test_open":false,"final_test_open_count":0,"production_rollout_approved":false}\n' \
  "$run_id" "$target_node" "$queue_count" >"$run_dir/submission.json"
submission=$("$slurm_root/sbatch" --parsable --hold --job-name="$job_name" \
  --partition=a01 --nodelist="$target_node" --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=128G --gres=npu:910B3:1 --time=08:00:00 \
  --output="$run_dir/slurm-%j.out" --error="$run_dir/slurm-%j.out" \
  "$job_script" "$code_root" "$run_dir" "$authorization")
job_id=${submission%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response" >&2; exit 3; }
held_record=$("$slurm_root/scontrol" show job -o "$job_id")
cancel_new_held_job() {
  current=$("$slurm_root/scontrol" show job -o "$job_id" 2>/dev/null || true)
  if [[ "$current" == *"UserId=$account_user("* && "$current" == *"JobState=PENDING"* \
     && "$current" == *"Reason=JobHeldUser"* && "$current" == *"RunTime=00:00:00"* ]]; then
    "$slurm_root/scancel" "$job_id"
  else
    echo "refusing automatic cancel: job is no longer safely held" >&2
  fi
}
failed=0
for expected in "JobName=$job_name" "UserId=$account_user(" "JobState=PENDING" \
  "Reason=JobHeldUser" "RunTime=00:00:00" "Partition=a01" "ReqNodeList=$target_node" \
  "NumNodes=1" "NumCPUs=8" "NumTasks=1" "CPUs/Task=8" "MinMemoryNode=128G" \
  "TimeLimit=08:00:00" "Command=$job_script" "TresPerNode=gres/npu:910B3:1"; do
  [[ "$held_record" == *"$expected"* ]] || { echo "held validation missing: $expected" >&2; failed=1; }
done
if (( failed )); then cancel_new_held_job; exit 4; fi
held_count=$("$slurm_root/squeue" -h -u "$account_user" -o '%i' | awk 'NF {n++} END {print n+0}')
if (( held_count > queue_limit )); then cancel_new_held_job; exit 4; fi
printf '%s\n' "$held_record" >"$run_dir/held_job_record.txt"
"$slurm_root/scontrol" release "$job_id"
post_count=$("$slurm_root/squeue" -h -u "$account_user" -o '%i' | awk 'NF {n++} END {print n+0}')
(( post_count <= queue_limit )) || { echo "queue limit exceeded after release" >&2; exit 5; }
printf '{"job_id":%s,"run_id":"%s","run_dir":"%s","target_node":"%s","queue_before":%s,"queue_after":%s}\n' \
  "$job_id" "$run_id" "$run_dir" "$target_node" "$queue_count" "$post_count"
