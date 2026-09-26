#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id required}
[[ "$run_id" =~ ^qwen3-8b-operator-(subspace-capture|protected-capture|fit-prefilter|behavior-screening|behavior-selection|controls-selection|final-test)-[0-9]{8}T[0-9]{6}Z$ ]] || {
  echo "invalid operator run id" >&2
  exit 2
}
ssh_target=${CLUSTER_SSH_TARGET:?set CLUSTER_SSH_TARGET}
tailscale_container=${CLUSTER_TUNNEL_CONTAINER:?set CLUSTER_TUNNEL_CONTAINER}
cluster_host=${CLUSTER_HOST:?set CLUSTER_HOST}

remote_script='
set -euo pipefail
run_id=$1
project_root=/workspace/context-mismatch-qwen3-8b
run_dir=$project_root/runs/$run_id
archive_name=$run_id.tar.gz
archive_dir=$project_root/evidence
archive_final=$archive_dir/$archive_name
archive_incoming=$archive_dir/.$archive_name.incoming
staging=$project_root/.archive-$run_id.incoming
slurm_root=/service/slurm/24.11.5/bin

[[ $(id -un) == researcher ]] || { echo "refusing unexpected user" >&2; exit 2; }
[[ -d "$run_dir" ]] || { echo "run directory is missing" >&2; exit 2; }
[[ -f "$run_dir/COMPLETE" && -f "$run_dir/exit_status.json" ]] || {
  echo "operator terminal markers are missing" >&2; exit 2;
}
[[ ! -e "$archive_final" && ! -e "$archive_incoming" ]] || {
  echo "refusing existing operator archive target" >&2; exit 2;
}
[[ ! -e "$staging" ]] || { echo "preserving interrupted staging directory" >&2; exit 2; }

readarray -t gate_values < <(python3 - "$run_dir" "$run_id" <<"PY"
import hashlib, json, math, pathlib, re, sys
root=pathlib.Path(sys.argv[1]); run_id=sys.argv[2]
match=re.fullmatch(r"qwen3-8b-operator-(subspace-capture|protected-capture|fit-prefilter|behavior-screening|behavior-selection|controls-selection|final-test)-[0-9]{8}T[0-9]{6}Z",run_id)
if not match: raise SystemExit("invalid run id")
stage=match.group(1)
status=json.loads((root/"exit_status.json").read_text())
if status.get("exit_code") != 0: raise SystemExit("job-local exit code is not zero")
if status.get("production_rollout_approved") is not False: raise SystemExit("production boundary changed")
summary={"stage":stage}
if stage in {"subspace-capture","protected-capture","fit-prefilter"}:
    value=json.loads((root/"pipeline_summary.json").read_text())
    expected_stage=stage.replace("-","_")
    if value.get("stage") != expected_stage: raise SystemExit("pipeline stage mismatch")
    if value.get("final_test_open") is not False or value.get("production_rollout_approved") is not False:
        raise SystemExit("pipeline safety boundary mismatch")
    if stage == "subspace-capture":
        if value.get("rows") != 3072 or value.get("shards",0) < 1: raise SystemExit("subspace capture summary mismatch")
        capture=json.loads((root/"subspace_capture/capture_manifest.json").read_text())
        if capture.get("rows") != 3072 or capture.get("boundary_rows") != 768: raise SystemExit("subspace capture count mismatch")
        summary.update(rows=3072,boundary_rows=768,candidates=0)
    elif stage == "protected-capture":
        if value.get("rows") != 4008 or value.get("shards",0) < 1: raise SystemExit("protected capture summary mismatch")
        capture=json.loads((root/"protected_capture/capture_manifest.json").read_text())
        if capture.get("rows") != 4008: raise SystemExit("protected capture count mismatch")
        summary.update(rows=4008,boundary_rows=0,candidates=0)
    else:
        required={"grid_rows":15660,"shortlist_rows":42,"materialized_candidates":42,"all_scores_finite":True}
        for field, expected in required.items():
            if value.get(field) != expected: raise SystemExit(f"fit summary mismatch: {field}")
        summary.update(rows=15660,boundary_rows=0,candidates=42)
elif stage in {"behavior-screening","behavior-selection","controls-selection"}:
    mode=stage.replace("-","_")
    ledger=json.loads((root/"batch_ledger.json").read_text())
    if ledger.get("mode") != mode or ledger.get("candidate_count") != len(ledger.get("candidates",[])):
        raise SystemExit("candidate batch ledger mismatch")
    if not ledger.get("candidates"): raise SystemExit("candidate batch is empty")
    expected_rows=2088 if stage == "controls-selection" else 3072
    for row in ledger["candidates"]:
        candidate_id=row.get("candidate_id")
        analysis=json.loads((root/row["analysis"]).read_text())
        audit=analysis.get("audit",{})
        if audit.get("success") is not True or audit.get("row_count") != expected_rows:
            raise SystemExit("candidate batch audit failed: "+str(candidate_id))
        identity=json.loads((root/row["identity_report"]).read_text())
        if identity.get("success") is not True or float(identity.get("max_error",1.0)) != 0.0:
            raise SystemExit("candidate identity failed: "+str(candidate_id))
    summary.update(rows=expected_rows*ledger["candidate_count"],boundary_rows=0,candidates=ledger["candidate_count"])
else:
    value=json.loads((root/"final_test_summary.json").read_text())
    if value.get("stage") != "operator_final_test" or value.get("rows") != 6144:
        raise SystemExit("final-test summary mismatch")
    if value.get("final_test_open") is not True or value.get("final_test_open_count") != 1:
        raise SystemExit("final-test opening metadata mismatch")
    if value.get("production_rollout_approved") is not False:
        raise SystemExit("production boundary changed")
    authorization=json.loads(next(root.glob("*operator-final-test-code-v*.json")).read_text())
    operator=json.loads((root/"operator_final_test_analysis.json").read_text())
    baseline=json.loads((root/"governance_crossover_final_test_analysis.json").read_text())
    expected_key_sha=authorization.get("expected_key_sha256")
    for name, analysis in (("operator",operator),("baseline",baseline)):
        audit=analysis.get("audit",{})
        if audit.get("success") is not True or audit.get("row_count") != 6144:
            raise SystemExit(f"{name} final-test audit failed")
        if audit.get("unique_job_keys") != 6144:
            raise SystemExit(f"{name} final-test unique-key mismatch")
        if audit.get("observed_key_sha256") != expected_key_sha:
            raise SystemExit(f"{name} final-test key hash mismatch")
        if audit.get("expected_key_sha256") != expected_key_sha:
            raise SystemExit(f"{name} final-test expected-key binding mismatch")
        if audit.get("nonfinite_job_keys"):
            raise SystemExit(f"{name} final-test contains non-finite rows")
    if value.get("expected_key_sha256") != expected_key_sha:
        raise SystemExit("final-test summary key binding mismatch")
    summary.update(rows=6144,boundary_rows=0,candidates=1)
job_id=status.get("job_id")
if not isinstance(job_id,str) or not job_id.isdigit(): raise SystemExit("invalid job id")
print(job_id); print(stage); print(summary["rows"]); print(summary["boundary_rows"]); print(summary["candidates"])
PY
)
job_id=${gate_values[0]}
stage=${gate_values[1]}
rows=${gate_values[2]}
boundary_rows=${gate_values[3]}
candidates=${gate_values[4]}
slurm_source="scontrol live controller record"
if slurm_record=$("$slurm_root/scontrol" show job -o "$job_id" 2>/dev/null); then
  for required_text in \
    "JobId=$job_id" "UserId=researcher(" "JobState=COMPLETED" "ExitCode=0:0" \
    "RunTime=" "Partition=a01" "NodeList=a07"; do
    [[ "$slurm_record" == *"$required_text"* ]] || {
      echo "terminal Slurm validation missing: $required_text" >&2; exit 3;
    }
  done
  if [[ "$stage" == fit-prefilter ]]; then
    [[ "$slurm_record" != *"TresPerNode=gres/npu:"* ]] || { echo "fit terminal record unexpectedly has NPU" >&2; exit 3; }
  else
    [[ "$slurm_record" == *"TresPerNode=gres/npu:910B3:1"* ]] || { echo "operator NPU terminal record mismatch" >&2; exit 3; }
  fi
else
  slurm_source="sacct persistent accounting record"
  sacct_fields=JobIDRaw,JobName,User,State,ExitCode,Elapsed,Partition,NodeList,AllocTRES,ReqTRES
  sacct_output=$("$slurm_root/sacct" -X -j "$job_id" -n -P -o "$sacct_fields") || {
    echo "terminal Slurm accounting query failed" >&2; exit 3;
  }
  slurm_record=$(printf "%s\n" "$sacct_output" | awk -F "|" -v id="$job_id" '\''$1==id {print; exit}'\'')
  [[ -n "$slurm_record" ]] || { echo "terminal Slurm accounting record is absent" >&2; exit 3; }
  IFS="|" read -r accounting_job_id accounting_job_name accounting_user accounting_state \
    accounting_exit accounting_elapsed accounting_partition accounting_nodes \
    accounting_alloc_tres accounting_req_tres <<<"$slurm_record"
  [[ "$accounting_job_id" == "$job_id" ]] || { echo "accounting job id mismatch" >&2; exit 3; }
  [[ "$accounting_user" == researcher ]] || { echo "accounting user mismatch" >&2; exit 3; }
  [[ "$accounting_state" == COMPLETED ]] || { echo "accounting state is not COMPLETED" >&2; exit 3; }
  [[ "$accounting_exit" == 0:0 ]] || { echo "accounting exit code is not 0:0" >&2; exit 3; }
  [[ -n "$accounting_elapsed" && "$accounting_partition" == a01 && "$accounting_nodes" == a07 ]] || {
    echo "accounting resource placement mismatch" >&2; exit 3;
  }
  accounting_tres="$accounting_alloc_tres,$accounting_req_tres"
  if [[ "$stage" == fit-prefilter ]]; then
    [[ "$accounting_tres" != *"gres/npu"* ]] || { echo "fit accounting unexpectedly has NPU" >&2; exit 3; }
  else
    [[ "$accounting_tres" == *"gres/npu:910B3=1"* ]] || { echo "operator accounting NPU mismatch" >&2; exit 3; }
  fi
  slurm_record="fields=$sacct_fields\n$slurm_record"
fi

mkdir -p "$archive_dir" "$staging/runs"
cp -a "$run_dir" "$staging/runs/"
copied_run=$staging/runs/$run_id
[[ ! -e "$copied_run/artifact_tree.sha256" ]] || { echo "copied run already has artifact manifest" >&2; exit 2; }
printf "%s\n" "$slurm_record" >"$staging/SLURM_TERMINAL_RECORD.txt"
printf "%s\n" \
  "{\"schema_version\":1,\"run_id\":\"$run_id\",\"job_id\":$job_id,\"stage\":\"$stage\",\"scientific_audit_complete\":true,\"rows\":$rows,\"boundary_rows\":$boundary_rows,\"candidate_count\":$candidates,\"slurm_completed_verified\":true,\"slurm_verification_source\":\"$slurm_source\",\"production_rollout_approved\":false}" \
  >"$staging/TRANSFER_STATUS.json"
(
  cd "$copied_run"
  find . -type f ! -name artifact_tree.sha256 -print | LC_ALL=C sort | xargs sha256sum >artifact_tree.sha256
  sha256sum -c artifact_tree.sha256
)
COPYFILE_DISABLE=1 tar --no-xattrs -C "$staging" -czf "$archive_incoming" .
archive_sha=$(sha256sum "$archive_incoming" | awk "{print \$1}")
mv "$archive_incoming" "$archive_final"
rm -rf -- "$staging"
printf "archive_name=%s\narchive_sha256=%s\narchive_path=%s\n" \
  "$archive_name" "$archive_sha" "$archive_final"
'
printf -v cluster_command '%q ' bash -c "$remote_script" -- "$run_id"
printf -v outer_command '%q ' docker exec -i "$tailscale_container" ssh "$cluster_host" "$cluster_command"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" "$outer_command"
