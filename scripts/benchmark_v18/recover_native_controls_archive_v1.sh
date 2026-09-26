#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --job-id ID --run-id RUN_ID --receipt PATH" >&2
}

job_id=
run_id=
receipt=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id) job_id=${2:-}; shift 2 ;;
    --run-id) run_id=${2:-}; shift 2 ;;
    --receipt) receipt=${2:-}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "$job_id" =~ ^[0-9]+$ ]] || { usage; exit 2; }
[[ "$run_id" =~ ^qwen3-5-9b-cdge-v4-2-native-controls-CDGE42-[0-9a-f]{16}-[0-9]{8}T[0-9]{6}Z$ ]] || {
  usage; exit 2;
}
[[ -n "$receipt" && ! -e "$receipt" ]] || {
  echo "receipt path is empty or already exists" >&2; exit 2;
}

repo=$(cd "$(dirname "$0")/../.." && pwd)
archive_name=$run_id.tar.gz
cluster_path=/workspace/context-mismatch-qwen3-5-9b/evidence/$archive_name
mirror_path=/workspace/context-mismatch-qwen3-5-9b/evidence-archives/$archive_name
ssh_target=${CLUSTER_SSH_TARGET:-archive-host}
container=${CLUSTER_TAILSCALE_CONTAINER:-tunnel-container}
cluster_host=${CLUSTER_CLUSTER_HOST:-compute-cluster}

record=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" \
  "docker exec $container ssh $cluster_host 'tar -xOzf $cluster_path ./SLURM_TERMINAL_RECORD.txt'")
for token in "JobId=$job_id" "UserId=researcher(" "JobState=COMPLETED" "ExitCode=0:0"; do
  [[ "$record" == *"$token"* ]] || {
    echo "captured terminal record missing: $token" >&2; exit 3;
  }
done

remote=$(command cat <<'REMOTE'
set -euo pipefail
archive_name=$1; run_id=$2; job_id=$3
source=/workspace/context-mismatch-qwen3-5-9b/evidence/$archive_name
mirror=/workspace/context-mismatch-qwen3-5-9b/evidence-archives/$archive_name
incoming=$mirror.incoming
[[ -r $source ]] || { echo "cluster archive missing" >&2; exit 3; }
sha=$(sha256sum "$source" | awk '{print $1}')
terminal=$(tar -xOzf "$source" ./SLURM_TERMINAL_RECORD.txt)
for token in "JobId=$job_id" "UserId=researcher(" "JobState=COMPLETED" "ExitCode=0:0"; do
  [[ $terminal == *"$token"* ]] || { echo "archived terminal record mismatch: $token" >&2; exit 3; }
done
tar -xOzf "$source" ./TRANSFER_STATUS.json | python3 -c '
import json, sys
run_id, job_id = sys.argv[1], int(sys.argv[2])
value = json.load(sys.stdin)
expected = {
    "run_id": run_id,
    "job_id": job_id,
    "stage": "qwen35_cdge_v4_2_native_protected_controls",
    "scientific_audit_complete": True,
    "slurm_completed_verified": True,
    "final_test_open": False,
    "final_test_open_count": 0,
    "production_rollout_approved": False,
}
for key, wanted in expected.items():
    if value.get(key) != wanted:
        raise SystemExit(f"transfer status mismatch: {key}")
' "$run_id" "$job_id"
mkdir -p "$(dirname "$mirror")"
if [[ -e $mirror ]]; then
  [[ $(sha256sum "$mirror" | awk '{print $1}') == "$sha" ]] || {
    echo "existing mirror SHA mismatch" >&2; exit 3;
  }
else
  [[ ! -e $incoming ]] || { echo "incoming mirror path exists" >&2; exit 2; }
  cp "$source" "$incoming"
  [[ $(sha256sum "$incoming" | awk '{print $1}') == "$sha" ]] || exit 3
  mv "$incoming" "$mirror"
fi
printf 'archive_sha256=%s\nmirror_sha256=%s\n' "$sha" "$(sha256sum "$mirror" | awk '{print $1}')"
REMOTE
)
printf -v cluster_command '%q ' bash -c "$remote" -- "$archive_name" "$run_id" "$job_id"
printf -v outer '%q ' docker exec -i "$container" ssh "$cluster_host" "$cluster_command"
promotion=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" "$outer")
archive_sha=$(printf '%s\n' "$promotion" | awk -F= '$1=="archive_sha256" {print $2}')
mirror_sha=$(printf '%s\n' "$promotion" | awk -F= '$1=="mirror_sha256" {print $2}')
[[ "$archive_sha" =~ ^[0-9a-f]{64}$ && "$mirror_sha" == "$archive_sha" ]] || {
  echo "remote promotion did not return matching SHA values" >&2; exit 3;
}

local_root=$repo/artifacts/qwen3-5-9b-v1/evidence
local_path=$local_root/$run_id
mkdir -p "$local_root"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/q35-controls-recovery.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
archive=$tmp/$archive_name
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" \
  "docker exec $container ssh $cluster_host 'cat $cluster_path'" >"$archive"
[[ $(shasum -a 256 "$archive" | awk '{print $1}') == "$archive_sha" ]] || {
  echo "local archive SHA mismatch" >&2; exit 3;
}
mkdir "$tmp/extracted"
tar -xzf "$archive" -C "$tmp/extracted"
(
  cd "$tmp/extracted/runs/$run_id"
  shasum -a 256 -c artifact_tree.sha256
)
if [[ -e "$local_path" ]]; then
  (
    cd "$local_path/runs/$run_id"
    shasum -a 256 -c artifact_tree.sha256
  )
  diff -qr "$tmp/extracted" "$local_path"
else
  mv "$tmp/extracted" "$local_path"
fi

mkdir -p "$(dirname "$receipt")"
incoming_receipt=$receipt.incoming.$$
python3 - "$job_id" "$run_id" "$archive_name" "$archive_sha" "$cluster_path" \
  "$mirror_path" "$local_path" "$receipt" "$record" <<'PY' >"$incoming_receipt"
import json, sys
job_id, run_id, name, sha, cluster_path, mirror_path, local_path, receipt, record = sys.argv[1:]
print(json.dumps({
    "schema_version": 2,
    "job_id": int(job_id),
    "run_id": run_id,
    "archive_name": name,
    "archive_sha256": sha,
    "cluster_shared_path": cluster_path,
    "durable_mirror_path": mirror_path,
    "host_data_path": mirror_path,
    "local_bundle_path": local_path,
    "slurm_terminal_record": record,
    "cluster_shared_copy_verified": True,
    "durable_mirror_copy_verified": True,
    "host_data_copy_verified": True,
    "local_copy_verified": True,
    "transit_host_copy_retained": False,
    "recovery_reason": "code-v106 pull helper omitted durable-mirror promotion",
    "receipt_path": receipt,
}, indent=2, sort_keys=True))
PY
if ! ln "$incoming_receipt" "$receipt"; then
  echo "refusing to overwrite a concurrently created receipt" >&2
  exit 2
fi
rm "$incoming_receipt"
printf 'receipt=%s\narchive_name=%s\narchive_sha256=%s\n' "$receipt" "$archive_name" "$archive_sha"
