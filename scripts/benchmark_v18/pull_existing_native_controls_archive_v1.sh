#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?run id}
expected_sha=${2:?archive sha256}
[[ "$run_id" =~ ^qwen3-5-9b-cdge-v4-2-native-controls-CDGE42-[0-9a-f]{16}-[0-9]{8}T[0-9]{6}Z$ ]] || exit 2
[[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2

repo=$(cd "$(dirname "$0")/../.." && pwd)
archive_name=$run_id.tar.gz
source=/workspace/context-mismatch-qwen3-5-9b/evidence/$archive_name
destination_root=$repo/artifacts/qwen3-5-9b-v1/evidence-archives
destination=$destination_root/$archive_name
[[ ! -e "$destination" ]] || { echo "destination already exists" >&2; exit 2; }
mkdir -p "$destination_root"
incoming=$destination.incoming.$$
trap 'rm -f "$incoming"' EXIT

ssh_target=${CLUSTER_SSH_TARGET:-archive-host}
container=${CLUSTER_TAILSCALE_CONTAINER:-tunnel-container}
cluster_host=${CLUSTER_CLUSTER_HOST:-compute-cluster}
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" \
  "docker exec $container ssh $cluster_host 'cat $source'" >"$incoming"
observed_sha=$(shasum -a 256 "$incoming" | awk '{print $1}')
[[ "$observed_sha" == "$expected_sha" ]] || { echo "archive SHA mismatch" >&2; exit 3; }
tar -tzf "$incoming" ./SLURM_TERMINAL_RECORD.txt ./TRANSFER_STATUS.json >/dev/null
mv "$incoming" "$destination"
trap - EXIT
printf 'local_archive=%s\narchive_sha256=%s\n' "$destination" "$observed_sha"
