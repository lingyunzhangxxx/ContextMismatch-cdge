#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
ssh_target=${CLUSTER_SSH_TARGET:-archive-host}
tailscale_container=${CLUSTER_TAILSCALE_CONTAINER:-tunnel-container}
cluster_host=${CLUSTER_CLUSTER_HOST:-compute-cluster}
work_root=/workspace/context-mismatch-qwen3-8b
remote_final=$work_root/code-v57
remote_incoming=$work_root/.code-v57.incoming
v3_analysis=$repo_root/artifacts/qwen3-8b-v1/evidence/qwen3-8b-governance-directional-operator-dev-20260728T203900Z/runs/qwen3-8b-governance-directional-operator-dev-20260728T203900Z/directional_behavior_operator_dev.analysis.json

for required in \
  "$repo_root/protocol/QWEN3_8B_ADSGE_V4_POST_FIT_BEHAVIOR_DIAGNOSTIC_V1.json" \
  "$repo_root/protocol/QWEN3_8B_ABSTAINING_DIRECTIONAL_GOVERNANCE_EDITOR_V4.json" \
  "$repo_root/scripts/benchmark_v8/analyze_v4_behavior_diagnostic.py" \
  "$repo_root/scripts/benchmark_v8/materialize_v4_behavior_diagnostic_authorization.py" \
  "$repo_root/ascend/qwen3_8b/v4_behavior_diagnostic_v56_job.sh" \
  "$repo_root/ascend/qwen3_8b/submit_v4_behavior_diagnostic_v56.sh" \
  "$repo_root/ascend/qwen3_8b/archive_v4_behavior_diagnostic_v56_evidence.sh" \
  "$v3_analysis"; do
  [[ -r "$required" ]] || { echo "missing code-v56 input: $required" >&2; exit 2; }
done

staging=$(mktemp -d "${TMPDIR:-/tmp}/qwen3-8b-v4-behavior-code-v57.XXXXXX")
archive=$staging.tar.gz
cleanup() {
  rm -rf -- "$staging"
  rm -f -- "$archive"
}
trap cleanup EXIT
mkdir -p "$staging/ascend/qwen3_8b" "$staging/artifacts" "$staging/protocol" \
  "$staging/scripts" "$staging/tests"
for version in 1 2 3 4 5 8; do
  mkdir -p "$staging/scripts/benchmark_v$version"
  cp "$repo_root/scripts/benchmark_v$version/"*.py "$staging/scripts/benchmark_v$version/"
done
cp "$repo_root/scripts/__init__.py" "$staging/scripts/"
cp "$repo_root/ascend/qwen3_8b/v4_behavior_diagnostic_v56_job.sh" \
  "$staging/v4_behavior_diagnostic_v56_job.sh"
cp "$repo_root/ascend/qwen3_8b/submit_v4_behavior_diagnostic_v56.sh" \
  "$staging/submit_v4_behavior_diagnostic_v56.sh"
cp "$repo_root/ascend/qwen3_8b/"{v4_behavior_diagnostic_v56_job.sh,submit_v4_behavior_diagnostic_v56.sh,archive_v4_behavior_diagnostic_v56_evidence.sh} \
  "$staging/ascend/qwen3_8b/"
cp "$repo_root/ascend/qwen3_8b/model_source_contract.json" "$staging/model_source_contract.json"
cp "$repo_root/ascend/qwen3_8b/verify_model_source.py" "$staging/verify_model_source.py"
cp "$repo_root/ascend/hash_artifact_tree.py" "$staging/hash_artifact_tree.py"
cp "$repo_root/protocol/"*.json "$staging/protocol/"
cp "$repo_root/artifacts/qwen3-8b-v1/"{QWEN3_8B_OPERATOR_SITE_MANIFEST_V1.json,benchmark_manifest.jsonl,benchmark_manifest.report.json,governance_crossover_operator_dev_design_audit_v2.json} \
  "$staging/artifacts/"
cp "$v3_analysis" "$staging/artifacts/v3_directional_behavior_operator_dev.analysis.json"
cp "$repo_root/tests/"{test_directional_governance_v3.py,test_abstaining_directional_governance_v4.py,test_v4_behavior_diagnostic.py} \
  "$staging/tests/"
(
  cd "$staging"
  find . -type f ! -name bundle.sha256 -print | LC_ALL=C sort | xargs shasum -a 256 >bundle.sha256
)
COPYFILE_DISABLE=1 tar --no-xattrs -C "$staging" -czf "$archive" .

remote_script='
set -euo pipefail
incoming=$1
final=$2
work_root=$3
[[ $(id -un) == researcher ]] || { echo "refusing unexpected user" >&2; exit 2; }
workspace=/workspace
[[ $(stat -c %U "$workspace") == researcher ]] || { echo "workspace owner mismatch" >&2; exit 2; }
filesystem=$(findmnt -n -o FSTYPE -T "$workspace" | tail -n 1)
[[ "$filesystem" == nfs || "$filesystem" == nfs4 ]] || { echo "workspace is not NFS" >&2; exit 2; }
[[ -r "$workspace" && -w "$workspace" && -x "$workspace" ]] || { echo "workspace access mismatch" >&2; exit 2; }
[[ ! -e "$incoming" && ! -e "$final" ]] || { echo "refusing existing immutable code target" >&2; exit 2; }
mkdir -p "$work_root"
mkdir "$incoming"
tar -xzf - -C "$incoming"
if find "$incoming" -name "._*" -print -quit | grep -q .; then exit 3; fi
(cd "$incoming" && sha256sum -c bundle.sha256)
chmod 750 "$incoming/v4_behavior_diagnostic_v56_job.sh" \
  "$incoming/submit_v4_behavior_diagnostic_v56.sh" \
  "$incoming/ascend/qwen3_8b/v4_behavior_diagnostic_v56_job.sh" \
  "$incoming/ascend/qwen3_8b/submit_v4_behavior_diagnostic_v56.sh" \
  "$incoming/ascend/qwen3_8b/archive_v4_behavior_diagnostic_v56_evidence.sh" \
  "$incoming/scripts/benchmark_v8/materialize_v4_behavior_diagnostic_authorization.py" \
  "$incoming/verify_model_source.py" "$incoming/hash_artifact_tree.py"
manifest_sha=$(sha256sum "$incoming/bundle.sha256" | awk "{print \$1}")
mv "$incoming" "$final"
printf "bundle_promoted=%s\nbundle_manifest_sha256=%s\n" "$final" "$manifest_sha"
'
printf -v cluster_command '%q ' bash -c "$remote_script" -- \
  "$remote_incoming" "$remote_final" "$work_root"
printf -v outer_command '%q ' docker exec -i "$tailscale_container" ssh "$cluster_host" "$cluster_command"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" "$outer_command" <"$archive"
