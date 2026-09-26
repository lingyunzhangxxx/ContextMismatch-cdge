#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
ssh_target=${CLUSTER_SSH_TARGET:-archive-host}
tailscale_container=${CLUSTER_TAILSCALE_CONTAINER:-tunnel-container}
cluster_host=${CLUSTER_CLUSTER_HOST:-compute-cluster}
skill_root=${CODEX_SKILLS_ROOT:-/workspace/local-user/.codex-baizhi/skills}
selector=$skill_root/ascend-cluster/scripts/select_node.py
home_root=/workspace/context-mismatch-qwen3-8b
work_root=/workspace/context-mismatch-qwen3-8b
code_root=$work_root/code-v57
timestamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
created_utc=${2:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}

[[ "$timestamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid timestamp" >&2; exit 2; }
[[ "$created_utc" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || {
  echo "invalid created UTC" >&2; exit 2;
}
for required in "$selector" "$repo_root/ascend/qwen3_8b/submit_v4_behavior_diagnostic_v56.sh"; do
  [[ -r "$required" ]] || { echo "missing launch prerequisite: $required" >&2; exit 2; }
done

staging=$(mktemp -d "${TMPDIR:-/tmp}/qwen3-v56-behavior-launch.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
snapshot=$staging/node-selection.json
"$selector" --partition a01 --npu-type 910B3 --npus 1 --cpus 8 \
  --mem-mib 131072 --json >"$snapshot"
selected_node=$(python3 - "$snapshot" <<'PY'
import json,sys
value=json.load(open(sys.argv[1])); node=value.get('node','')
if not (len(node)==3 and node[0]=='a' and node[1:].isdigit()): raise SystemExit('invalid selected node')
print(node)
PY
)
snapshot_sha=$(shasum -a 256 "$snapshot" | awk '{print $1}')
snapshot_b64=$(base64 <"$snapshot" | tr -d '\n')
snapshot_remote=$home_root/authorizations/QWEN3_8B_V57_NODE_SELECTION_$timestamp.json
authorization=$home_root/authorizations/QWEN3_8B_ADSGE_V4_POST_FIT_BEHAVIOR_DIAGNOSTIC_CODE_V57_$timestamp.json

remote_script='
set -euo pipefail
code_root=$1
snapshot=$2
authorization=$3
selected_node=$4
timestamp=$5
created_utc=$6
snapshot_sha=$7
snapshot_b64=$8
home_root=/workspace/context-mismatch-qwen3-8b
[[ $(id -un) == researcher ]] || { echo "refusing unexpected user" >&2; exit 2; }
[[ -d "$code_root" && -r "$code_root/bundle.sha256" ]] || { echo "code-v57 is missing" >&2; exit 2; }
[[ ! -e "$snapshot" && ! -e "$authorization" ]] || { echo "refusing existing launch controls" >&2; exit 2; }
queue_count=$(/service/slurm/24.11.5/bin/squeue -h -u researcher -o "%i" | awk "NF {n++} END {print n+0}")
(( queue_count < 7 )) || { echo "queue limit reached before materialization" >&2; exit 2; }
mkdir -p "$home_root/authorizations"
incoming=$snapshot.incoming.$$
printf "%s" "$snapshot_b64" | base64 -d >"$incoming"
[[ $(sha256sum "$incoming" | awk "{print \$1}") == "$snapshot_sha" ]] || {
  echo "snapshot transfer SHA mismatch" >&2; exit 3;
}
mv "$incoming" "$snapshot"
editor=$code_root/protocol/QWEN3_8B_ABSTAINING_DIRECTIONAL_GOVERNANCE_EDITOR_V4.json
diagnostic=$code_root/protocol/QWEN3_8B_ADSGE_V4_POST_FIT_BEHAVIOR_DIAGNOSTIC_V1.json
crossover=$code_root/protocol/GOVERNANCE_TASK_CROSSOVER_V1.json
sites=$code_root/artifacts/QWEN3_8B_OPERATOR_SITE_MANIFEST_V1.json
manifest=$code_root/artifacts/benchmark_manifest.jsonl
design=$code_root/artifacts/governance_crossover_operator_dev_design_audit_v2.json
v3_analysis=$code_root/artifacts/v3_directional_behavior_operator_dev.analysis.json
v4_run=$home_root/runs/qwen3-8b-governance-abstaining-router-fit-20260729T010436Z
checkpoint=$v4_run/abstaining_router_fit/abstaining_directional_editor__layer_27__mlp.pt
fit_report=$v4_run/abstaining_router_fit/abstaining_router_fit_report.json
model_manifest=$home_root/runs/qwen3-8b-full-behavior-20260727T121047Z/node_model_verification.json
PYTHONPATH="$code_root" python3 -m scripts.benchmark_v8.materialize_v4_behavior_diagnostic_authorization \
  --code-root "$code_root" --selector-snapshot "$snapshot" --selected-node "$selected_node" \
  --created-utc "$created_utc" --editor-contract "$editor" \
  --diagnostic-contract "$diagnostic" --crossover-contract "$crossover" \
  --operator-site-manifest "$sites" --manifest "$manifest" --design-audit "$design" \
  --checkpoint "$checkpoint" --fit-report "$fit_report" \
  --v3-operator-dev-analysis "$v3_analysis" --model-manifest "$model_manifest" \
  --output "$authorization"
CLUSTER_TARGET_NODE="$selected_node" \
  "$code_root/submit_v4_behavior_diagnostic_v56.sh" \
  "$code_root" "$authorization" "$timestamp"
'
printf -v cluster_command '%q ' bash -c "$remote_script" -- "$code_root" \
  "$snapshot_remote" "$authorization" "$selected_node" "$timestamp" \
  "$created_utc" "$snapshot_sha" "$snapshot_b64"
printf -v outer_command '%q ' docker exec -i "$tailscale_container" ssh "$cluster_host" "$cluster_command"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ssh_target" "$outer_command"
