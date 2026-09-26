#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

python_bin=${PYTHON:-python3}
command=${1:-demo}
if [[ $# -gt 0 ]]; then shift; fi

require_modern_python() {
  "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else "Python 3.10+ is required")'
}

case "$command" in
  demo)
    "$python_bin" scripts/public_demo.py "$@"
    ;;
  test)
    require_modern_python
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
      "$python_bin" -m unittest discover -s tests -v
    ;;
  audit)
    "$python_bin" scripts/privacy_audit.py
    ;;
  all)
    require_modern_python
    "$python_bin" scripts/public_demo.py
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
      "$python_bin" -m unittest discover -s tests -v
    "$python_bin" scripts/privacy_audit.py
    ;;
  pilot)
    require_modern_python
    exec "$python_bin" scripts/run_experiment.py "$@"
    ;;
  length-scan)
    require_modern_python
    exec "$python_bin" scripts/run_length_scan.py "$@"
    ;;
  *)
    echo "Usage: ./run.sh {demo|test|audit|all|pilot|length-scan} [arguments...]" >&2
    exit 2
    ;;
esac
