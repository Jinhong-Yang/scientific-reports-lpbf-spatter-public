#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" -m pytest -q tests/test_submission_artifacts.py tests/test_submission_audit.py tests/test_release_archive.py
if [ -f release/SHA256SUMS.txt ]; then
  "$PYTHON_BIN" scripts/verify_sha256s.py release/SHA256SUMS.txt
fi
if [ "${REBUILD_MANUSCRIPT:-0}" = "1" ]; then
  "$PYTHON_BIN" src/reporting/build_submission_artifacts.py --build
fi
printf '%s\n' 'Reproducibility checks complete.'
