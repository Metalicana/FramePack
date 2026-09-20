#!/usr/bin/env bash
set -euo pipefail
# Explicit absolute interpreter; no activation, SSH, or scheduler submission.
: "STUDY_PYTHON:?Set STUDY_PYTHON to an absolute Python executable with NumPy}"
case "$STUDY_PYTHON" in /*) ;; *) echo 'STUDY_PYTHON must be absolute' >&2; exit 2;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$STUDY_PYTHON" -u "$SCRIPT_DIR/study.py" "$@"
