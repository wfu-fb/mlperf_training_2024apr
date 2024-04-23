#!/bin/bash
# Redirects logs to the $DUMP_DIR/<job version>/< set up.
# If $DUMP_DIR is not set, this is a no-op.

set -eEu -o pipefail

if [[ -z "${DUMP_DIR-}" ]]; then
  exec "$@"
fi

PATH_SUFFIX=""
if [[ "${DUMP_DIR}" != *"${MAST_HPC_JOB_NAME}"* ]]; then
  PATH_SUFFIX="_${MAST_HPC_JOB_NAME}"
fi

LOG_DIR="${DUMP_DIR}/logs/v${MAST_HPC_JOB_VERSION}_attempt${MAST_HPC_JOB_ATTEMPT_INDEX}${PATH_SUFFIX}/"
mkdir -p "$LOG_DIR"

HOSTNAME=$(hostname)
HOST=${HOSTNAME%.facebook.com}

STDERR="$LOG_DIR"/"$HOST"_stderr.log
STDOUT="$LOG_DIR"/"$HOST"_stdout.log

TIMESTAMP="$(date +%s)"

echo "========== ($TIMESTAMP) Task $TW_TASK_ID STDOUT: $* ==========" >> "$STDOUT"
echo "========== ($TIMESTAMP) Task $TW_TASK_ID STDERR: $* ==========" >> "$STDERR"

exec "$@" > >(tee -a "$STDOUT") 2> >(tee -a "$STDERR" >&2)
