#!/bin/bash

set -eExu -o pipefail


FUSE_SRC="ws://ws.ai.eag0genai/genai_fair_llm"
FUSE_DST="/mnt/wsfuse"

mkdir -p "$FUSE_DST"

/packages/oil.oilfs/scripts/genai_wrapper.sh "$FUSE_SRC" "$FUSE_DST" -u "$AI_RM_ATTRIBUTION"
