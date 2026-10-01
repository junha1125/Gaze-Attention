#!/bin/bash
# Clone lmms-eval at the commit used for the paper, apply the Gaze Attention patch and install it.
#
# Usage (from the repository root, inside the environment where `cambrian` is installed):
#   bash lmms_eval_integration/setup.sh [destination]
#
# Default destination: third_party/lmms-eval
set -e

LMMS_EVAL_COMMIT=40a369db5e8c2b02ba437d2e6a60471e8d7e4da3

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEST="${1:-${HERE}/../third_party/lmms-eval}"

if [ ! -d "${DEST}/.git" ]; then
    mkdir -p "$(dirname "${DEST}")"
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git "${DEST}"
fi

cd "${DEST}"
git checkout "${LMMS_EVAL_COMMIT}"
if git apply --reverse --check "${HERE}/gaze_attention.patch" 2>/dev/null; then
    echo "Patch already applied."
else
    git apply "${HERE}/gaze_attention.patch"
fi

# The constraints keep the versions required by the cambrian package (torch, transformers, ...).
pip install -e . -c "${HERE}/constraints.txt"

echo "lmms-eval is installed in ${DEST}"
