#!/usr/bin/env bash
# Usage: bash setup_models.sh [DESTINATION_DIRECTORY]
# Run from your cloned locality-dilemma repository on an authorized PSC node.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ ! -f download_models.py || ! -f requirements-local.txt ]]; then
  echo "ERROR: Place setup_models.sh in the locality-dilemma repository (next to download_models.py)." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  echo "ERROR: Python 3.10+ is required. PSC's default Python may be 3.6." >&2
  echo "Set PYTHON_BIN to an available Python 3.11/3.12 interpreter, e.g.:" >&2
  echo "  PYTHON_BIN=/path/to/python3.12 bash setup_models.sh" >&2
  exit 1
fi

ENV_DIR="${LOCALITY_VENV:-$HOME/locality-dilemma-venv}"
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"
python -m pip install -r requirements-local.txt

# Optional destination argument: when supplied, model files are written there by
# running from that destination? download_models.py hardcodes models/{base,adapters},
# so create symlink models -> destination (only if safe).
if [[ $# -ge 1 ]]; then
  MODEL_DIR="$1"
  mkdir -p "$MODEL_DIR"
  if [[ -L models ]]; then
    existing="$(readlink -f models)"
    desired="$(readlink -f "$MODEL_DIR")"
    if [[ "$existing" != "$desired" ]]; then
      echo "ERROR: models symlink points to $existing, not $desired" >&2; exit 1
    fi
  elif [[ -e models ]]; then
    echo "ERROR: models directory already exists. Omit destination or move it manually." >&2; exit 1
  else
    ln -s "$(readlink -f "$MODEL_DIR")" models
  fi
fi

python download_models.py --adapters predibase/customer_support predibase/magicoder

python - <<'PY'
import json
from pathlib import Path
expected = {'predibase__customer_support', 'predibase__magicoder'}
base = Path('models/base')
assert (base / 'config.json').is_file(), 'Missing base model config.json'
for adapter in sorted(expected):
    f = Path('models/adapters') / adapter / 'adapter_config.json'
    assert f.is_file(), f'Missing adapter config: {f}'
    config = json.loads(f.read_text())
    actual_base = config.get('base_model_name_or_path')
    assert actual_base == 'mistralai/Mistral-7B-v0.1', (adapter, actual_base)
    print(f'{adapter}: base={actual_base}, rank={config.get("r")}, targets={config.get("target_modules")}')
print('Model and adapter configuration checks passed.')
PY

du -sh models/base models/adapters/*
echo 'Download setup complete. GPU inference requires a separate authorized allocation and vLLM environment.'
