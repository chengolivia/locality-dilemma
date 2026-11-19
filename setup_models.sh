#!/usr/bin/env bash
# ==============================================================================
# Model & Environment Setup Automation Script
# Root wrapper delegating to scripts/setup/setup_models.sh
# ==============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/scripts/setup/setup_models.sh" "$@"
