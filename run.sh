#!/usr/bin/env bash
# ==============================================================================
# MoSPI PAIMANA Predictive Intelligence Platform - Linux/macOS Runner
# ==============================================================================
set -e

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python run.py
