#!/usr/bin/env bash
# ==============================================================================
# MoSPI PAIMANA Predictive Intelligence Platform - Linux/macOS Setup Script
# ==============================================================================
set -e

echo "----------------------------------------------------------------------"
echo "  Setting up MoSPI PAIMANA Platform (Group Project Environment)"
echo "----------------------------------------------------------------------"

# 1. Determine Python command (python3 or python)
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python is not installed. Please install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: $PY_VERSION"

# 2. Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    $PYTHON_CMD -m venv .venv
else
    echo "Existing virtual environment found in .venv."
fi

# 3. Activate virtual environment
# shellcheck disable=SC1091
source .venv/bin/activate

# 4. Install / upgrade dependencies
echo "Installing dependencies from backend/requirements.txt..."
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# 5. Build the model artifacts from the harvested MoSPI panel.
#    Every performance figure the platform reports comes from these files, so
#    a checkout without them serves a degraded legacy model and the benchmark
#    endpoints return 503. Regenerating is deterministic (seeded).
if [ ! -f "model/paimana_model_metrics.json" ] || [ "${REBUILD_MODEL:-0}" = "1" ]; then
    echo "Harvesting 13 freeze months from the live MoSPI portal..."
    python scripts/harvest_monthly.py
    echo "Building longitudinal panel from the harvested MoSPI data..."
    python scripts/build_panel.py
    echo "Training and benchmarking the early-warning model..."
    python scripts/train_model.py
    echo "Measuring early-warning lead time..."
    python scripts/lead_time_backtest.py
    echo "Checking whether the label tracks physical distress..."
    python scripts/label_validity.py
    echo "Testing robustness to the reconstructed time axis (retrains 5x, ~2 min)..."
    python scripts/ordering_sensitivity.py
else
    echo "Model artifacts already present (set REBUILD_MODEL=1 to rebuild)."
fi

# 6. Run automated test suite to verify installation
echo "Running automated test suite (service tests + frozen golden cases)..."
PYTHONPATH=backend pytest backend tests -q

echo ""
echo "======================================================================"
echo "  Setup Complete! All test suites passed."
echo "  To start the server, simply run:"
echo "      ./run.sh"
echo "  or:"
echo "      source .venv/bin/activate && python run.py"
echo "======================================================================"
