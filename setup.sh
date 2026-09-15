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

# 5. Run automated test suite to verify installation
echo "Running automated test suite (140 tests)..."
PYTHONPATH=backend pytest backend -q

echo ""
echo "======================================================================"
echo "  Setup Complete! All test suites passed."
echo "  To start the server, simply run:"
echo "      ./run.sh"
echo "  or:"
echo "      source .venv/bin/activate && python run.py"
echo "======================================================================"
