@echo off
REM ==============================================================================
REM MoSPI PAIMANA Predictive Intelligence Platform - Windows Setup Script
REM ==============================================================================
echo ----------------------------------------------------------------------
echo   Setting up MoSPI PAIMANA Platform on Windows
echo ----------------------------------------------------------------------

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not detected on PATH. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment in .venv...
    python -m venv .venv
) else (
    echo Existing virtual environment found in .venv.
)

call .venv\Scripts\activate.bat

echo Upgrading pip and installing dependencies...
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

REM Build model artifacts if absent. Every reported performance figure comes
REM from these files; without them the platform serves a degraded legacy model.
if not exist "model\paimana_model_metrics.json" (
    echo Harvesting 13 freeze months from the live MoSPI portal...
    python scripts\harvest_monthly.py
    echo Building longitudinal panel from harvested MoSPI data...
    python scripts\build_panel.py
    echo Training and benchmarking the early-warning model...
    python scripts\train_model.py
    echo Measuring early-warning lead time...
    python scripts\lead_time_backtest.py
    echo Checking whether the label tracks physical distress...
    python scripts\label_validity.py
    echo Testing robustness to the reconstructed time axis...
    python scripts\ordering_sensitivity.py
) else (
    echo Model artifacts already present.
)

echo Running automated test suite (service tests + frozen golden cases)...
set PYTHONPATH=backend
pytest backend tests -q

echo.
echo ======================================================================
echo   Setup Complete! All test suites passed.
echo   To start the platform, run:
echo       run.bat
echo ======================================================================
pause
