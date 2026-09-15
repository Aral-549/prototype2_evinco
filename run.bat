@echo off
REM ==============================================================================
REM MoSPI PAIMANA Predictive Intelligence Platform - Windows Server Runner
REM ==============================================================================
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
python run.py
pause
