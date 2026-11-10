@echo off
cd /d "%~dp0"
echo Starting Momentum Lab at http://127.0.0.1:5000  (close this window to stop)
python -m momentum_lab app
pause
