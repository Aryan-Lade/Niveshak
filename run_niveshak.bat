@echo off
rem Starts the Niveshak API and web terminal on this computer and opens it in your browser.
rem Double-click this file, or run it from a terminal that has your Python environment active.
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Activate your Python environment, or install Python 3.10 or newer, then run this again.
  pause
  exit /b 1
)

python -c "import fastapi, uvicorn, numpy, pandas, yaml, pydantic" >nul 2>nul
if errorlevel 1 (
  echo Some Python packages are missing. Install everything the project uses with:
  echo     python -m pip install -r requirements.txt
  echo To run only the web terminal, this smaller set is enough:
  echo     python -m pip install fastapi uvicorn numpy pandas pyyaml pydantic python-dotenv
  pause
  exit /b 1
)

echo Starting Niveshak at http://localhost:8000/niveshak
echo Leave this window open while you use the site. Press Ctrl+C to stop it.
start "" /min cmd /c "ping -n 4 127.0.0.1 >nul && start http://localhost:8000/niveshak"
python -m uvicorn stock_engine.api.main:app --host 127.0.0.1 --port 8000
echo.
echo The server has stopped. If it stopped straight away, read the message above: port 8000 may already be in use.
pause
