@echo off
echo Starting LandSentry 2.0 Platform...

REM Start the FastAPI backend server in a new command window
echo Starting FastAPI Backend...
start cmd /k "python -m uvicorn backend.main:app --reload --port 8000"

REM Wait a couple of seconds to ensure backend initializes
timeout /t 3 /nobreak >nul

REM Start the Streamlit frontend dashboard
echo Starting Streamlit Dashboard...
python -m streamlit run dashboard/app.py

echo LandSentry Platform processes have finished!
pause
