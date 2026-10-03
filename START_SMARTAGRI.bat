@echo off
setlocal
cd /d "%~dp0"
color 2F

echo.
echo ================================================
echo          SMARTAGRI AI - FARMER MODE
echo ================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Python was not found. Install Python 3.11 or newer and try again.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
echo.
echo Checking required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Check your internet connection or install the packages manually.
    pause
    exit /b 1
)

if not exist "models\crop_farmer_models.joblib" (
    echo.
    echo Trained models not found. Training them now...
    python train_models.py
    if errorlevel 1 (
        echo Model training failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting SmartAgri AI...
echo Open http://127.0.0.1:5000 in your browser.
echo Press Ctrl+C in this window to stop the application.
echo.
python app.py
pause
