@echo off
REM Medical Diagnosis API - Startup Script for Windows

echo.
echo ========================================
echo Medical Diagnosis Report API
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

REM Check if venv exists, create if not
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Check if requirements are installed
echo Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
    echo.
)

echo [OK] All dependencies installed
echo.

REM Check if model files exist
echo Checking model files...
if not exist "encoder_model.keras" (
    echo WARNING: encoder_model.keras not found
)
if not exist "full_model.keras" (
    echo WARNING: full_model.keras not found
)
if not exist "tokenizer.pkl" (
    echo WARNING: tokenizer.pkl not found
)
echo.

REM Start the API
echo.
echo ========================================
echo Starting Medical Diagnosis API Server...
echo ========================================
echo.
echo Server will run on: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

if errorlevel 1 (
    echo.
    echo ERROR: API failed to start
    pause
    exit /b 1
)

pause
