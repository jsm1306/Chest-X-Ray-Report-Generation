#!/bin/bash

# Medical Diagnosis API - Startup Script for Linux/macOS

echo ""
echo "========================================"
echo "Medical Diagnosis Report API"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org or using your package manager"
    exit 1
fi

echo "[OK] Python is installed"
echo ""

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"
echo ""

# Create venv if doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "[OK] Virtual environment created"
    echo ""
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi
echo "[OK] Virtual environment activated"
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
pip show fastapi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    echo "[OK] Dependencies installed"
    echo ""
fi

echo "[OK] All dependencies installed"
echo ""

# Check if model files exist
echo "Checking model files..."
if [ ! -f "encoder_model.keras" ]; then
    echo "WARNING: encoder_model.keras not found"
fi
if [ ! -f "full_model.keras" ]; then
    echo "WARNING: full_model.keras not found"
fi
if [ ! -f "tokenizer.pkl" ]; then
    echo "WARNING: tokenizer.pkl not found"
fi
echo ""

# Start the API
echo ""
echo "========================================"
echo "Starting Medical Diagnosis API Server..."
echo "========================================"
echo ""
echo "Server will run on: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: API failed to start"
    exit 1
fi
