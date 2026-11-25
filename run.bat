@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   USED CAR PRICE PREDICTOR - Windows Launcher
echo ============================================================

if exist ".venv\Scripts\activate.bat" (
    echo [1/3] Activating project virtual environment (.venv)...
    call ".venv\Scripts\activate.bat"
) else (
    echo [1/3] .venv not found. Using system Python...
)

echo [2/3] Checking core dependencies...
python -c "import pandas, numpy, sklearn, shap, streamlit, joblib, plotly" 2>nul
if errorlevel 1 (
    echo Missing dependencies detected. Installing from requirements.txt...
    pip install -r requirements.txt
)

if not exist "models\best_model.pkl" (
    echo Trained model not found. Running training pipeline first...
    python -m src.train
)

echo [3/3] Starting Streamlit web application...
streamlit run app/app.py
pause
