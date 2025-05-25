@echo off
echo Setting up synthetic e-commerce fraud data generator project...

REM Check if UV is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo UV is not installed. Please install UV from https://astral.sh/uv
    echo Then run this script again.
    exit /b 1
)

echo Creating virtual environment with UV...
uv venv
echo Virtual environment created.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
uv pip install -r requirements.txt

echo Setup complete! You can now run the generator with:
echo uv run ecommerce_fraud_data_generator.py generate
