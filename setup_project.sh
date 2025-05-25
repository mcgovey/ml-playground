#!/bin/bash

# Check if UV is installed
if ! command -v uv &> /dev/null
then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc || source ~/.zshrc
fi

echo "Creating virtual environment with UV..."
uv venv
echo "Virtual environment created."

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete! You can now run the generator with:"
echo "uv run ecommerce_fraud_data_generator.py generate"
