#!/bin/bash
set -e

echo "🔧 Setting up Python virtual environment..."
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "📦 Installing Python requirements..."
  pip install -r requirements.txt
fi

echo "☁️ Installing Google Cloud SDK..."

# Install dependencies
sudo apt-get update && sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    lsb-release

# Add the Cloud SDK distribution URI and public key
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --dearmor | sudo tee /usr/share/keyrings/cloud.google.gpg > /dev/null

# Install the Cloud SDK
sudo apt-get update && sudo apt-get install -y google-cloud-sdk

echo "✅ gcloud CLI installed successfully."
gcloud version
