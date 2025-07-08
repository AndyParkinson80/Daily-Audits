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

# Add Cloud SDK repo and key
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
  | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --dearmor | sudo tee /usr/share/keyrings/cloud.google.gpg > /dev/null

# Install gcloud
sudo apt-get update && sudo apt-get install -y google-cloud-sdk

echo "✅ gcloud installed:"
gcloud version

echo "🔐 Authenticating with GCP service account..."

if [ -z "$GCP" ]; then
  echo "⚠️  GCP_SERVICE_ACCOUNT_KEY_JSON is not set! Skipping auth."
else
  echo "$GCP" > /tmp/gcp-key.json
  gcloud auth activate-service-account --key-file=/tmp/gcp-key.json
  gcloud config set project api-integrations-412107  # TODO: Replace with actual project ID
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
  echo "✅ Authenticated and ADC configured."
fi
