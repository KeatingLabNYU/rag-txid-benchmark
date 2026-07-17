#!/bin/bash
# Quick setup script for Ollama on AWS EC2
# Run this on your EC2 instance after launching

set -e

echo "=========================================="
echo "Ollama AWS Setup Script"
echo "=========================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Verify GPU
echo "🎮 Checking GPU..."
nvidia-smi || {
    echo "❌ GPU not detected! Make sure you're using a GPU instance (g5, g6, p3, etc.)"
    exit 1
}

# Install Ollama
echo "🦙 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama to accept external connections
echo "🔧 Configuring Ollama for external access..."
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

# Reload and restart
echo "🔄 Restarting Ollama service..."
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl enable ollama

# Wait for service to start
sleep 5

# Verify installation
echo "✅ Verifying Ollama installation..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running!"
else
    echo "❌ Ollama failed to start"
    sudo journalctl -u ollama -n 50
    exit 1
fi

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📍 Your Ollama server is at: http://${PUBLIC_IP}:11434"
echo ""
echo "🔐 Security Group Check:"
echo "   Make sure port 11434 is open in your EC2 Security Group!"
echo ""
echo "📥 Next Steps:"
echo "   1. Pull models:"
echo "      ollama pull llama3.3:70b"
echo "      ollama pull qwen2.5:72b"
echo ""
echo "   2. On your local machine, add to .env:"
echo "      OLLAMA_URL=http://${PUBLIC_IP}:11434"
echo ""
echo "   3. Test from local machine:"
echo "      curl http://${PUBLIC_IP}:11434/api/tags"
echo ""
echo "   4. Run benchmarks:"
echo "      python run_ollama_model.py --model llama3.3:70b"
echo ""
echo "=========================================="
