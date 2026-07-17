# Setting Up Ollama on AWS EC2

## Step 1: Launch EC2 Instance

### Via AWS Console:
1. Go to EC2 Dashboard
2. Click "Launch Instance"
3. Configure:
   - **Name**: `ollama-gpu-server`
   - **AMI**: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
   - **Instance type**: `g5.xlarge` (or g6.xlarge for cheaper)
   - **Key pair**: Create/select your SSH key
   - **Security Group**: 
     - Allow SSH (port 22) from your IP
     - Allow Custom TCP (port 11434) from your IP (for Ollama API)
   - **Storage**: 100 GB gp3 (for models)
4. Launch instance

### Save Instance Details:
```bash
# Note your instance public IP
INSTANCE_IP=<your-ec2-public-ip>
```

---

## Step 2: Connect and Install Ollama

```bash
# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@${INSTANCE_IP}

# Update system
sudo apt update && sudo apt upgrade -y

# Verify GPU
nvidia-smi

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama to listen on all interfaces (not just localhost)
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

# Reload and restart Ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Verify it's running
curl http://localhost:11434/api/tags
```

---

## Step 3: Pull Models

```bash
# Pull models you want to use
ollama pull llama3.3:70b
ollama pull llama3.1:70b  
ollama pull qwen2.5:72b
ollama pull mistral:7b

# List installed models
ollama list
```

---

## Step 4: Test from Local Machine

```bash
# From your Mac, test connection (replace with your instance IP)
curl http://${INSTANCE_IP}:11434/api/tags

# Should return list of models
```

---

## Step 5: Update Your .env File

Add the Ollama server URL to your `.env`:

```bash
# Add this line
OLLAMA_URL=http://<your-ec2-ip>:11434
```

---

## Step 6: Run Your RAG Benchmark

```bash
# From your Mac
conda activate galen
python run_ollama_model.py --model llama3.3:70b --ollama-url http://${INSTANCE_IP}:11434
```

---

## Cost Optimization Tips

### 1. Use Spot Instances (70% cheaper)
```bash
# When launching, choose "Request Spot instances"
# Set max price to ~$0.50/hr for g5.xlarge
```

### 2. Stop When Not Using
```bash
# Stop instance (doesn't delete, just pauses billing)
aws ec2 stop-instances --instance-ids i-xxxxx

# Start when needed
aws ec2 start-instances --instance-ids i-xxxxx
```

### 3. Use Instance Scheduler
Create a Lambda to auto-stop at night:
- Run 9am-6pm weekdays = $8/day vs $24/day

---

## Troubleshooting

### Ollama not accessible from outside:
```bash
# Check if port is open
sudo ufw status
sudo ufw allow 11434/tcp

# Check Ollama is listening on 0.0.0.0
sudo systemctl status ollama
sudo journalctl -u ollama -f
```

### GPU not detected:
```bash
# Reinstall NVIDIA drivers
sudo apt install -y nvidia-driver-535
sudo reboot
```

### Out of memory:
```bash
# Use smaller models or quantized versions
ollama pull llama3.1:70b-q4_0  # 4-bit quantized (uses less VRAM)
```

---

## Security Best Practices

1. **Restrict Security Group**: Only allow your IP
2. **Use VPC**: Put in private subnet with VPN access
3. **Enable CloudWatch**: Monitor costs
4. **Use IAM roles**: No hard-coded credentials
5. **Enable encryption**: For EBS volumes

---

## Alternative: Use AWS Batch + ECS

For batch processing (not interactive):
1. Use ECS with GPU containers
2. Run Ollama in Docker
3. Process all queries in batch
4. Auto-shutdown when done

This is cheaper for one-time benchmarks!
