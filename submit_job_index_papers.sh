#!/bin/bash
#SBATCH --job-name=rag-benchmark
#SBATCH --output=benchmark_out_%j.log
#SBATCH --error=benchmark_err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G              # Request 64GB RAM (Plenty for BGE-M3 + Reranker)
#SBATCH --partition=a100_short # Request GPU partition
#SBATCH --gres=gpu:a100:1      # Request 1 A100 GPU
#SBATCH --time=02:00:00        # 2 hour limit

# 1. Load System Modules
module load cuda/12.9        # Load CUDA (Adjust version if needed for your HPC)

# 2. Diagnostics (Check if GPU is actually visible)
echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU Status:"
nvidia-smi
echo "------------------------------------------------"

# 3. Activate Virtual Environment
# (Assumes you already ran the install step interactively)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated."
else
    echo "❌ Error: 'venv' directory not found. Please run installation first."
    exit 1
fi

# 4. Check for API Keys
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file missing! Cannot authenticate with AWS."
    exit 1
fi

# 5. Run the Benchmark
echo "🚀 Starting Benchmark..."
python run_benchmark.py
