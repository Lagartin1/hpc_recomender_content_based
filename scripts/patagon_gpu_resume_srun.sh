#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p L40                    # GPU partition
#SBATCH --gpus=1                  # Request 1 GPU
#SBATCH -c 8                      # Request 8 CPUs per task
#SBATCH --mem=128G                # Request 128 GB RAM

# OTHER PARAMS
#SBATCH -J reco_patagon_gpu_resume
#SBATCH -o reco_patagon_gpu_resume-%j.out
#SBATCH -e reco_patagon_gpu_resume-%j.err
#SBATCH --time=08:00:00

set -euo pipefail

IMAGE="lagartin1/hpc-recommender-patagon-cuda:cuda"
CONFIGS="${CONFIGS:-10000:50000,50000:100000,100000:200000,150000:250000,200000:300000,300000:400000,500000:500000}"
BLOCK_SIZE="${BLOCK_SIZE:-1000}"
MAX_INTERACTIONS="${MAX_INTERACTIONS:-10000000}"
MIN_RATING="${MIN_RATING:-1}"

mkdir -p results

if [ ! -f data/Electronics.jsonl.gz ] || [ ! -f data/meta_Electronics.jsonl.gz ]; then
  echo "ERROR: faltan datasets en ./data"
  echo "Esperado:"
  echo "  data/Electronics.jsonl.gz"
  echo "  data/meta_Electronics.jsonl.gz"
  exit 1
fi

pwd
date

echo "========================================"
echo "Benchmark GPU Patagon desde 10000:50000 hasta 500000:500000"
echo "Imagen: ${IMAGE}"
echo "Configs: ${CONFIGS}"
echo "Block size: ${BLOCK_SIZE}"
echo "Max interactions: ${MAX_INTERACTIONS}"
echo "Min rating: ${MIN_RATING}"
echo "Backends: torch_gpu, ray_cuda"
echo "========================================"

srun \
  --container-image="${IMAGE}" \
  --container-name="reco-patagon-gpu-resume" \
  --container-workdir=/app \
  --container-mounts="${PWD}/data:/app/data:ro,${PWD}/results:/app/results" \
  python3 /app/src/benchmark_amazon/run_amazon_patagon.py \
    --backends torch_gpu,ray_cuda \
    --configs "${CONFIGS}" \
    --max-interactions "${MAX_INTERACTIONS}" \
    --min-rating "${MIN_RATING}" \
    --block-size "${BLOCK_SIZE}" \
    --csv /app/results/amazon_patagon_cuda_resume_benchmark.csv \
    --output-dir /app/results

echo "========================================"
echo "Benchmark GPU resume terminado"
echo "Resultado: results/amazon_patagon_cuda_resume_benchmark.csv"
echo "========================================"
date
