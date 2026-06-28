#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p cpu                    # CPU partition
#SBATCH -c 8                      # Request 8 CPUs per task
#SBATCH --mem=128G                # Request 128 GB RAM

# OTHER PARAMS
#SBATCH -J reco_patagon_cpu_resume
#SBATCH -o reco_patagon_cpu_resume-%j.out
#SBATCH -e reco_patagon_cpu_resume-%j.err
#SBATCH --time=08:00:00

set -euo pipefail

IMAGE="lagartin1/hpc-recommender-patagon:cpu"
CONFIGS="${CONFIGS:-150000:250000,200000:300000,300000:400000,500000:500000}"
BLOCK_SIZE="${BLOCK_SIZE:-250}"
RAY_MAX_IN_FLIGHT="${RAY_MAX_IN_FLIGHT:-2}"

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
echo "Benchmark CPU Patagon desde 150000:250000 hasta 500000:500000"
echo "Imagen: ${IMAGE}"
echo "Configs: ${CONFIGS}"
echo "Block size: ${BLOCK_SIZE}"
echo "========================================"

echo "========================================"
echo "Backend: numpy"
echo "========================================"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

srun \
  --container-image="${IMAGE}" \
  --container-name="reco-patagon-resume-numpy" \
  --container-workdir=/app \
  --container-mounts="${PWD}/data:/app/data:ro,${PWD}/results:/app/results" \
  python /app/src/benchmark_amazon/run_amazon_patagon.py \
    --backends numpy \
    --configs "${CONFIGS}" \
    --block-size "${BLOCK_SIZE}" \
    --csv /app/results/amazon_patagon_numpy_resume_benchmark.csv \
    --output-dir /app/results

echo "========================================"
echo "Backend: ray_cpu"
echo "========================================"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

srun \
  --container-image="${IMAGE}" \
  --container-name="reco-patagon-resume-ray" \
  --container-workdir=/app \
  --container-mounts="${PWD}/data:/app/data:ro,${PWD}/results:/app/results" \
  python /app/src/benchmark_amazon/run_amazon_patagon.py \
    --backends ray_cpu \
    --configs "${CONFIGS}" \
    --block-size "${BLOCK_SIZE}" \
    --ray-max-in-flight "${RAY_MAX_IN_FLIGHT}" \
    --csv /app/results/amazon_patagon_ray_cpu_resume_benchmark.csv \
    --output-dir /app/results

echo "========================================"
echo "Benchmark CPU resume terminado"
echo "Resultados:"
echo "  results/amazon_patagon_numpy_resume_benchmark.csv"
echo "  results/amazon_patagon_ray_cpu_resume_benchmark.csv"
echo "========================================"
date
