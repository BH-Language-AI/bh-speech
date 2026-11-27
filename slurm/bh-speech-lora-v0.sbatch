#!/bin/bash
#SBATCH --job-name=bh-speech-v0
#SBATCH --account=project_465003647
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=02:50:00
#SBATCH --output=bh-speech-v0-%j.out
# BH-Speech v0 — whisper-large-v3 LoRA bf16 (ROCm, bez bitsandbytes).
# Upotreba: sbatch bh-speech-lora-v0.sbatch <job-dir-na-scratchu>
# (job-dir sadrzi train_bh_speech.py; izlaz: out/adapter + out/metrics.txt)
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch bh-speech-lora-v0.sbatch <job-dir>}"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export PYLIBS=/scratch/project_465003647/agentmujo/pylibs
export HF_HUB_CACHE=/scratch/project_465003647/agentmujo/hfcache
export SINGULARITYENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export SINGULARITYENV_PYTHONPATH="$PYLIBS"
export HF_DATASETS_CACHE=/scratch/project_465003647/agentmujo/hfds
export TMPDIR=/scratch/project_465003647/agentmujo/tmp
export TRITON_CACHE_DIR=/scratch/project_465003647/agentmujo/tmp/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/project_465003647/agentmujo/tmp/inductor
export XDG_CACHE_HOME=/scratch/project_465003647/agentmujo/tmp/xdg
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_TMPDIR="$TMPDIR"
mkdir -p "$PYLIBS" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$JOBDIR/out"
# Speech dodaci (dijeljeni PYLIBS; torch se NE dira).
# torchcodec MORA iza rm-garda (ime pocinje sa torch* pa bi ga guard obrisao):
# datasets>=5 trazi torchcodec za Audio decode (vidjeno: job 24044810).
srun -n1 singularity exec "$SIF" pip install -q --target="$PYLIBS" jiwer soundfile librosa \
  && rm -rf "$PYLIBS"/torch* "$PYLIBS"/nvidia* "$PYLIBS"/triton* "$PYLIBS"/bitsandbytes*
srun -n1 singularity exec "$SIF" pip install -q --target="$PYLIBS" torchcodec
# Guard: ROCm torch iz kontejnera.
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('ROCM-TORCH-OK', torch.__version__)"
cd "$JOBDIR"
srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
  python -u /job/train_bh_speech.py --out /job/out
echo TRAIN_DONE
