#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --job-name=training
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=80        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --output=/mbz/users/haolong.jia/opt/logs/slurm_%x_%j.out
#SBATCH --error=/mbz/users/haolong.jia/opt/logs/slurm_%x_%j.err

module load cuda/12.4
source activate 
conda activate torchtitan
cd /mbz/users/haolong.jia/opt/torchtitan
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

NNODES=2
GPUS_PER_NODE=8
LOG_RANK=${LOG_RANK:-0}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
echo $SLURM_JOB_NODELIST
export LOGLEVEL=INFO

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --rdzv_id $RANDOM
    --rdzv_backend c10d
    --rdzv_endpoint $head_node_ip:29500
)

set -ex

# CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}
CONFIG_FILE="/mbz/users/haolong.jia/opt/torchtitan/train_configs/llama3_3b.toml"

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
srun torchrun ${DISTRIBUTED_ARGS[@]} \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    train.py --job.config_file ${CONFIG_FILE} $overrides