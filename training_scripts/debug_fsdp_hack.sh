#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=36000  # 1 hour timeout

export WANDB_MODE=offline

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
export CUDA_VISIBLE_DEVICES="3,6"
NGPU=${NGPU:-"2"}
LOG_RANK=${LOG_RANK:-0}
TOML_NAME=llama3_1.5b_dynamic_hack_debug
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/$TOML_NAME.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# rm -r /workspace/0215_opt/torchtitan/outputs_llama3_1.5b_dynamic_half_hack_debug

export TORCH_DISTRIBUTED_DEBUG=INFO

n=4 # `dropping.num_max // dropping.num_each + 1`, when dropping.sim_threshold is much low than 1.0, otherwise it should be `training.steps // checkpoint.interval + 1`

export WANDB_API_KEY=""
export WANDB_PROJECT="optimal-training-strategy"
export WANDB_ENTITY="torchtitan-opt"

for ((i=1; i<=n; i++)); do
    export WANDB_RUN_NAME="${TOML_NAME}_round_${i}"
    echo "dropping round $i"
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
        train_dynamic_reinit.py --job.config_file ${CONFIG_FILE} $overrides
done



