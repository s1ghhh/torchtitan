# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama.model import Transformer, TransformerModelArgs
from torchtitan.models.llama.Dropped_model_init import DroppedTransformer, DroppedTransformerModelArgs
from torchtitan.models.llama.Dynamic_model import DynamicTransformer, DynamicTransformerModelArgs
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.train_spec import register_train_spec, TrainSpec

from .parallelize_llama import parallelize_llama
from .pipeline_llama import pipeline_llama

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "DroppedTransformerModelArgs",
    "DynamicTransformerModelArgs",
    "Transformer",
    "DroppedTransformer",
    "DynamicTransformer",
    "llama3_configs",
]


llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=8, n_heads=16, rope_theta=500000
    ),
    "3B": TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=1024,
        rope_theta=200000,
        max_seq_len=4096,
    ),
    "3B_dropped_init": DroppedTransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=1024,
        rope_theta=200000,
        max_seq_len=4096,
        drop_list=['*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#', '*#', '#'],
    ),
    "3B_dynamic": DynamicTransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=8192,
        multiple_of=1024,
        rope_theta=200000,
        max_seq_len=4096,
    ),
    "3B_dynamic_debug": DynamicTransformerModelArgs(
        dim=512,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        ffn_hidden_size=1024,
        multiple_of=1024,
        rope_theta=200000,
        max_seq_len=1024,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


register_train_spec(
    TrainSpec(
        name="llama3",
        cls=Transformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )
)

register_train_spec(
    TrainSpec(
        name="dropped_llama3",
        cls=DroppedTransformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )
)

register_train_spec(
    TrainSpec(
        name="dynamic_llama3",
        cls=DynamicTransformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
    )
)