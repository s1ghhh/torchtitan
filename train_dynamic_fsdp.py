# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from datetime import timedelta

import torch

from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_tokenizer
from torchtitan.parallelisms import ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.train_spec import get_train_spec
from torchtitan.utils import device_module, device_type, import_module_from_path
from torchtitan.models.llama.Dynamic_model import DynamicTransformer

# @torch.no_grad
# def drop_layer(model, dropped_attn_list: list, dropped_mlp_list: list, device):
#     """
#     Drop some layers.

#     Args:
        

#     Returns:
        

#     """

#     # for layer_id in range(len(self.layers.values())):
#     #     if layer_id in dropped_attn_list:
#     #         self.layers[str(layer_id)].attention = nn.Identity().to(self.layers[str(layer_id)].attention.device)
#     #         self.layers[str(layer_id)].attention_norm = nn.Identity().to(self.layers[str(layer_id)].attention_norm.device)
#     #         self.layers[str(layer_id)].drop_type = self.layers[str(layer_id)].drop_type.replace("*", "")
#     #     if layer_id in dropped_mlp_list:
#     #         self.layers[str(layer_id)].feed_forward = nn.Identity().to(self.layers[str(layer_id)].feed_forward.device)
#     #         self.layers[str(layer_id)].ffn_norm = nn.Identity().to(self.layers[str(layer_id)].ffn_norm.device)
#     #         self.layers[str(layer_id)].drop_type = self.layers[str(layer_id)].drop_type.replace("#", "")

#     for layer_id in range(len(self.layers.values())):
#         if layer_id in dropped_attn_list:
#             self.layers[str(layer_id)]._checkpoint_wrapped_module.attention = nn.Identity().to(device)
#             self.layers[str(layer_id)]._checkpoint_wrapped_module.attention_norm = nn.Identity().to(device)
#             self.layers[str(layer_id)].drop_type = self.layers[str(layer_id)].drop_type.replace("*", "")
#         if layer_id in dropped_mlp_list:
#             self.layers[str(layer_id)]._checkpoint_wrapped_module.feed_forward = nn.Identity().to(device)
#             self.layers[str(layer_id)]._checkpoint_wrapped_module.ffn_norm = nn.Identity().to(device)
#             self.layers[str(layer_id)].drop_type = self.layers[str(layer_id)].drop_type.replace("#", "")

# def rebuild_adamw_optimizers_container(model, optimizers_container):
#     new_params = list(model.parameters())
#     new_optimizers = []
    
#     # 获取原 `OptimizersContainer` 的 `optimizer_kwargs` 和 `name`
#     optimizer_kwargs = optimizers_container.optimizer_kwargs
#     name = optimizers_container.name

#     # 遍历原 container 中的每个优化器（假设都是 AdamW）
#     for optimizer in optimizers_container:
#         # 从参数组中提取超参数（这里假设只有一个参数组）
#         old_pg = optimizer.param_groups[0]
#         lr = old_pg.get('lr', 1e-3)
#         betas = old_pg.get('betas', (0.9, 0.999))
#         eps = old_pg.get('eps', 1e-8)
#         weight_decay = old_pg.get('weight_decay', 0)
#         amsgrad = old_pg.get('amsgrad', False)
        
#         # 创建新的 AdamW 优化器（使用新的模型参数）
#         new_optimizer = torch.optim.AdamW(new_params, lr=lr, betas=betas,
#                                             eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        
#         # 将旧优化器中与新参数对应的状态转移到新的优化器中
#         new_optimizer_state = {}
#         for param in new_params:
#             for old_param, state in optimizer.state.items():
#                 if param is old_param:
#                     new_optimizer_state[param] = state
#                     break
        
#         new_optimizer.state = new_optimizer_state
#         new_optimizers.append(new_optimizer)

#     # 以新的优化器列表构造一个新的 OptimizersContainer
#     new_optimizers_container = type(optimizers_container)(new_optimizers, optimizer_kwargs, name)
    
#     return new_optimizers_container

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        import_module_from_path(job_config.experimental.custom_model_path)

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[train_spec.name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)
    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # build model (using meta init)
    model_cls = train_spec.cls
    model_config = train_spec.config[job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(
        f"Building {train_spec.name} {job_config.model.flavor} with {model_config}"
    )
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    model_param_count = utils.get_num_params(model)
    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {train_spec.name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    # TODO: compiling loss function causes CUDA errors, turning off for now
    # if job_config.training.compile:
    #     loss_fn = torch.compile(loss_fn)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None

    # if torch.distributed.get_rank() == 0:
    #     import debugpy
    #     try:
    #         debugpy.listen(8201)
    #         print("Waiting for debugger attach")
    #         debugpy.wait_for_client()
    #     except Exception as e:
    #         print(e)

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.init_weights(buffer_device=buffer_device)
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        # import inspect
        # logger.info(inspect.getmro(type(model)))  

        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
        model.train()
        # logger.info(inspect.getmro(type(model)))  

        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed checkpoint using a single device, to disable sharding"
        assert (
            job_config.checkpoint.enable_checkpoint
        ), "Must enable checkpointing when creating a seed checkpoint"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)
    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree * job_config.training.gradient_accumulation_steps}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # if torch.distributed.get_rank() == 0 and train_state.step == 1:
            #     import debugpy
            #     try:
            #         debugpy.listen(8201)
            #         print("Waiting for debugger attach")
            #         debugpy.wait_for_client()
            #     except Exception as e:
            #         print(e)

            optimizers.zero_grad()
            for micro_step in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                input_ids, labels = batch
                ntokens_since_last_log += labels.numel()
                data_loading_times.append(time.perf_counter() - data_load_start)

                input_ids = input_ids.to(device_type)
                labels = labels.to(device_type)

                # apply context parallelism if cp is enabled
                # ensure CP handles the separate freqs_cis buffer for each pp stage
                optional_context_parallel_ctx = (
                    utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels] + [m.freqs_cis for m in model_parts],
                        cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                        cp_no_restore_buffers={input_ids, labels},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                if parallel_dims.pp_enabled:
                    # Pipeline Parallel forward / backward inside step() call
                    with train_context(optional_context_parallel_ctx):
                        targets, losses = (labels, []) if has_last_stage else (None, None)
                        if has_first_stage:
                            pp_schedule.step(input_ids, target=targets, losses=losses)
                        else:
                            pp_schedule.step(target=targets, losses=losses)

                    # accumulate losses across pipeline microbatches
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                    
                    # GS: may raise value
                    loss = loss / job_config.training.gradient_accumulation_steps
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        pred, _, _ = model(input_ids)
                        loss = loss_fn(pred, labels)
                        # pred.shape=(bs, seq_len, vocab_size)
                        # need to free to before bwd to avoid peaking memory
                        del pred
                        loss = loss / job_config.training.gradient_accumulation_steps
                        loss.backward()

            # clip gradients
            utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(loss, world_mesh["dp_cp"]),
                        utils.dist_max(loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss * job_config.training.gradient_accumulation_steps,
                    "loss_metrics/global_max_loss": global_max_loss * job_config.training.gradient_accumulation_steps,
                    "throughput(tps)": tps,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss * job_config.training.gradient_accumulation_steps:7.4f}  "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            # if train_state.step % 4 == 0:
            #     print()
            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

            if train_state.step % job_config.dropping.drop_freq == 0 and isinstance(model, DynamicTransformer):
                with torch.no_grad():
                    data_loader_sims = build_hf_data_loader(
                        job_config.dropping.dataset,
                        job_config.dropping.dataset_path,
                        tokenizer,
                        job_config.dropping.batch_size,
                        job_config.dropping.seq_len,
                        dp_degree,
                        dp_rank,
                    )
                    sim_data_iterator = iter(data_loader_sims)
                    sims_attn_sum = [0 for _ in range(model.n_layers)]
                    sims_mlp_sum = [0 for _ in range(model.n_layers)]
                    logger.info("Start profiling cos_sims")
                    for sim_step in range(job_config.dropping.macro_steps):
                        batch = next(sim_data_iterator)
                        input_ids, labels = batch
                        input_ids = input_ids.to(device_type)
                        output, sims_attn, sims_mlp = model(input_ids, layer_sim_type="*")
                        del output
                        sims_attn_sum = list(map(lambda a, b: a + b, sims_attn_sum, sims_attn))
                        sims_mlp_sum = list(map(lambda a, b: a + b, sims_mlp_sum, sims_mlp))

                    logger.info("Finish profiling cos_sims")
                    sims_attn_sum = [x / job_config.dropping.macro_steps for x in sims_attn_sum]
                    sims_mlp_sum = [x / job_config.dropping.macro_steps for x in sims_mlp_sum]


                    attn_filtered_indices = [(i, val) for i, val in enumerate(sims_attn_sum) if val > job_config.dropping.sim_threshold]
                    attn_top_indices = [i for i, _ in sorted(attn_filtered_indices, key=lambda x: x[1], reverse=True)[:job_config.dropping.num_each]]
                    mlp_filtered_indices = [(i, val) for i, val in enumerate(sims_mlp_sum) if val > job_config.dropping.sim_threshold]
                    mlp_top_indices = [i for i, _ in sorted(mlp_filtered_indices, key=lambda x: x[1], reverse=True)[:job_config.dropping.num_each]]

                    logger.info(f"attn_top_indices: {attn_top_indices} \n\nmlp_top_indices: {mlp_top_indices}")

                    logger.info("Start dropping modules")
                    model.drop_layer(attn_top_indices, mlp_top_indices, device)
                    logger.info("Finish dropping modules")

                    # optimizers = rebuild_adamw_optimizers_container(model, optimizers)

                    # logger.info("Start removing redundant optimizer state")

                    # new_params = list(model.parameters())

                    # # 获取优化器的现有参数
                    # for optimizer in optimizers.optimizers:
                    #     existing_params = optimizer.param_groups[0]['params']

                    #     # 确保 new_params 里的参数类型与 optimizer.param_groups 一致
                    #     new_params = [p if isinstance(p, type(existing_params[0])) else existing_params[0].to(p.device) for p in new_params]

                    #     # 只保留仍然有效的参数
                    #     optimizer.param_groups[0]['params'] = [p for p in new_params if any(torch.equal(p, ep) for ep in existing_params)]

                    #     # 清理 state
                    #     optimizer_state = optimizer.state
                    #     optimizer.state = {param: optimizer_state[param] for param in optimizer.param_groups[0]['params'] if param in optimizer_state}

                    # logger.info("Finish removing redundant optimizer state")


                    # logger.info("Checking optimizer parameters after dropping modules...")
                    # for optimizer in optimizers.optimizers:
                    #     param_ids = {id(p) for p in optimizer.param_groups[0]['params']}
                    #     state_ids = set(optimizer.state.keys())

                    #     missing_params = state_ids - param_ids
                    #     if missing_params:
                    #         logger.warning(f"Optimizer state contains {len(missing_params)} missing params!")

                    #     orphan_states = param_ids - state_ids
                    #     if orphan_states:
                    #         logger.warning(f"Optimizer param_groups contain {len(orphan_states)} orphan params!")



                    logger.info("Start removing redudant optimizer state")
                    # new_params = list(model.parameters())
                    logger.info(f"s1ghhhh: {type(optimizers)}")
                    for optimizer in optimizers:

                        optimizer_state = optimizer.state
                        new_optimizer_state = {}

                        lost_param_list = []
                        
                        fuck_list = list(optimizer_state.items())
                        idx = 0
                        for name, param in model.state_dict().items():
                            
                            for kkk, vvv in fuck_list:
                                # if torch.equal(param, kkk._local_tensor):
                                # if isinstance(kkk, torch.distributed._tensor.DTensor):
                                #     kkk = kkk.to_local()
                                # if isinstance(param, torch.distributed._tensor.DTensor):
                                #     param = param.to_local()
                                # print(type(kkk))
                                # print(type(param))
                                if torch.equal(param.to_local() if isinstance(param, torch.distributed._tensor.DTensor) else param, kkk.to_local() if isinstance(kkk, torch.distributed._tensor.DTensor) else kkk):
                                    new_optimizer_state[idx] = optimizer_state[kkk]
                                    idx+=1
                                    break

                                    # logger.info(f"{name} {param}")
                                    # lost_param_list.append(name)
                        torch.distributed.barrier()
                        for name, param in model.state_dict().items():
                            for shit in lost_param_list:
                                try:
                                    if torch.equal(param.to_local(), shit):
                                        logger.info(name)
                                except:
                                    logger.info(f"666 {name}")
                        torch.distributed.barrier()
                    param_groups = optimizers.optimizers[0].state_dict()["param_groups"]
                    param_groups[0]["params"] = [i for i in range(len(new_optimizer_state))]
                    state_dict = {"state": new_optimizer_state, "param_groups": param_groups}
                    # optimizer.state = new_optimizer_state
                    # optimizers.optimizers[0].load_state_dict(state_dict)
                # optimizers.load_state_dict( optimizers.optimizers[0].state_dict())
                new_optimizers = train_spec.build_optimizers_fn([model], job_config)
                new_optimizers.optimizers[0].load_state_dict(state_dict)
                shit = new_optimizers.state_dict()
                optimizers = new_optimizers
                checkpoint.states["optimizer"] = optimizers
                torch.distributed.barrier()
                logger.info("Finish removing redudant optimizer state")

            optimizers.zero_grad()



                
    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()