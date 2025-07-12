# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is modified to convert a single, non-sharded checkpoint file
# into the Hugging Face Llama format.

import argparse
import gc
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

# --- Model Configuration ---
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # Needs to be set
    ffn_hidden_size: Optional[int] = None # To be set: Calculated or explicit intermediate size
    multiple_of: int = 256  # Not directly used by HF LlamaConfig if ffn_hidden_size is set
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 2048

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        # Ensure ffn_hidden_size is set if not provided explicitly
        # (Example calculation, adjust if your model uses a different formula)
        if self.ffn_hidden_size is None:
             # Default Llama calculation based on dim
             hidden_dim = int(2 * self.dim / 3)
             # Ensure it's a multiple of multiple_of
             self.ffn_hidden_size = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
             warnings.warn(f"ffn_hidden_size not explicitly provided, calculated to: {self.ffn_hidden_size}")


# Define your model configurations here
# Ensure 'norm_eps' and 'ffn_hidden_size' are correct for your model
model_configs = {
    "3B": ModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_hidden_size=8192,  # Use the explicitly provided value
        multiple_of=1024,      # Keep for reference, ffn_hidden_size takes precedence
        rope_theta=200000,
        max_seq_len=4096,
        norm_eps=1e-5,         # Assuming standard Llama norm eps, adjust if needed
        # vocab_size will be set by --vocab_size argument
    ),
    # Add other configurations here if needed:
    # "another_flavor": ModelArgs(dim=..., n_layers=..., ...)
}

# --- Constants and Helper Functions ---

# Context length mapping (can be used as fallback or reference)
CONTEXT_LENGTH_FOR_VERSION = {"Guard-3": 131072, "3.2": 131072, "3.1": 131072, "3": 8192, "2": 4096, "1": 2048}

def is_llama_3(version):
    # Helper to check if version indicates Llama 3 specific settings
    return str(version).startswith("3") or str(version).lower() == "guard-3"

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f, indent=4) # Add indent for readability

# Permute function - might not be needed for non-sharded checkpoints depending on format
# def permute(w, n_heads, dim1, dim2):
#     return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

# --- Core Conversion Logic ---

def write_model(
    model_path: str,
    input_base_path: str,
    model_cfg: ModelArgs,
    steps: int,
    safe_serialization: bool = True,
    llama_version: str = "3", # Provide a default or ensure it's passed
    vocab_size: Optional[int] = None,
    instruct: bool = False,
    push_to_hub: bool = False,
):
    """
    Converts a single non-sharded checkpoint file to Hugging Face Llama format.

    Args:
        model_path: Path to save the converted HF model.
        input_base_path: Directory containing the input checkpoint file.
        model_cfg: ModelArgs object with the model architecture configuration.
        steps: Step number identifying the checkpoint file (e.g., for hf-ckpt--step-{steps}.pth).
        safe_serialization: Whether to save using safetensors.
        llama_version: String identifying the Llama base version (e.g., "1", "2", "3", "3.1").
        vocab_size: The vocabulary size of the model.
        instruct: Whether the model is an instruction-tuned variant.
        push_to_hub: Whether to push the converted model to the Hugging Face Hub.
    """
    print("Converting the model (assuming non-sharded input).")

    if vocab_size is None:
        raise ValueError("vocab_size must be provided when not converting tokenizer.")

    # --- Configuration Derivation ---
    n_layers = model_cfg.n_layers
    n_heads = model_cfg.n_heads
    dim = model_cfg.dim
    dims_per_head = dim // n_heads
    base = model_cfg.rope_theta
    # Calculate inverse frequencies for RoPE
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # Determine max_position_embeddings, prioritizing config, then version
    max_position_embeddings = model_cfg.max_seq_len
    print(f"Using max_position_embeddings from model config: {max_position_embeddings}")
    if max_position_embeddings is None:
        # Fallback if not set in config (should be set by ModelArgs now)
        warnings.warn("max_seq_len not found in model_cfg, attempting fallback based on llama_version.")
        if base > 10000.0 and not is_llama_3(llama_version):
             max_position_embeddings = 16384 # Llama 2 CodeLlama-like?
        else:
            max_position_embeddings = CONTEXT_LENGTH_FOR_VERSION.get(str(llama_version), 8192) # Defaulting
        print(f"Using fallback max_position_embeddings: {max_position_embeddings}")


    num_key_value_heads = model_cfg.n_kv_heads
    if num_key_value_heads is None: # Should be set by ModelArgs __post_init__
        num_key_value_heads = n_heads

    # --- Load Single Checkpoint ---
    checkpoint_filename = f"step-{steps}.pt" # Adapt this if your naming scheme differs
    checkpoint_path = os.path.join(input_base_path, checkpoint_filename)

    if not os.path.exists(checkpoint_path):
         # Example fallback: Check for common consolidated names if step-based name fails
         fallback_filenames = ["consolidated.00.pth", "pytorch_model.bin"]
         found_fallback = False
         for fname in fallback_filenames:
             alt_path = os.path.join(input_base_path, fname)
             if os.path.exists(alt_path):
                 checkpoint_path = alt_path
                 found_fallback = True
                 print(f"Warning: Primary checkpoint '{checkpoint_filename}' not found. Using fallback: '{fname}'")
                 break
         if not found_fallback:
            raise FileNotFoundError(
                f"Checkpoint file not found. Tried primary name '{checkpoint_filename}' and fallbacks {fallback_filenames} in directory '{input_base_path}'"
            )

    print(f"Loading weights from single checkpoint: {checkpoint_path}")
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint file '{checkpoint_path}': {e}")
        raise

    # Handle potential nesting (e.g., weights under 'model' or 'state_dict' key)
    if 'model' in loaded and isinstance(loaded['model'], dict):
        print("Checkpoint seems nested under 'model' key. Using weights from loaded['model'].")
        loaded = loaded['model']
    elif 'state_dict' in loaded and isinstance(loaded['state_dict'], dict):
        print("Checkpoint seems nested under 'state_dict' key. Using weights from loaded['state_dict'].")
        loaded = loaded['state_dict']


    param_count = 0
    index_dict = {"weight_map": {}}

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmp_model_path:
        print(f"Using temporary directory for conversion: {tmp_model_path}")

        # --- Process Layers ---
        for layer_i in range(n_layers):
            print(f"Converting layer {layer_i}")
            # Standard Hugging Face naming convention for sharded checkpoints
            filename = f"pytorch_model-{layer_i + 1:05d}-of-{n_layers + 1:05d}.bin"
            state_dict = {}

            # --- Define expected keys in the loaded checkpoint ---
            # Adapt these key patterns if your checkpoint uses different names
            q_proj_key = f"layers.{layer_i}.attention.wq.weight"
            k_proj_key = f"layers.{layer_i}.attention.wk.weight"
            v_proj_key = f"layers.{layer_i}.attention.wv.weight"
            o_proj_key = f"layers.{layer_i}.attention.wo.weight"
            attn_norm_key = f"layers.{layer_i}.attention_norm.weight" # Input LayerNorm

            gate_proj_key = f"layers.{layer_i}.feed_forward.w1.weight" # MLP gate proj
            down_proj_key = f"layers.{layer_i}.feed_forward.w2.weight" # MLP down proj
            up_proj_key = f"layers.{layer_i}.feed_forward.w3.weight"   # MLP up proj
            ffn_norm_key = f"layers.{layer_i}.ffn_norm.weight"      # Post Attention LayerNorm

            # --- Map weights: Check for existence and add to state_dict ---
            # Attention weights
            if q_proj_key in loaded:
                state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = loaded[q_proj_key]
            else: warnings.warn(f"Weight not found in checkpoint: {q_proj_key}")
            if k_proj_key in loaded:
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = loaded[k_proj_key]
            else: warnings.warn(f"Weight not found in checkpoint: {k_proj_key}")
            if v_proj_key in loaded:
                state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = loaded[v_proj_key]
            else: warnings.warn(f"Weight not found in checkpoint: {v_proj_key}")
            if o_proj_key in loaded:
                state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = loaded[o_proj_key]
            else: warnings.warn(f"Weight not found in checkpoint: {o_proj_key}")

            # Attention Norm
            if attn_norm_key in loaded:
                state_dict[f"model.layers.{layer_i}.input_layernorm.weight"] = loaded[attn_norm_key]
            else: warnings.warn(f"Weight not found in checkpoint: {attn_norm_key}")

            # MLP weights
            if gate_proj_key in loaded:
                state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = loaded[gate_proj_key]
            else: warnings.warn(f"Weight not found in checkpoint: {gate_proj_key}")
            if down_proj_key in loaded:
                state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = loaded[down_proj_key]
            else: warnings.warn(f"Weight not found in checkpoint: {down_proj_key}")
            if up_proj_key in loaded:
                state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = loaded[up_proj_key]
            else: warnings.warn(f"Weight not found in checkpoint: {up_proj_key}")

            # FFN Norm
            if ffn_norm_key in loaded:
                state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = loaded[ffn_norm_key]
            else: warnings.warn(f"Weight not found in checkpoint: {ffn_norm_key}")


            # Add RoPE inverse frequencies (calculated, not loaded)
            state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

            # --- Save layer weights and update index ---
            if not state_dict:
                warnings.warn(f"No weights found or mapped for layer {layer_i}, skipping save for {filename}")
                continue # Skip saving if no weights were found for this layer

            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()

            print(f"Saving weights for layer {layer_i} to {filename}")
            torch.save(state_dict, os.path.join(tmp_model_path, filename))
            # It's good practice to delete the state dict for the layer to free memory
            del state_dict
            gc.collect()

        # --- Process Non-Layer Weights (Embeddings, Final Norm, LM Head) ---
        print("Converting non-layer weights (tok_embeddings, norm, lm_head)")
        # Use the next file index for the final weights
        filename = f"pytorch_model-{n_layers + 1:05d}-of-{n_layers + 1:05d}.bin"
        state_dict = {}

        # Define expected keys for final layers
        tok_embed_key = "tok_embeddings.weight"
        norm_key = "norm.weight"
        lm_head_key = "output.weight"

        # Alternative keys often used in HF models (might be present if converting HF->HF)
        alt_tok_embed_key = "model.embed_tokens.weight"
        alt_norm_key = "model.norm.weight"
        alt_lm_head_key = "lm_head.weight"

        # Map final weights, checking primary and alternative keys
        if tok_embed_key in loaded:
            state_dict["model.embed_tokens.weight"] = loaded[tok_embed_key]
        elif alt_tok_embed_key in loaded:
             state_dict["model.embed_tokens.weight"] = loaded[alt_tok_embed_key]
        else:
            warnings.warn(f"Token embedding weight not found (tried {tok_embed_key}, {alt_tok_embed_key})")

        if norm_key in loaded:
            state_dict["model.norm.weight"] = loaded[norm_key]
        elif alt_norm_key in loaded:
             state_dict["model.norm.weight"] = loaded[alt_norm_key]
        else:
             warnings.warn(f"Final norm weight not found (tried {norm_key}, {alt_norm_key})")

        if lm_head_key in loaded:
            state_dict["lm_head.weight"] = loaded[lm_head_key]
        elif alt_lm_head_key in loaded:
             state_dict["lm_head.weight"] = loaded[alt_lm_head_key]
        else:
            warnings.warn(f"LM head weight not found (tried {lm_head_key}, {alt_lm_head_key})")

        # Save final weights and update index
        if not state_dict:
            warnings.warn(f"No final weights (embeddings, norm, lm_head) found or mapped, skipping save for {filename}")
        else:
            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            print(f"Saving final weights to {filename}")
            torch.save(state_dict, os.path.join(tmp_model_path, filename))

        # --- Write Configs ---
        # Weight map index
        index_dict["metadata"] = {"total_size": param_count * 2} # Assuming float16/bfloat16 (2 bytes)
        write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

        # Determine BOS/EOS tokens based on Llama version (adjust if needed)
        if is_llama_3(llama_version):
            bos_token_id = 128000
            # Use <|eot_id|> (128009) for instruct, <|end_of_text|> (128001) otherwise
            eos_token_id = 128009 if instruct else 128001
            # Generation config might use a list for instruct models: [128001, 128009]
            gen_eos_token_id = [128001, 128009] if instruct else 128001
        else: # Llama 1 & 2
            bos_token_id = 1
            eos_token_id = 2
            gen_eos_token_id = 2

        # Rope Scaling (specific to Llama 3.1+)
        rope_scaling = None
        if llama_version in ["3.1", "3.2", "Guard-3"]:
             factor = 32.0 if llama_version == "3.2" else 8.0 # Example factor logic
             rope_scaling = {
                 "type": "llama3", # Use "llama3" type for newer scaling
                 "factor": factor,
                 "low_freq_factor": 1.0,
                 "high_freq_factor": 4.0,
                 "original_max_position_embeddings": 8192, # Base length before scaling
             }

        # Create LlamaConfig using parameters from model_cfg
        # Determine torch_dtype (prefer bfloat16 if available and appropriate)
        # Check if loaded tensors provide dtype info, otherwise default
        try:
            # Peek at a tensor's dtype if possible
            sample_tensor = next(iter(loaded.values()))
            loaded_dtype = sample_tensor.dtype
            if loaded_dtype in [torch.float16, torch.bfloat16]:
                hf_torch_dtype = str(loaded_dtype).split('.')[-1] # "float16" or "bfloat16"
            else:
                hf_torch_dtype = "bfloat16" # Default to bfloat16 if unsure or loaded as float32
                warnings.warn(f"Loaded tensor dtype is {loaded_dtype}. Defaulting HF config dtype to bfloat16.")
        except Exception:
            hf_torch_dtype = "bfloat16" # Fallback default
            warnings.warn("Could not determine loaded tensor dtype. Defaulting HF config dtype to bfloat16.")
        print(f"Setting HF model config torch_dtype to: {hf_torch_dtype}")

        # LlamaConfig uses ffn_hidden_size directly as intermediate_size
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=model_cfg.dim,
            intermediate_size=model_cfg.ffn_hidden_size,
            num_attention_heads=model_cfg.n_heads,
            num_hidden_layers=model_cfg.n_layers,
            rms_norm_eps=model_cfg.norm_eps,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            rope_theta=model_cfg.rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id, # Use single EOS for main config
            torch_dtype=hf_torch_dtype,
            # tie_word_embeddings=True if llama_version == "3.2" else False, # Example: check if needed
        )
        print("Saving model config.json")
        config.save_pretrained(tmp_model_path)

        # Create and save GenerationConfig
        generation_config = GenerationConfig(
            # Common generation parameters, adjust as needed
            max_length=max_position_embeddings, # Or a reasonable default like 2048
            bos_token_id=bos_token_id,
            eos_token_id=gen_eos_token_id, # Can be a list for instruct
            pad_token_id=config.pad_token_id if config.pad_token_id is not None else eos_token_id, # Common practice: pad = eos
            # Add other relevant generation params if known (do_sample, temperature, top_p, etc.)
            # Example:
            # do_sample=True,
            # temperature=0.6,
            # top_p=0.9,
        )
        print("Saving generation_config.json")
        generation_config.save_pretrained(tmp_model_path)

        # --- Final Model Loading & Saving ---
        # Make space so we can load the model properly now.
        del state_dict # Ensure final state_dict is deleted
        del loaded
        gc.collect()

        print("Reloading the model from converted weights for final save.")
        try:
            # Load using AutoModel for flexibility or LlamaForCausalLM directly
            # Use low_cpu_mem_usage for potentially large models
            model = LlamaForCausalLM.from_pretrained(
                tmp_model_path,
                torch_dtype=hf_torch_dtype, # Use the determined dtype
                low_cpu_mem_usage=True
            )

            # Avoid saving temporary path info in the final config
            if hasattr(model.config, '_name_or_path'):
                 del model.config._name_or_path

            print(f"Saving final model to: {model_path}")
            if push_to_hub:
                print(f"Pushing to Hub repository: {model_path}")
                # Ensure you are logged in (`huggingface-cli login`)
                model.push_to_hub(
                    repo_id=model_path,
                    safe_serialization=safe_serialization,
                    private=True, # Assuming private push, change if needed
                    use_temp_dir=False # Use the target dir directly
                    )
            else:
                print("Saving model to disk.")
                model.save_pretrained(model_path, safe_serialization=safe_serialization)
            print("Model conversion and saving complete.")

        except Exception as e:
            print(f"\n!!! Error during final model loading or saving: {e}")
            print("Please check the generated files in the temporary directory:")
            print(f"  {tmp_model_path}")
            print("Verify config.json, generation_config.json, and pytorch_model.bin.index.json")
            print("Also check the individual weight files (pytorch_model-*.bin) for issues.")
            raise

# --- Tokenizer Conversion (Optional - currently commented out) ---
# If you need tokenizer conversion, uncomment and adapt the following:
# from tokenizers import AddedToken, processors
# from transformers import LlamaTokenizer, PreTrainedTokenizerFast
# from transformers.convert_slow_tokenizer import TikTokenConverter
# try:
#     from transformers import LlamaTokenizerFast
# except ImportError as e:
#     warnings.warn(f"{e}\nUsing slow LlamaTokenizer.")
#     LlamaTokenizerFast = None # Fallback to slow tokenizer
#
# # Add Llama3Converter class here if needed (from original script)
# # class Llama3Converter(TikTokenConverter): ...
#
# def write_tokenizer(...):
#     # Add write_tokenizer function here if needed (from original script)
#     ...

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Convert a single non-sharded Llama checkpoint to Hugging Face format."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the NON-SHARDED Llama checkpoint file (e.g., hf-ckpt--step-XXX.pth or consolidated.00.pth).",
    )
    parser.add_argument(
        "--tt_flavor",
        required=True,
        help="Identifier for the model configuration (must be a key in `model_configs` dictionary, e.g., '3B').",
    )
    parser.add_argument(
        "--steps",
        required=True,
        type=int,
        help="Step number used in the primary checkpoint filename (e.g., for hf-ckpt--step-{steps}.pth). Used to find the file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write the converted Hugging Face model.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True, # Make required since tokenizer conversion is commented out
        help="Vocabulary size of the model."
    )
    # Optional Tokenizer Path (if uncommenting tokenizer code)
    # parser.add_argument(
    #     "--tokenizer_path",
    #     default=None,
    #     help="Path to the input tokenizer file (e.g., tokenizer.model or tokenizer.json). If provided, the tokenizer will also be converted.",
    # )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Whether to push the model to the Hugging Face Hub at `output_dir` (repo ID) instead of saving locally.",
    )
    parser.add_argument(
        "--safe_serialization",
        action=argparse.BooleanOptionalAction, # Allows --safe_serialization / --no-safe_serialization
        default=True,
        help="Whether to save using `safetensors` (default: True).",
    )
    parser.add_argument(
        "--llama_version",
        choices=["1", "2", "3", "3.1", "3.2", "Guard-3"],
        default="3", # Sensible default
        type=str,
        help="Base Llama version (affects config details like default context size, BOS/EOS, RoPE scaling).",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        default=False,
        help="Whether the model is an instruct model (affects EOS token and potentially chat template if tokenizer is converted).",
    )
    args = parser.parse_args()

    # --- Get Model Config ---
    if args.tt_flavor not in model_configs:
         raise ValueError(
             f"Unknown tt_flavor '{args.tt_flavor}'. Available flavors are: {list(model_configs.keys())}"
         )
    model_cfg = model_configs[args.tt_flavor]
    model_cfg.vocab_size = args.vocab_size # Set vocab size from argument

    # --- Tokenizer Conversion (Optional - currently commented out) ---
    # vocab_size_from_tokenizer = None
    # if args.tokenizer_path:
    #     print("Handling tokenizer...")
    #     if not os.path.exists(args.tokenizer_path):
    #         raise FileNotFoundError(f"Tokenizer input path not found: {args.tokenizer_path}")
    #     # Determine special tokens based on version/instruct args if needed
    #     # special_tokens = DEFAULT_LLAMA_SPECIAL_TOKENS.get(str(args.llama_version), []) # Define DEFAULT_LLAMA_SPECIAL_TOKENS if needed
    #     tokenizer = write_tokenizer(
    #         tokenizer_path=args.output_dir, # Save tokenizer in the same output dir
    #         input_tokenizer_path=args.tokenizer_path,
    #         llama_version=args.llama_version,
    #         # special_tokens=special_tokens,
    #         instruct=args.instruct,
    #         push_to_hub=args.push_to_hub,
    #     )
    #     vocab_size_from_tokenizer = len(tokenizer)
    #     print(f"Tokenizer converted/loaded. Vocab size from tokenizer: {vocab_size_from_tokenizer}")
    #     # Optional: Check consistency
    #     if args.vocab_size is not None and args.vocab_size != vocab_size_from_tokenizer:
    #         warnings.warn(f"Provided --vocab_size ({args.vocab_size}) differs from tokenizer vocab size ({vocab_size_from_tokenizer}). Using tokenizer's size.")
    #     final_vocab_size = vocab_size_from_tokenizer
    # elif args.vocab_size is not None:
    #     final_vocab_size = args.vocab_size
    #     print(f"Tokenizer path not provided, using specified --vocab_size: {final_vocab_size}")
    # else:
    #     raise ValueError("Either --tokenizer_path or --vocab_size must be provided.")
    # # --- End of commented-out Tokenizer section ---

    # Use vocab_size directly from args since tokenizer part is commented out
    final_vocab_size = args.vocab_size

    # --- Model Conversion ---
    print(f"\nStarting model conversion:")
    print(f"  Flavor:        '{args.tt_flavor}'")
    print(f"  Input Dir:     '{args.input_dir}'")
    print(f"  Checkpoint Step: {args.steps}")
    print(f"  Output Dir/Repo: '{args.output_dir}'")
    print(f"  Llama Version: {args.llama_version}")
    print(f"  Vocab Size:    {final_vocab_size}")
    print(f"  Instruct:      {args.instruct}")
    print(f"  Safetensors:   {args.safe_serialization}")
    print(f"  Push to Hub:   {args.push_to_hub}\n")

    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_cfg=model_cfg, # Pass the config object
        steps=args.steps,
        safe_serialization=args.safe_serialization,
        llama_version=args.llama_version,
        vocab_size=final_vocab_size,
        instruct=args.instruct,
        push_to_hub=args.push_to_hub,
    )

if __name__ == "__main__":
    # Ensure model_configs is defined before calling main()
    main()