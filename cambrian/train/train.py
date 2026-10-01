# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import pathlib
import re

import torch
import transformers
from PIL import ImageFile

from cambrian import conversation as conversation_lib
from cambrian.model.builder import select_model_class
from cambrian.utils import rank0_print

from cambrian.train.arguments import DataArguments, TrainingArguments, ModelArguments
from cambrian.train.dataloader import make_supervised_data_module
from cambrian.train.cambrian_trainer import CambrianTrainer, get_mm_adapter_state_maybe_zero_3, maybe_zero_3

torch.multiprocessing.set_sharing_strategy("file_descriptor")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

# On-disk key of the context-token embedding rows inside mm_projector.bin
CONTEXT_EMBEDDING_KEY = "compressed_context_embeddings"


def is_projector_only_training(args):
    """True when only the projector (and context tokens) are trained, i.e. stage 1."""
    if getattr(args, "tune_mm_mlp_adapter", False):
        return True
    mm_tunable_parts = getattr(args, "mm_tunable_parts", None)
    if mm_tunable_parts is None:
        return False
    parts = mm_tunable_parts.split(",")
    return len(parts) == 1 and ("mm_mlp_adapter" in mm_tunable_parts or "mm_vision_resampler" in mm_tunable_parts)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    projector_only = is_projector_only_training(trainer.args)

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {projector_only}")
    if projector_only:
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        # Save the context-token embedding rows (not the full embedding table).
        ctx_ids = getattr(trainer.model.config, "context_token_ids", [])
        if ctx_ids:
            embed_w = trainer.model.get_input_embeddings().weight
            embed_w_full = maybe_zero_3(embed_w, ignore_status=True).cpu()
            weight_to_save[CONTEXT_EMBEDDING_KEY] = embed_w_full[ctx_ids].clone()
            rank0_print(f"Saved context-token embeddings ({len(ctx_ids)} rows, ids={ctx_ids}) into mm_projector.bin")
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_model(model_args, training_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    if model_args.vision_tower is None:
        raise ValueError("--vision_tower is required.")

    # Qwen2.5 or Qwen3, from the config of the base language model / of the checkpoint to continue from
    config_class, model_class = select_model_class(model_args.model_name_or_path, model_args.model_name_or_path)

    config = config_class.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        config.use_pos_skipping = model_args.use_pos_skipping
        config.pos_skipping_range = model_args.pos_skipping_range

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
    )
    return model


def setup_context_tokens(model, tokenizer, model_args, tunable_parts):
    """Add the learnable context tokens to the vocabulary and make them trainable."""
    # The token strings keep their original names so that released tokenizers stay compatible.
    context_tokens = [f"<compressed_context_{i}>" for i in range(model_args.num_context_tokens)]
    already_in_vocab = sum(1 for t in context_tokens if t in tokenizer.get_vocab())
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": context_tokens})
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    rank0_print(f"[ContextToken] tokens already in vocab: {already_in_vocab}/{len(context_tokens)}, num_added: {num_added}")

    context_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in context_tokens]
    original_vocab_size = len(tokenizer) - model_args.num_context_tokens
    assert context_token_ids == list(range(original_vocab_size, len(tokenizer))), \
        f"context token IDs {context_token_ids} are not contiguous at the end of vocab [{original_vocab_size}:{len(tokenizer)})"
    model.config.context_token_ids = context_token_ids
    model.config.num_context_tokens = model_args.num_context_tokens
    rank0_print(f"[ContextToken] Added {model_args.num_context_tokens} context tokens: {context_tokens} (ids: {context_token_ids})")

    # Load context-token embeddings from a pretrained mm_projector.bin (stage 1 output)
    pretrain_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
    if pretrain_adapter is not None:
        mm_weights = torch.load(pretrain_adapter, map_location="cpu")
        ctx_token_w = mm_weights.get(CONTEXT_EMBEDDING_KEY, None)
        if ctx_token_w is not None:
            embed_weight = model.get_input_embeddings().weight
            if embed_weight.shape[0] < max(context_token_ids) + 1:
                raise RuntimeError(
                    f"embed_tokens has {embed_weight.shape[0]} rows but context_token_ids "
                    f"go up to {max(context_token_ids)}. Resize token embeddings first."
                )
            embed_weight.data[context_token_ids] = ctx_token_w.to(device=embed_weight.device, dtype=embed_weight.dtype)
            rank0_print(f"[ContextToken] Loaded context-token embeddings ({ctx_token_w.shape[0]} rows) from {pretrain_adapter}")
        del mm_weights

    # Make the context-token embeddings trainable even when the language model is frozen
    if "mm_language_model" not in tunable_parts:
        embed_weight = model.get_input_embeddings().weight
        embed_weight.requires_grad_(True)

        def freeze_non_context_grad(grad, n=original_vocab_size):
            grad[:n] = 0
            return grad

        embed_weight.register_hook(freeze_non_context_grad)
        rank0_print(f"[ContextToken] Gradient hook registered: only context-token rows [{original_vocab_size}:] are trainable")


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank

    # --- Detect a GPU-count mismatch when resuming from a DeepSpeed checkpoint ---
    # DeepSpeed stores one optimizer state file per rank, so its checkpoints cannot be
    # resumed with a different number of GPUs. In that case the model weights of the
    # latest checkpoint are reloaded and the optimizer starts fresh; the step counter and
    # the learning rate schedule continue.
    _gpu_mismatch_resume = False
    _ckpt_dirs = sorted(
        pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    if _ckpt_dirs:
        _latest_ckpt = _ckpt_dirs[-1]
        _gs_dirs = list(_latest_ckpt.glob("global_step*"))
        if _gs_dirs:
            _optim_files = list(_gs_dirs[0].glob("*_optim_states.pt"))
            _world_size = int(os.environ.get("WORLD_SIZE", 1))
            if len(_optim_files) != _world_size:
                rank0_print(f"[Resume] GPU count mismatch: checkpoint={len(_optim_files)}, current={_world_size}")
                rank0_print(f"[Resume] Redirecting model_name_or_path to {_latest_ckpt}")
                model_args.model_name_or_path = str(_latest_ckpt)
                # the checkpoint already contains the trained projector and context tokens
                model_args.pretrain_mm_mlp_adapter = None
                _gpu_mismatch_resume = True

    model = get_model(model_args, training_args)
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")

    rank0_print(f"Prompt version: {model_args.version}")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    if model_args.version not in conversation_lib.conv_templates:
        raise ValueError(f"Unknown prompt version: {model_args.version}. Choose from {list(conversation_lib.conv_templates)}.")
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_grid_pinpoints is not None:
        if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
            try:
                patch_size = data_args.image_processor.size[0]
            except Exception as e:
                patch_size = data_args.image_processor.size["shortest_edge"]

            assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
            # Use regex to extract the range from the input string
            matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
            range_start = tuple(map(int, matches[0]))
            range_end = tuple(map(int, matches[-1]))
            # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
            grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
            # Multiply all elements by patch_size
            data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
        elif isinstance(data_args.image_grid_pinpoints, str):
            data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.add_time_instruction = data_args.add_time_instruction
    model.config.force_sample = data_args.force_sample

    ### Deciding train which part of the model
    rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
    model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
    # Set the entire model to not require gradients by default
    model.requires_grad_(False)
    vision_tower.requires_grad_(False)
    model.get_model().mm_projector.requires_grad_(False)
    model.get_model().vision_resampler.requires_grad_(False)
    # Parse the mm_tunable_parts to decide which parts to unfreeze
    tunable_parts = model_args.mm_tunable_parts.split(",")
    if "mm_mlp_adapter" in tunable_parts:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    if "mm_vision_resampler" in tunable_parts:
        for p in model.get_model().vision_resampler.parameters():
            p.requires_grad = True
    if "mm_vision_tower" in tunable_parts:
        for name, param in model.named_parameters():
            if "vision_tower" in name:
                param.requires_grad_(True)
    if "mm_language_model" in tunable_parts:
        for name, param in model.named_parameters():
            if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                param.requires_grad_(True)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"[Model Stats] Total parameters: ~{total_params/1e6:.2f} M")
    rank0_print(f"[Model Stats] Trainable parameters: ~{trainable_params/1e6:.2f} M")

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # === Learnable context tokens ===
    if model_args.num_context_tokens > 0:
        setup_context_tokens(model, tokenizer, model_args, tunable_parts)

    # === Gaze attention ===
    model.config.gaze_region_size = model_args.gaze_region_size
    model.config.gaze_topk_ratio = model_args.gaze_topk_ratio
    model.config.enable_gaze_attn = model_args.enable_gaze_attn
    model.config.enable_progressive_topk = model_args.enable_progressive_topk
    model.config.progressive_topk_start_ratio = model_args.progressive_topk_start_ratio
    model.config.progressive_topk_end_ratio = model_args.progressive_topk_end_ratio
    model.config.progressive_topk_decay_fraction = model_args.progressive_topk_decay_fraction

    if model_args.enable_progressive_topk:
        model.config.gaze_topk_ratio = model_args.progressive_topk_start_ratio
        rank0_print(f"[Gaze] Progressive TopK enabled: ratio will decay {model_args.progressive_topk_start_ratio} -> {model_args.progressive_topk_end_ratio} over the first {model_args.progressive_topk_decay_fraction*100:.0f}% of training")

    if model_args.enable_gaze_attn:
        model._swap_to_gaze_attention()
        rank0_print("[Gaze] Swapped attention -> GazeAttention for all layers")

    # --- Extract scheduler.pt from the DeepSpeed checkpoint if missing ---
    if _gpu_mismatch_resume:
        _sched_path = _latest_ckpt / "scheduler.pt"
        if not _sched_path.exists():
            _gs_dir = list(_latest_ckpt.glob("global_step*"))
            _model_states = list(_gs_dir[0].glob("mp_rank_*_model_states.pt")) if _gs_dir else []
            if _model_states:
                _ms = torch.load(str(_model_states[0]), map_location="cpu")
                if "lr_scheduler" in _ms:
                    torch.save(_ms["lr_scheduler"], str(_sched_path))
                    rank0_print(f"[Resume] Extracted scheduler state from {_model_states[0].name} -> scheduler.pt")
                del _ms

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CambrianTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if _gpu_mismatch_resume:
        trainer._gpu_mismatch_resume = True

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if torch.cuda.is_available():
        rank0_print(f"Peak GPU memory of the first process: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved")

    model.config.use_cache = True

    # The progressive TopK schedule ends at `progressive_topk_end_ratio`; make sure the
    # saved config carries the final (inference) ratio.
    if model_args.enable_progressive_topk:
        model.config.gaze_topk_ratio = model_args.progressive_topk_end_ratio

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
