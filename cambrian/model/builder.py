#    Copyright 2023 Haotian Liu
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


from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, PretrainedConfig

from cambrian.model import CambrianQwenConfig, CambrianQwenForCausalLM, CambrianQwen3Config, CambrianQwen3ForCausalLM
from cambrian.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.utils import rank0_print


_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def select_model_class(model_path, model_name=None):
    """Pick the Cambrian config / model classes from the config of a checkpoint or of a base language model.

    Falls back to the model name when the config cannot be read."""
    architectures = []
    try:
        config_dict, _ = PretrainedConfig.get_config_dict(model_path)
        architectures = config_dict.get("architectures") or []
        model_type = config_dict.get("model_type") or ""
    except Exception:
        model_type = ""

    qwen3 = (CambrianQwen3Config, CambrianQwen3ForCausalLM)
    qwen2 = (CambrianQwenConfig, CambrianQwenForCausalLM)

    # 1) architecture recorded in the checkpoint, 2) model_type, 3) model name.
    # Only the language model part of a name is compared: checkpoints of this repository are
    # CambrianQwen3ForCausalLM / cambrian_qwen3, a base language model is Qwen3ForCausalLM / qwen3,
    # and checkpoints saved under another class name (pre-release) are still recognised.
    if any(a.endswith("Qwen3ForCausalLM") for a in architectures):
        return qwen3
    if any(a.endswith(("QwenForCausalLM", "Qwen2ForCausalLM")) for a in architectures):
        return qwen2
    if model_type.endswith("qwen3"):
        return qwen3
    if model_type.endswith(("qwen", "qwen2")):
        return qwen2
    name = (model_name or "").lower()
    if "qwen3" in name:
        return qwen3
    if "qwen" in name:
        return qwen2
    raise ValueError(f"Cannot infer the model class of {model_path} (model_name={model_name}). Supported: Qwen2.5 (cambrian_qwen) and Qwen3 (cambrian_qwen3) based models.")


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    torch_dtype="float16",
    attn_implementation="sdpa",
    customized_config=None,
    overwrite_config=None,
    return_video_loader=False,
    **kwargs,
):
    """Load a (Gaze Attention) Cambrian checkpoint for inference.

    Args:
        model_path: local directory or Hugging Face repo id of the checkpoint.
        model_base: unused, kept for API compatibility (must be None).
        model_name: e.g. "cambrian_qwen3" (Qwen3) or "cambrian_qwen" (Qwen2.5). Only used as a
            fallback when the model class cannot be inferred from the checkpoint config.
        load_8bit, load_4bit: load the language model with 8-bit / 4-bit weights (bitsandbytes).
        overwrite_config: dict of config values to override at load time,
            e.g. {"gaze_topk_ratio": 0.25} to attend to more gaze regions.
        return_video_loader: additionally return a `load_video(path)` helper.

    Returns:
        tokenizer, model, image_processor, context_len[, video_loader]
    """
    if model_base is not None:
        raise ValueError("`model_base` is not supported: checkpoints are loaded as full models.")
    kwargs.pop("multimodal", None)  # kept for API compatibility

    kwargs["device_map"] = device_map
    if load_8bit or load_4bit:
        from transformers import BitsAndBytesConfig  # needs `pip install bitsandbytes`

        if load_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        if torch_dtype not in _TORCH_DTYPES:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}. Choose from {list(_TORCH_DTYPES)}.")
        kwargs["torch_dtype"] = _TORCH_DTYPES[torch_dtype]

    config_class, model_class = select_model_class(model_path, model_name)

    rank0_print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    cambrian_cfg = customized_config
    if overwrite_config is not None:
        if cambrian_cfg is None:
            cambrian_cfg = config_class.from_pretrained(model_path)
        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cambrian_cfg, k, v)
    if cambrian_cfg is not None:
        kwargs["config"] = cambrian_cfg

    model = model_class.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)
    rank0_print(f"Model Class: {model.__class__.__name__}")

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_device = device_map if isinstance(device_map, str) and device_map.startswith("cuda") else "cuda"
        vision_tower.to(device=vision_device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if getattr(model.config, "enable_gaze_attn", False):
        model._swap_to_gaze_attention()
        rank0_print(f"[Gaze] Gaze attention enabled (region size={model.config.gaze_region_size}, topk ratio={model.config.gaze_topk_ratio}, context tokens={getattr(model.config, 'num_context_tokens', 0)})")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    if return_video_loader:
        return tokenizer, model, image_processor, context_len, _make_video_loader(model.config)

    return tokenizer, model, image_processor, context_len


def _make_video_loader(model_cfg):
    from cambrian.utils import process_video_with_decord

    default_fps = getattr(model_cfg, "video_fps", 1)
    default_upbound = getattr(model_cfg, "frames_upbound", 32)
    default_force = getattr(model_cfg, "force_sample", False)

    def load_video(video_path, video_fps=None, frames_upbound=None, force_sample=None):
        """Sample frames from a video file.

        Args:
            video_path: path to the video file.
            video_fps: frames sampled per second (default: config value or 1).
            frames_upbound: maximum number of frames (default: config value or 32).
            force_sample: always sample `frames_upbound` frames uniformly.

        Returns:
            frames: numpy array [num_frames, H, W, 3]
            frame_idx: list[int], frame indices in the original video
            metadata: dict(video_time, frame_time, num_frames)
        """
        data_args = SimpleNamespace(
            video_fps=video_fps if video_fps is not None else default_fps,
            frames_upbound=frames_upbound if frames_upbound is not None else default_upbound,
            force_sample=force_sample if force_sample is not None else default_force,
        )
        video, video_time, frame_time, num_frames, frame_idx = process_video_with_decord(video_path, data_args)
        return video, frame_idx, {"video_time": video_time, "frame_time": frame_time, "num_frames": num_frames}

    return load_video
