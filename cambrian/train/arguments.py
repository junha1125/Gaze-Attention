from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-4B-Instruct-2507", metadata={"help": "Language model (Qwen2.5 / Qwen3) or a previously trained checkpoint to continue from."})
    version: Optional[str] = field(default="qwen_3", metadata={"help": "Prompt template: 'plain' for projector pretraining, 'qwen_3' / 'qwen_1_5' for finetuning."})

    # Which parts of the model to train, e.g. "mm_mlp_adapter" (stage 1) or
    # "mm_vision_tower,mm_mlp_adapter,mm_language_model" (stage 2).
    mm_tunable_parts: Optional[str] = field(default=None)
    freeze_backbone: bool = field(default=False)

    # Vision encoder / projector
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to mm_projector.bin produced by projector pretraining."})
    mm_projector_type: Optional[str] = field(default="linear", metadata={"help": "'mlp2x_gelu' (no compression) or 'mlp2x_gelu_conv2x' (4x token compression)."})
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default="flat")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    # Gaze attention
    num_context_tokens: int = field(default=0, metadata={"help": "Number of learnable context tokens appended after each image/frame. 0 disables."})
    gaze_region_size: int = field(default=6, metadata={"help": "Side length (in patches) of one gaze region."})
    gaze_topk_ratio: float = field(default=0.2, metadata={"help": "Fraction of gaze regions each text token attends to (TopK = ceil(ratio * n_regions))."})
    enable_gaze_attn: bool = field(default=False, metadata={"help": "Enable gaze attention (TopK region routing for text tokens)."})
    enable_progressive_topk: bool = field(default=False, metadata={"help": "Enable the dense-to-sparse progressive TopK schedule that decays gaze_topk_ratio during training."})
    progressive_topk_start_ratio: float = field(default=0.9, metadata={"help": "Starting gaze_topk_ratio for the progressive TopK schedule."})
    progressive_topk_end_ratio: float = field(default=0.1, metadata={"help": "Ending gaze_topk_ratio for the progressive TopK schedule."})
    progressive_topk_decay_fraction: float = field(default=0.6, metadata={"help": "Fraction of total training steps over which the ratio decays from start to end (e.g. 0.6 = first 60%)."})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Training annotations: a .json / .jsonl file, a .yaml file listing several datasets, or /path/to/{a,b,c}.json."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="sdpa", metadata={"help": "Attention implementation of the language model. Use 'sdpa' (needed for the context-token attention mask)."})
