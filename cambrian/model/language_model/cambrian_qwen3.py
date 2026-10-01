#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from cambrian.model.cambrian_arch import CambrianMetaModel, CambrianMetaForCausalLM
from cambrian.constants import MODALITY_CONTEXT_TOKEN
from .gaze_attention import (
    GazeAttention,
    build_context_token_causal_mask,
    build_gaze_ctx_prefill,
    build_gaze_ctx_decode,
    migrate_legacy_gaze_config,
)


class CambrianQwen3Config(Qwen3Config):
    model_type = "cambrian_qwen3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        migrate_legacy_gaze_config(self)


class CambrianQwen3Model(CambrianMetaModel, Qwen3Model):
    config_class = CambrianQwen3Config

    def __init__(self, config: Qwen3Config):
        super(CambrianQwen3Model, self).__init__(config)


class CambrianQwen3ForCausalLM(Qwen3ForCausalLM, CambrianMetaForCausalLM):
    config_class = CambrianQwen3Config

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "cambrian_qwen3"
        config.rope_scaling = None

        self.model = CambrianQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def _swap_to_gaze_attention(self):
        """Swap Qwen3Attention -> GazeAttention on every layer."""
        for layer in self.model.layers:
            layer.self_attn.__class__ = GazeAttention

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
        frame_indices: Optional[List] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, frame_indices=frame_indices)

        modality_types = getattr(self, "_modality_types", None)
        if modality_types is not None and inputs_embeds is not None and modality_types.shape[0] != inputs_embeds.shape[0]:
            # beam search / several return sequences: generate() repeats every input
            modality_types = modality_types.repeat_interleave(inputs_embeds.shape[0] // modality_types.shape[0], dim=0)

        # Restrict context tokens to their own frame via a 4D causal mask
        if (
            modality_types is not None
            and inputs_embeds is not None
            and (modality_types == MODALITY_CONTEXT_TOKEN).any()
        ):
            attention_mask = build_context_token_causal_mask(
                attention_mask, modality_types,
                inputs_embeds.dtype, inputs_embeds.device,
            )

        # Gaze routing context: built from modality_types at training/prefill,
        # rebuilt from the prefill cache at each decode step
        gaze_ctx = None
        if getattr(self.config, "enable_gaze_attn", False):
            if modality_types is not None and inputs_embeds is not None:
                gaze_ctx, self._gaze_cache = build_gaze_ctx_prefill(self.config, modality_types)
            elif modality_types is None and past_key_values is not None:
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                cache = getattr(self, "_gaze_cache", None)
                device = input_ids.device if input_ids is not None else (cache["image_mask"].device if cache else None)
                gaze_ctx, self._gaze_cache = build_gaze_ctx_decode(self.config, cache, batch_size, device)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gaze_ctx=gaze_ctx,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        frame_indices: Optional[List] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # Clear stale gaze caches from any previous generate() call
        self._gaze_cache = None
        self._modality_types = None
        if getattr(self.config, "enable_gaze_attn", False):
            for layer in self.model.layers:
                if hasattr(layer.self_attn, "_cached_descriptors"):
                    layer.self_attn._cached_descriptors = None

        if images is not None:
            # Prompts of different lengths are padded on the left, so that the last position of
            # every sequence is a real token (tokenizer_padding_side of the config is for training).
            padding_side = getattr(self.config, "tokenizer_padding_side", "right")
            self.config.tokenizer_padding_side = "left"
            try:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, frame_indices=frame_indices)
            finally:
                self.config.tokenizer_padding_side = padding_side
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if attention_mask is None:
            # No padding information was passed: all positions are real tokens. This is the mask that
            # generate() of Hugging Face would create itself, together with a warning.
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("cambrian_qwen3", CambrianQwen3Config)
AutoModelForCausalLM.register(CambrianQwen3Config, CambrianQwen3ForCausalLM)
