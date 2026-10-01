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


from abc import ABC, abstractmethod

import random
import torch

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, MODALITY_TEXT, MODALITY_IMAGE, MODALITY_CONTEXT_TOKEN, MODALITY_PAD
from cambrian.utils import rank0_print


class CambrianMetaModel:

    def __init__(self, config):
        super(CambrianMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(self.config, vision_cfg=self.vision_tower.config if hasattr(self.vision_tower, "config") else None)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        # Set num_patches_one_image dynamically from the vision tower,
        # e.g. siglip-384/patch16 -> 24x24=576, siglip-512/patch16 -> 32x32=1024
        num_patches = vision_tower.num_patches
        if self.config.mm_projector_type == "mlp2x_gelu_conv2x":
            # Conv2d(stride=2) halves each spatial dim -> patches / 4
            self.config.num_patches_one_image = num_patches // 4
        else:
            self.config.num_patches_one_image = num_patches
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config if hasattr(vision_tower, "config") else None)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


class CambrianMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, frame_indices=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            self._modality_types = None
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # === Context-token embeddings (precompute once) ===
        num_context_tokens = getattr(self.config, "num_context_tokens", 0)
        if num_context_tokens > 0:
            ctx_token_ids = torch.tensor(self.config.context_token_ids, device=input_ids.device)
            ctx_embeds = self.get_model().embed_tokens(ctx_token_ids)  # [num_ctx, dim]

        # ==========================================================
        # Branch A: images is a list or 5D tensor (multi-image / video)
        #   Each batch item may have multiple crops or video frames.
        # ==========================================================
        if type(images) is list or images.ndim == 5:
            # Ensure every image tensor is 4D: [num_crops_or_frames, C, H, W]
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            images_4d = []
            for image in images:
                if image.ndim == 4:
                    images_4d.append(image)
                else:
                    images_4d.append(image.unsqueeze(0))

            # Concatenate all images across the batch for a single encode pass,
            # then split back per batch item.
            # Each element: [num_crops_or_frames, num_patches, hidden_dim]
            split_sizes = [image.shape[0] for image in images_4d]
            concat_images = torch.cat(images_4d, dim=0)
            encoded_image_features = self.encode_images(concat_images)
            image_features = list(torch.split(encoded_image_features, split_sizes))

            expected = getattr(self.config, "num_patches_one_image", 576)
            for idx, feat in enumerate(image_features):
                patch_num_error = (f"num_patches_one_image={expected} (projector={getattr(self.config, 'mm_projector_type', 'unknown')}).\n"
                                   f"Batch {idx}: expected {expected} patches, got {feat.shape[1]} (shape={feat.shape})")
                assert feat.shape[1] == expected, patch_num_error

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            if mm_patch_merge_type != "flat":
                raise NotImplementedError(f"mm_patch_merge_type={mm_patch_merge_type} is not supported. Only 'flat' is implemented.")

            image_modality_types = []
            if num_context_tokens > 0:
                new_features = []
                for idx, x in enumerate(image_features):
                    # Per-frame (video) / per-crop (anyres) context-token insertion.
                    # For single images (x.shape[0]==1) this equals appending
                    # the context tokens once at the end.
                    mtype_unit_img, mtype_repeat = x.shape[1], x.shape[0]
                    ctx_exp = ctx_embeds.unsqueeze(0).expand(x.shape[0], -1, -1)
                    x = torch.cat([x, ctx_exp], dim=1)
                    new_features.append(x.flatten(0, 1))
                    # modality type: [IMAGE * n_patches, CONTEXT * num_ctx] per crop/frame
                    unit_mtype = torch.cat([
                        torch.full((mtype_unit_img,), MODALITY_IMAGE, dtype=torch.long, device=x.device),
                        torch.full((num_context_tokens,), MODALITY_CONTEXT_TOKEN, dtype=torch.long, device=x.device),
                    ])
                    image_modality_types.append(unit_mtype.repeat(mtype_repeat))
                image_features = new_features
            else:
                image_features = [x.flatten(0, 1) for x in image_features]
                image_modality_types = [
                    torch.full((x.shape[0],), MODALITY_IMAGE, dtype=torch.long, device=x.device)
                    for x in image_features
                ]

        # ==========================================================
        # Branch B: simple 4D tensor, one image per batch item
        # ==========================================================
        else:
            image_features = self.encode_images(images)

            expected = getattr(self.config, "num_patches_one_image", 576)
            patch_num_error = (f"num_patches_one_image={expected} (projector={getattr(self.config, 'mm_projector_type', 'unknown')}).\n"
                               f"Expected {expected} patches, got {image_features.shape[1]} (shape={image_features.shape})")
            assert image_features.shape[1] == expected, patch_num_error

            if num_context_tokens > 0:
                num_patches = image_features.shape[1]
                ctx_exp = ctx_embeds.unsqueeze(0).expand(image_features.shape[0], -1, -1)
                image_features = torch.cat([image_features, ctx_exp], dim=1)
                single_mtype = torch.cat([
                    torch.full((num_patches,), MODALITY_IMAGE, dtype=torch.long, device=image_features.device),
                    torch.full((num_context_tokens,), MODALITY_CONTEXT_TOKEN, dtype=torch.long, device=image_features.device),
                ])
                image_modality_types = [single_mtype for _ in range(image_features.shape[0])]
            else:
                single_mtype = torch.full((image_features.shape[1],), MODALITY_IMAGE, dtype=torch.long, device=image_features.device)
                image_modality_types = [single_mtype for _ in range(image_features.shape[0])]

        # ==========================================================
        # Step 1: Fill in defaults for None inputs (save originals)
        # ==========================================================
        orig_labels = labels
        orig_position_ids = position_ids
        orig_attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # ==========================================================
        # Step 2: Remove padding tokens using attention_mask
        #   After this, input_ids/labels become lists of 1D tensors
        #   (variable length per batch item, padding removed)
        # ==========================================================
        input_ids = [cur_ids[cur_mask] for cur_ids, cur_mask in zip(input_ids, attention_mask)]
        labels = [cur_lab[cur_mask] for cur_lab, cur_mask in zip(labels, attention_mask)]

        # ==========================================================
        # Step 3: Replace IMAGE_TOKEN_INDEX positions with actual
        #   image feature embeddings, for each batch item
        # ==========================================================
        new_input_embeds = []
        new_labels = []
        new_modality_types = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # "num_images" is the number of <image> placeholders:
            # a video = 1 placeholder, one image = 1 placeholder, two images = 2 placeholders.
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if num_images == 0:
                # No image tokens in this sequence.
                # Cat zero-length image tensor to keep image encoder in
                # the compute graph for gradient flow.
                cur_image_features = image_features[cur_image_idx]
                text_embeds = self.get_model().embed_tokens(cur_input_ids)
                combined_embeds = torch.cat([text_embeds, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(combined_embeds)
                new_labels.append(labels[batch_idx])
                new_modality_types.append(
                    torch.full((combined_embeds.shape[0],), MODALITY_TEXT, dtype=torch.long, device=combined_embeds.device)
                )
                cur_image_idx += 1
                continue

            # Locate all IMAGE_TOKEN_INDEX positions.
            # Add sentinel -1 (start) and seq_len (end) so that slicing
            #   [boundaries[i]+1 : boundaries[i+1]]
            # gives each text-only segment between image tokens.
            #
            # Example: tokens = [A, B, <IMG>, C, D, <IMG>, E]
            #   image_positions = [2, 5]
            #   boundaries      = [-1, 2, 5, 7]
            #   text segments   = [0:2]=[A,B], [3:5]=[C,D], [6:7]=[E]
            image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            boundaries = [-1] + image_positions + [cur_input_ids.shape[0]]

            cur_labels = labels[batch_idx]
            text_id_segments = []
            text_label_segments = []
            for i in range(len(boundaries) - 1):
                seg_start = boundaries[i] + 1
                seg_end = boundaries[i + 1]
                text_id_segments.append(cur_input_ids[seg_start:seg_end])
                text_label_segments.append(cur_labels[seg_start:seg_end])

            # Embed all text tokens at once, then split back into segments
            segment_lengths = [seg.shape[0] for seg in text_label_segments]
            all_text_embeds = self.get_model().embed_tokens(torch.cat(text_id_segments))
            text_embed_segments = torch.split(all_text_embeds, segment_lengths, dim=0)

            # Interleave: [text_seg_0, img_feat_0, text_seg_1, img_feat_1, ..., text_seg_N]
            interleaved_embeds = []
            interleaved_labels = []
            interleaved_mtypes = []

            for i in range(num_images + 1):
                interleaved_embeds.append(text_embed_segments[i])
                interleaved_labels.append(text_label_segments[i])
                interleaved_mtypes.append(
                    torch.full((text_embed_segments[i].shape[0],), MODALITY_TEXT, dtype=torch.long, device=cur_labels.device)
                )

                if i < num_images:  # text segments = num_images+1, image segments = num_images
                    try:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_mtype = image_modality_types[cur_image_idx]
                    except IndexError:  # two <image> placeholders in a conversation, but only one image provided
                        cur_image_features = image_features[cur_image_idx - 1]
                        cur_image_mtype = image_modality_types[cur_image_idx - 1]
                    cur_image_idx += 1
                    interleaved_embeds.append(cur_image_features)
                    interleaved_mtypes.append(cur_image_mtype)
                    # Image tokens get IGNORE_INDEX label (no loss computed on them)
                    num_img_tokens = cur_image_features.shape[0]
                    img_labels = torch.full((num_img_tokens,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                    interleaved_labels.append(img_labels)

            interleaved_embeds = [x.to(self.device) for x in interleaved_embeds]
            new_input_embeds.append(torch.cat(interleaved_embeds))
            new_labels.append(torch.cat(interleaved_labels))
            new_modality_types.append(torch.cat(interleaved_mtypes))

        # ==========================================================
        # Step 4: Truncate to max length
        #   (image embeddings can make sequences exceed tokenizer max)
        # ==========================================================
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None and not self.training:
            longest = max(x.shape[0] for x in new_input_embeds)
            if longest > tokenizer_model_max_length:
                rank0_print(
                    f"[WARNING] The input has {longest} tokens (visual + text) and is truncated to tokenizer_model_max_length={tokenizer_model_max_length}. "
                    "Use fewer frames or load the model with overwrite_config={'tokenizer_model_max_length': <larger value>}."
                )
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        new_modality_types = [x[:tokenizer_model_max_length] for x in new_modality_types]

        # ==========================================================
        # Step 5: Pad all sequences to same length, stack into batch
        # ==========================================================
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        padded_embeds = []
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        padded_modality_types = torch.full((batch_size, max_len), MODALITY_PAD, dtype=torch.long, device=new_labels[0].device)

        padding_side = getattr(self.config, "tokenizer_padding_side", "right")

        for i, (cur_embed, cur_label, cur_mtype) in enumerate(zip(new_input_embeds, new_labels, new_modality_types)):
            seq_len = cur_embed.shape[0]
            pad_len = max_len - seq_len
            embed_dim = cur_embed.shape[1]
            pad_tensor = torch.zeros((pad_len, embed_dim), dtype=cur_embed.dtype, device=cur_embed.device)

            if padding_side == "left":
                padded_embeds.append(torch.cat((pad_tensor, cur_embed), dim=0))
                if seq_len > 0:
                    padded_labels[i, -seq_len:] = cur_label
                    attention_mask[i, -seq_len:] = True
                    position_ids[i, -seq_len:] = torch.arange(0, seq_len, dtype=position_ids.dtype, device=position_ids.device)
                    padded_modality_types[i, -seq_len:] = cur_mtype
            else:
                padded_embeds.append(torch.cat((cur_embed, pad_tensor), dim=0))
                if seq_len > 0:
                    padded_labels[i, :seq_len] = cur_label
                    attention_mask[i, :seq_len] = True
                    position_ids[i, :seq_len] = torch.arange(0, seq_len, dtype=position_ids.dtype, device=position_ids.device)
                    padded_modality_types[i, :seq_len] = cur_mtype

        new_input_embeds = torch.stack(padded_embeds, dim=0)

        # ==========================================================
        # Step 6: Restore None for originally-None inputs
        # ==========================================================
        if orig_labels is None:
            new_labels = None
        else:
            new_labels = padded_labels

        if orig_attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=orig_attention_mask.dtype)

        if orig_position_ids is None:
            position_ids = None

        # Optional: randomized position skipping during training
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        # Per-token modality map, consumed by the gaze routing / context-token mask
        self._modality_types = padded_modality_types
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        pass
