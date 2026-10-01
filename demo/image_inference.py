#!/usr/bin/env python
"""Single-image inference with a Gaze Attention checkpoint.

Example:
    python demo/image_inference.py \
        --model-path <checkpoint dir or Hugging Face repo id> \
        --image assets/example2_dog.jpg \
        --question "Describe this image."
"""

import argparse
import copy
import warnings

import torch
from PIL import Image

from cambrian.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from cambrian.conversation import conv_templates
from cambrian.mm_utils import process_images, tokenizer_image_token
from cambrian.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image", type=str, default="assets/example2_dog.jpg")
    parser.add_argument("--question", type=str, nargs="+", default=["Describe this image."], help="One or more questions (each is asked independently).")
    parser.add_argument("--conv-mode", type=str, default="qwen_3")
    parser.add_argument("--gaze-topk-ratio", type=float, default=None, help="Override the fraction of gaze regions each query attends to.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    overwrite_config = {"gaze_topk_ratio": args.gaze_topk_ratio} if args.gaze_topk_ratio is not None else None
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "cambrian_qwen3", device_map=args.device, attn_implementation="sdpa", overwrite_config=overwrite_config
    )
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=args.device) for _image in image_tensor]

    for question in args.question:
        conv = copy.deepcopy(conv_templates[args.conv_mode])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"\nQ: {question}\nA: {answer}")


if __name__ == "__main__":
    main()
