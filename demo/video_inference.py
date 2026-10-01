#!/usr/bin/env python
"""Video inference with a Gaze Attention checkpoint.

Example:
    python demo/video_inference.py \
        --model-path <checkpoint dir or Hugging Face repo id> \
        --video assets/jobs.mp4 \
        --question "Describe what happens in this video."
"""

import argparse
import copy
import warnings

import torch

from cambrian.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from cambrian.conversation import conv_templates
from cambrian.mm_utils import tokenizer_image_token
from cambrian.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--video", type=str, default="assets/jobs.mp4")
    parser.add_argument("--question", type=str, nargs="+", default=["Describe what happens in this video."], help="One or more questions (each is asked independently).")
    parser.add_argument("--conv-mode", type=str, default="qwen_3")
    parser.add_argument("--max-frames", type=int, default=64, help="Frames are sampled at 1 FPS, or uniformly when the video is longer than this many seconds.")
    parser.add_argument("--gaze-topk-ratio", type=float, default=None, help="Override the fraction of gaze regions each query attends to.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    overwrite_config = {"gaze_topk_ratio": args.gaze_topk_ratio} if args.gaze_topk_ratio is not None else None
    tokenizer, model, image_processor, _, load_video = load_pretrained_model(
        args.model_path, None, "cambrian_qwen3", device_map=args.device, attn_implementation="sdpa", overwrite_config=overwrite_config, return_video_loader=True
    )
    model.eval()

    frames, frame_idx, meta = load_video(args.video, video_fps=1, frames_upbound=args.max_frames)
    print(f"Sampled {len(frame_idx)} frames from {args.video} ({meta['video_time']:.1f}s)")
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(dtype=torch.float16, device=args.device)

    for question in args.question:
        conv = copy.deepcopy(conv_templates[args.conv_mode])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[video_tensor],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"\nQ: {question}\nA: {answer}")


if __name__ == "__main__":
    main()
