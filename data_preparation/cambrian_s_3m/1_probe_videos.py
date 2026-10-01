#!/usr/bin/env python3
r"""Measure the duration of every video and drop the videos that cannot be decoded.

Example:
    python data_preparation/cambrian_s_3m/1_probe_videos.py \
        --input datasets/Cambrian-S-3M/cambrian_s_3m_filtered.jsonl \
        --video-root datasets/Cambrian-S-3M \
        --output datasets/Cambrian-S-3M/cambrian_s_3m_probed.jsonl

For every sample with a "video" entry:
  1. `ffprobe` reads the duration (samples are dropped when it fails or times out),
  2. decord decodes the first frame in a subprocess (samples are dropped when it fails or hangs),
  3. the duration in seconds is stored as "video_duration".
Samples without a "video" entry are dropped.

The input is split into `--num-chunks` parts. Every part is written to
<work-dir>/chunk_XXXX.jsonl when it is finished, and finished parts are skipped when the script
is started again, so a long run can be resumed. Keep --num-chunks unchanged when resuming.

Requires: ffprobe (ffmpeg), decord.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def get_video_duration(video_path, timeout):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def is_video_decodable(video_path, timeout):
    """Decode the first frame with decord in a subprocess, so that a hanging decoder can be killed."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from decord import VideoReader; vr = VideoReader(sys.argv[1]); vr[0]; print('OK')", video_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        return False


def process_chunk(chunk_idx, lines, args):
    chunk_path = os.path.join(args.work_dir, f"chunk_{chunk_idx:04d}.jsonl")
    if os.path.exists(chunk_path):
        print(f"[chunk {chunk_idx}] already done, skipped")
        return

    kept = []
    stats = {"no_video": 0, "ffprobe_failed": 0, "decode_failed": 0}
    for i, line in enumerate(lines):
        sample = json.loads(line)
        video = sample.get("video")
        if not isinstance(video, str):
            stats["no_video"] += 1
            continue
        video_path = os.path.join(args.video_root, video)

        duration = get_video_duration(video_path, args.ffprobe_timeout)
        if duration is None:
            stats["ffprobe_failed"] += 1
            continue
        if not is_video_decodable(video_path, args.decord_timeout):
            stats["decode_failed"] += 1
            continue

        sample["video_duration"] = duration
        kept.append(json.dumps(sample, ensure_ascii=False))
        if (i + 1) % 1000 == 0:
            print(f"[chunk {chunk_idx}] {i + 1:,}/{len(lines):,}", flush=True)

    tmp_path = chunk_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    os.replace(tmp_path, chunk_path)
    print(f"[chunk {chunk_idx}] kept {len(kept):,}/{len(lines):,} {stats}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input .jsonl file")
    parser.add_argument("--video-root", required=True, help="Folder that the video paths are relative to")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--work-dir", default=None, help="Folder for the per-chunk results (default: <output>.chunks)")
    parser.add_argument("--num-chunks", type=int, default=64, help="Number of chunks (= unit of resuming)")
    parser.add_argument("--workers", type=int, default=16, help="Number of chunks processed in parallel")
    parser.add_argument("--ffprobe-timeout", type=int, default=10, help="Seconds")
    parser.add_argument("--decord-timeout", type=int, default=10, help="Seconds")
    args = parser.parse_args()
    args.work_dir = args.work_dir or args.output + ".chunks"
    os.makedirs(args.work_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    print(f"Loaded {len(lines):,} samples")

    chunk_size = max(1, (len(lines) + args.num_chunks - 1) // args.num_chunks)
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_chunk, idx, chunk, args) for idx, chunk in enumerate(chunks)]
        for future in futures:
            future.result()

    # Merge the chunks in the original order
    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for idx in range(len(chunks)):
            with open(os.path.join(args.work_dir, f"chunk_{idx:04d}.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    total += 1
    print(f"Kept {total:,}/{len(lines):,} samples -> {args.output}")


if __name__ == "__main__":
    main()
