import os
import sys
import contextlib

import numpy as np
import torch.distributed as dist

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Please install decord to use video processing functions.")


@contextlib.contextmanager
def _suppress_stderr():
    """Suppress C-level stderr (e.g., FFmpeg warnings) inside the block."""
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)


def process_video_with_decord(video_file, data_args):
    """Sample frames from a video.

    Frames are first sampled at `data_args.video_fps` frames per second. If that
    yields more than `data_args.frames_upbound` frames (or `force_sample` is
    set), `frames_upbound` frames are sampled uniformly instead.

    Returns (frames [N, H, W, 3], video_time, frame_time, num_frames, frame_idx).
    """
    with _suppress_stderr():
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i / avg_fps for i in frame_idx]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    with _suppress_stderr():
        video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample, frame_idx


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
