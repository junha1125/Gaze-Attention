#!/usr/bin/env python3
"""
Test: gaze-region partitioning for different vision encoders / projectors.

Verifies:
  1. SigLipImageProcessor follows the image size of the vision config
  2. compute_gaze_regions with 576 patches (384px, 24x24) and K=6
  3. compute_gaze_regions with 1024 patches (512px, 32x32) and K=8
  4. compute_gaze_regions rejects a region size that does not divide the grid
  5. compute_gaze_regions with 144 patches (384px + 4x token compression) and K=6
  6. Multi-frame (video) inputs: regions are numbered consecutively across frames
  7. SigLipVisionEmbeddings position embeddings match the image size
"""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from cambrian.constants import MODALITY_TEXT, MODALITY_IMAGE, MODALITY_CONTEXT_TOKEN
from cambrian.model.language_model.gaze_attention import compute_gaze_regions

DEVICE = torch.device("cpu")
N_TEXT_PRE, N_TEXT_POST, N_CTX = 5, 10, 4


def make_modality_types(num_patches, num_frames=1, num_ctx=N_CTX):
    """[5 text] ([num_patches image] [num_ctx context]) x num_frames [10 text]"""
    frame_block = num_patches + num_ctx
    S = N_TEXT_PRE + num_frames * frame_block + N_TEXT_POST
    mtype = torch.full((1, S), MODALITY_TEXT, dtype=torch.long)
    for f in range(num_frames):
        start = N_TEXT_PRE + f * frame_block
        mtype[0, start : start + num_patches] = MODALITY_IMAGE
        mtype[0, start + num_patches : start + frame_block] = MODALITY_CONTEXT_TOKEN
    return mtype


def check_regions(num_patches, K, num_frames=1):
    grid = int(num_patches ** 0.5)
    mtype = make_modality_types(num_patches, num_frames)
    result = compute_gaze_regions(mtype, num_patches, K, DEVICE)
    assert result is not None, "compute_gaze_regions returned None"

    regions_per_frame = (grid // K) ** 2
    expected_regions = regions_per_frame * num_frames
    assert result["n_regions_per_item"][0].item() == expected_regions
    assert result["region_positions"].shape == (1, expected_regions, K * K)
    assert result["region_valid_mask"].shape == (1, expected_regions)
    assert result["region_valid_mask"].all(), "Not all regions are valid"

    # Image positions map to a region, everything else to -1
    ptr = result["position_to_region"][0]
    is_image = mtype[0] == MODALITY_IMAGE
    assert (ptr[~is_image] == -1).all(), "Non-image positions must map to -1"
    image_regions = ptr[is_image]
    assert image_regions.min() == 0 and image_regions.max() == expected_regions - 1
    for r in range(expected_regions):
        count = (image_regions == r).sum().item()
        assert count == K * K, f"Region {r}: expected {K * K} patches, got {count}"

    # Regions of frame f only contain patches of frame f
    frame_block = num_patches + N_CTX
    for f in range(num_frames):
        start = N_TEXT_PRE + f * frame_block
        pos = result["region_positions"][0, f * regions_per_frame : (f + 1) * regions_per_frame]
        assert pos.min() >= start and pos.max() < start + num_patches, f"Frame {f}: region crosses the frame boundary"
    return expected_regions


def test_image_processor_size():
    from cambrian.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor, SigLipVisionConfig

    for image_size in (384, 512):
        cfg = SigLipVisionConfig(image_size=image_size, patch_size=16)
        proc = SigLipImageProcessor(size=(cfg.image_size, cfg.image_size), crop_size={"height": cfg.image_size, "width": cfg.image_size})
        assert proc.size == (image_size, image_size)
        assert proc.crop_size == {"height": image_size, "width": image_size}
    print("[PASS] Test 1: SigLipImageProcessor follows the config image size")


def test_gaze_regions_384_K6():
    n = check_regions(num_patches=576, K=6)
    assert n == 16
    print(f"[PASS] Test 2: 576 patches + K=6 -> {n} regions of 36 patches")


def test_gaze_regions_512_K8():
    n = check_regions(num_patches=1024, K=8)
    assert n == 16
    print(f"[PASS] Test 3: 1024 patches + K=8 -> {n} regions of 64 patches")


def test_gaze_regions_invalid_region_size():
    mtype = make_modality_types(1024)
    try:
        compute_gaze_regions(mtype, 1024, 6, DEVICE)
        assert False, "Should have raised AssertionError for 32 % 6 != 0"
    except AssertionError as e:
        assert "not divisible" in str(e), f"Unexpected assert message: {e}"
    print("[PASS] Test 4: K=6 is rejected for a 32x32 grid")


def test_gaze_regions_token_compression():
    n = check_regions(num_patches=144, K=6)
    assert n == 4
    n3 = check_regions(num_patches=144, K=3)
    assert n3 == 16
    print(f"[PASS] Test 5: 144 patches -> {n} regions (K=6), {n3} regions (K=3)")


def test_gaze_regions_video():
    n = check_regions(num_patches=144, K=6, num_frames=8)
    assert n == 32
    n512 = check_regions(num_patches=1024, K=8, num_frames=4)
    assert n512 == 64
    print(f"[PASS] Test 6: video inputs -> {n} regions (8 frames x 144), {n512} regions (4 frames x 1024)")


def test_vision_embeddings_position_match():
    from cambrian.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig, SigLipVisionEmbeddings

    for image_size, expected_patches in [(384, 576), (512, 1024)]:
        cfg = SigLipVisionConfig(image_size=image_size, patch_size=16, hidden_size=64)
        emb = SigLipVisionEmbeddings(cfg)
        assert emb.num_patches == expected_patches
        assert emb.position_embedding.num_embeddings == expected_patches
        output = emb(torch.randn(1, 3, image_size, image_size))
        assert output.shape == (1, expected_patches, 64)
    print("[PASS] Test 7: SigLipVisionEmbeddings position embeddings match 384 and 512")


if __name__ == "__main__":
    test_image_processor_size()
    test_gaze_regions_384_K6()
    test_gaze_regions_512_K8()
    test_gaze_regions_invalid_region_size()
    test_gaze_regions_token_compression()
    test_gaze_regions_video()
    test_vision_embeddings_position_match()
    print("\n=== ALL 7 TESTS PASSED ===")
