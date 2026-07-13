from __future__ import annotations

import numpy as np

from droplet_fusion.config import SegmentationConfig
from droplet_fusion.io import load_tiff_stack, synthetic_relaxing_ellipse_stack
from droplet_fusion.segmentation import (
    estimate_background_masked_poly,
    segment_frame,
    segment_stack,
    write_mask_stack,
    write_segmentation_qc_csv,
)


def _disk_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    yy, xx = np.indices(shape)
    cy, cx = center
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def _gradient_frame(shape: tuple[int, int] = (72, 80)) -> np.ndarray:
    yy, xx = np.indices(shape)
    return 1000.0 + 0.8 * yy + 0.5 * xx


def test_segment_frame_removes_border_touching_object():
    frame = _gradient_frame()
    center_droplet = _disk_mask(frame.shape, center=(36, 40), radius=9)
    border_object = np.zeros(frame.shape, dtype=bool)
    border_object[:22, :22] = True
    frame[center_droplet] += 1200
    frame[border_object] += 2000

    result = segment_frame(frame.astype(np.uint16), SegmentationConfig(min_object_area_px=50))

    assert result.valid
    assert result.failure_reason is None
    assert result.mask[36, 40]
    assert not np.any(result.mask[0, :])
    assert not np.any(result.mask[:, 0])


def test_segment_frame_selects_largest_non_border_object():
    frame = _gradient_frame()
    large_droplet = _disk_mask(frame.shape, center=(36, 40), radius=10)
    small_droplet = _disk_mask(frame.shape, center=(18, 60), radius=5)
    frame[large_droplet] += 1500
    frame[small_droplet] += 1500

    result = segment_frame(frame.astype(np.uint16), SegmentationConfig(min_object_area_px=20))

    assert result.valid
    assert result.candidate_count >= 2
    assert result.mask[36, 40]
    assert not result.mask[18, 60]


def test_segment_frame_recovers_primary_droplet_near_border_object():
    frame = _gradient_frame((96, 104))
    primary = _disk_mask(frame.shape, center=(50, 50), radius=15)
    border_droplet = _disk_mask(frame.shape, center=(50, 5), radius=15)
    dim_bridge = np.zeros(frame.shape, dtype=bool)
    dim_bridge[45:56, 17:38] = True

    frame[primary] += 1600
    frame[border_droplet] += 1600
    frame[dim_bridge] += 700

    result = segment_frame(frame.astype(np.uint16), SegmentationConfig(min_object_area_px=80, snr_threshold=1.0))

    assert result.valid
    assert result.mask[50, 50]
    assert not np.any(result.mask[:, 0])
    assert not result.mask[50, 5]


def test_segment_frame_rejects_constant_frame():
    frame = np.full((40, 48), 1000, dtype=np.uint16)

    result = segment_frame(frame, SegmentationConfig(min_object_area_px=20))

    assert not result.valid
    assert result.failure_reason == "no_foreground_signal"
    assert result.mask.shape == frame.shape
    assert not np.any(result.mask)


def test_background_estimation_falls_back_when_no_background_pixels_are_available():
    frame = np.full((32, 32), 2000, dtype=np.uint16)
    rough_mask = np.ones(frame.shape, dtype=bool)

    background = estimate_background_masked_poly(frame, rough_mask, order=2, dilation_px=4)

    assert background.shape == frame.shape
    np.testing.assert_allclose(background, 2000.0)


def test_segment_stack_and_mask_writer_round_trip_binary_uint8_masks(tmp_path):
    stack = synthetic_relaxing_ellipse_stack(n_frames=4, shape=(64, 72), radius_px=12.0)
    result = segment_stack(stack, SegmentationConfig(min_object_area_px=50))

    assert result.masks.shape == stack.shape
    assert result.masks.dtype == bool
    assert all(frame_result.valid for frame_result in result.frame_results)

    path = write_mask_stack(tmp_path / "masks.tif", result.masks)
    written = load_tiff_stack(path)

    assert written.shape == stack.shape
    assert written.dtype == np.uint8
    assert set(np.unique(written)).issubset({0, 255})

    qc_path = write_segmentation_qc_csv(tmp_path / "segmentation_qc.csv", result.qc_rows)
    assert qc_path.exists()


def test_segment_frame_expands_dim_droplet_boundary_beyond_bright_core():
    shape = (96, 104)
    frame = _gradient_frame(shape)
    full_droplet = _disk_mask(shape, center=(48, 52), radius=20)
    bright_core = _disk_mask(shape, center=(48, 52), radius=7)
    frame[full_droplet] += 260
    frame[bright_core] += 1800

    result = segment_frame(
        frame.astype(np.uint16),
        SegmentationConfig(
            min_object_area_px=80,
            snr_threshold=2.0,
            background_poly_order=1,
            background_mask_dilation_px=12,
            boundary_threshold_percentile=55.0,
        ),
    )

    assert result.valid
    assert np.count_nonzero(result.mask) >= 0.75 * np.count_nonzero(full_droplet)
    assert result.mask[48, 52]
    assert result.mask[48, 70]


def test_convex_hull_outer_envelope_fills_false_boundary_notch():
    shape = (96, 104)
    frame = _gradient_frame(shape)
    full_droplet = _disk_mask(shape, center=(48, 52), radius=24)
    false_notch = np.zeros(shape, dtype=bool)
    false_notch[24:50, 49:56] = True
    frame[full_droplet] += 1400
    frame[full_droplet & false_notch] -= 1400

    plain = segment_frame(
        frame.astype(np.uint16),
        SegmentationConfig(min_object_area_px=80, outer_envelope_mode="none"),
    )
    repaired = segment_frame(
        frame.astype(np.uint16),
        SegmentationConfig(min_object_area_px=80, outer_envelope_mode="convex_hull"),
    )

    assert plain.valid
    assert repaired.valid
    assert not plain.mask[36, 52]
    assert repaired.mask[36, 52]
    assert repaired.area_px > plain.area_px
    assert repaired.selected_source is not None
    assert repaired.selected_source.endswith("+convex_hull")


def test_temporal_segmentation_rejects_single_frame_centroid_jump():
    shape = (90, 100)
    stack = np.zeros((3, *shape), dtype=np.uint16)
    target = _disk_mask(shape, center=(45, 50), radius=16)
    off_target = _disk_mask(shape, center=(22, 78), radius=7)
    for frame_index in (0, 2):
        stack[frame_index] = _gradient_frame(shape)
        stack[frame_index, target] += 1500
    stack[1] = _gradient_frame(shape)
    stack[1, off_target] += 1800

    result = segment_stack(
        stack,
        SegmentationConfig(
            min_object_area_px=80,
            segmentation_mode="temporal",
            max_centroid_jump_px=12.0,
            min_area_ratio_to_reference=0.5,
        ),
    )

    assert result.frame_results[0].valid
    assert not result.frame_results[1].valid
    assert result.frame_results[1].failure_reason == "no_temporally_consistent_object"
    assert result.frame_results[2].valid
