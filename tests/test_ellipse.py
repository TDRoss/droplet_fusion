from __future__ import annotations

import csv
import math

import numpy as np
import pytest
from skimage.draw import ellipse

from droplet_fusion.cli import build_parser, config_from_args
from droplet_fusion.config import SegmentationConfig
from droplet_fusion.ellipse import fit_ellipse_to_mask_boundary
from droplet_fusion.io import write_synthetic_tiff
from droplet_fusion.pipeline import FRAME_MEASUREMENT_COLUMNS, build_frame_measurements, run_pipeline
from droplet_fusion.segmentation import segment_stack


def _ellipse_mask(
    shape: tuple[int, int] = (96, 112),
    *,
    center: tuple[float, float] = (48.0, 56.0),
    semi_major_px: float = 24.0,
    semi_minor_px: float = 12.0,
) -> np.ndarray:
    rr, cc = ellipse(
        r=center[0],
        c=center[1],
        r_radius=semi_minor_px,
        c_radius=semi_major_px,
        shape=shape,
    )
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def test_fit_ellipse_to_mask_boundary_recovers_axis_lengths_and_ar():
    mask = _ellipse_mask(semi_major_px=22.0, semi_minor_px=11.0)

    result = fit_ellipse_to_mask_boundary(mask)

    assert result.valid
    assert result.failure_reason is None
    assert result.center_x_px == pytest.approx(56.0, abs=0.75)
    assert result.center_y_px == pytest.approx(48.0, abs=0.75)
    assert result.major_axis_px == pytest.approx(44.0, rel=0.08)
    assert result.minor_axis_px == pytest.approx(22.0, rel=0.08)
    assert result.aspect_ratio == pytest.approx(2.0, rel=0.10)


def test_fit_ellipse_to_mask_boundary_rejects_too_few_pixels():
    mask = np.zeros((20, 24), dtype=bool)
    mask[10, 12] = True

    result = fit_ellipse_to_mask_boundary(mask)

    assert not result.valid
    assert result.failure_reason == "too_few_foreground_pixels"
    assert math.isnan(result.aspect_ratio)


def test_build_frame_measurements_uses_supplied_time_and_pixel_units():
    seconds_per_frame = 600.0
    um_per_pixel = 0.2125
    stack = np.zeros((2, 72, 80), dtype=np.uint16)
    mask = _ellipse_mask((72, 80), center=(36.0, 40.0), semi_major_px=12.0, semi_minor_px=8.0)
    stack[:, mask] = 40000
    segmentation = segment_stack(stack, SegmentationConfig(min_object_area_px=20))

    rows = build_frame_measurements(
        filename="movie.tif",
        segmentation_result=segmentation,
        seconds_per_frame=seconds_per_frame,
        um_per_pixel=um_per_pixel,
    )

    assert len(rows) == 2
    assert rows[1]["time_s"] == 600.0
    assert rows[0]["valid_mask"] is True
    assert rows[0]["failure_reason"] == ""
    assert rows[0]["area_um2"] == pytest.approx(rows[0]["area_px"] * um_per_pixel**2)
    assert rows[0]["major_axis_um"] == pytest.approx(rows[0]["major_axis_px"] * um_per_pixel)
    assert rows[0]["minor_axis_um"] == pytest.approx(rows[0]["minor_axis_px"] * um_per_pixel)
    assert rows[0]["AR"] >= 1.0


def test_pipeline_writes_frame_measurements_csv(tmp_path):
    data_dir = tmp_path / "data"
    write_synthetic_tiff(data_dir / "movie_001.tif", n_frames=3, shape=(64, 72), radius_px=12.0)
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "output"),
            "--seconds-per-frame",
            "600",
            "--um-per-pixel",
            "0.2125",
            "--no-make-videos",
        ]
    )
    config = config_from_args(args)

    run_pipeline(config)

    measurements_path = tmp_path / "output" / "movie_001" / "frame_measurements.csv"
    assert measurements_path.exists()

    with measurements_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert list(rows[0].keys()) == FRAME_MEASUREMENT_COLUMNS
    assert float(rows[1]["time_s"]) == 600.0
    assert float(rows[0]["major_axis_um"]) == pytest.approx(float(rows[0]["major_axis_px"]) * 0.2125)
    assert float(rows[0]["AR"]) >= 1.0
