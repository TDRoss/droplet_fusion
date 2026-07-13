from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from droplet_fusion.cli import build_parser, config_from_args
from droplet_fusion.config import FitConfig
from droplet_fusion.fitting import (
    analyze_movie_measurements,
    ar_model,
    estimate_final_radius,
    fit_ar_relaxation,
)
from droplet_fusion.io import write_synthetic_tiff
from droplet_fusion.pipeline import run_pipeline


def _synthetic_measurement_rows(
    *,
    n_frames: int = 16,
    seconds_per_frame: float = 2.0,
    A: float = 0.8,
    tau_s: float = 8.0,
    area_px: float = 314.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in range(n_frames):
        time_s = frame * seconds_per_frame
        rows.append(
            {
                "filename": "movie.tif",
                "frame": frame,
                "time_s": time_s,
                "valid_mask": True,
                "used_for_fit": False,
                "failure_reason": "",
                "area_px": area_px,
                "area_um2": area_px,
                "centroid_x_px": 10.0,
                "centroid_y_px": 10.0,
                "major_axis_px": 1.0,
                "minor_axis_px": 1.0,
                "major_axis_um": 1.0,
                "minor_axis_um": 1.0,
                "AR": float(ar_model(np.array([time_s]), A, tau_s)[0]),
                "ellipse_center_x_px": 10.0,
                "ellipse_center_y_px": 10.0,
                "ellipse_angle_rad": 0.0,
            }
        )
    return rows


def test_fit_ar_relaxation_recovers_synthetic_tau():
    rows = _synthetic_measurement_rows(A=0.7, tau_s=10.0)

    result = fit_ar_relaxation(
        rows,
        FitConfig(fit_start_mode="first_valid", trim_tail_to_floor=False),
        filename="movie.tif",
    )

    assert result.valid
    assert result.fit_start_frame == 0
    assert result.n_fit_frames == len(rows)
    assert result.A == pytest.approx(0.7, rel=0.05)
    assert result.tau_fusion_s == pytest.approx(10.0, rel=0.05)
    assert result.fit_rmse < 1e-6
    assert result.fit_r2 == pytest.approx(1.0)


def test_trim_tail_to_floor_drops_flat_tail_but_keeps_tau():
    rows = _synthetic_measurement_rows(n_frames=16, A=0.7, tau_s=10.0)

    trimmed = fit_ar_relaxation(
        rows,
        FitConfig(fit_start_mode="first_valid", trim_tail_to_floor=True, tail_floor_fraction=0.1),
        filename="movie.tif",
    )
    full = fit_ar_relaxation(
        rows,
        FitConfig(fit_start_mode="first_valid", trim_tail_to_floor=False),
        filename="movie.tif",
    )

    assert trimmed.valid
    assert trimmed.n_fit_frames < full.n_fit_frames
    assert trimmed.tau_fusion_s == pytest.approx(10.0, rel=0.05)


def test_float_baseline_recovers_elevated_asymptote():
    n_frames = 20
    seconds_per_frame = 2.0
    A, tau_s, true_baseline = 0.6, 8.0, 1.05
    rows = _synthetic_measurement_rows(n_frames=n_frames, seconds_per_frame=seconds_per_frame)
    for frame, row in enumerate(rows):
        t = frame * seconds_per_frame
        row["AR"] = true_baseline + A * math.exp(-t / tau_s)

    fixed = fit_ar_relaxation(
        rows,
        FitConfig(fit_start_mode="first_valid", trim_tail_to_floor=False, float_baseline=False),
        filename="movie.tif",
    )
    floating = fit_ar_relaxation(
        rows,
        FitConfig(fit_start_mode="first_valid", trim_tail_to_floor=False, float_baseline=True),
        filename="movie.tif",
    )

    assert floating.valid
    assert floating.baseline == pytest.approx(true_baseline, abs=0.01)
    assert floating.tau_fusion_s == pytest.approx(tau_s, rel=0.05)
    # The free asymptote fits data that floors above 1 far better than the pinned model.
    assert floating.fit_r2 > fixed.fit_r2
    assert fixed.baseline == pytest.approx(1.0)


def test_decay_onset_skips_pre_collapse_plateau():
    rows = _synthetic_measurement_rows(n_frames=16, A=0.5, tau_s=6.0)
    plateau_ar = float(rows[0]["AR"])
    for index in range(4):
        rows[index]["AR"] = plateau_ar

    result = fit_ar_relaxation(
        rows,
        FitConfig(fit_start_mode="decay_onset", trim_tail_to_floor=False),
        filename="movie.tif",
    )

    assert result.valid
    assert result.fit_start_frame >= 3


def test_fit_ar_relaxation_uses_max_ar_start_frame():
    rows = _synthetic_measurement_rows(n_frames=10, A=0.5, tau_s=5.0)
    rows[0]["AR"] = 1.1
    rows[3]["AR"] = 1.9

    result = fit_ar_relaxation(rows, FitConfig(fit_start_mode="max_AR"), filename="movie.tif", min_fit_frames=5)

    assert result.valid
    assert result.fit_start_frame == 3
    assert min(result.used_frame_indices) == 3


def test_fit_ar_relaxation_fails_when_max_ar_is_last_valid_frame():
    rows = _synthetic_measurement_rows(n_frames=10, A=0.5, tau_s=5.0)
    rows[-1]["AR"] = 2.0

    result = fit_ar_relaxation(rows, FitConfig(fit_start_mode="max_AR"), filename="movie.tif")

    assert not result.valid
    assert result.failure_reason == "max_ar_at_last_frame"
    assert result.fit_start_frame == 9


def test_estimate_final_radius_uses_median_of_final_valid_areas():
    rows = _synthetic_measurement_rows(n_frames=8)
    for frame, row in enumerate(rows):
        row["area_px"] = 100.0 + frame

    result = estimate_final_radius(rows, n_final_frames=3, um_per_pixel=0.5)

    assert result.valid
    assert result.n_frames_used == 3
    assert result.R_px == pytest.approx(math.sqrt(106.0 / math.pi))
    assert result.R_um == pytest.approx(math.sqrt(106.0 / math.pi) * 0.5)


def test_analyze_movie_measurements_returns_inverse_capillary_velocity():
    rows = _synthetic_measurement_rows(tau_s=12.0, area_px=400.0)

    result = analyze_movie_measurements(
        filename="movie.tif",
        rows=rows,
        fit_config=FitConfig(fit_start_mode="first_valid"),
        um_per_pixel=0.25,
    )

    assert result.status == "ok"
    assert result.tau_fusion_s == pytest.approx(12.0, rel=0.05)
    assert result.R_um == pytest.approx(math.sqrt(400.0 / math.pi) * 0.25)
    assert result.inverse_capillary_velocity_s_per_um == pytest.approx(result.tau_fusion_s / result.R_um)


def test_analyze_movie_measurements_warns_on_low_fit_r2():
    rows = _synthetic_measurement_rows(tau_s=12.0, area_px=400.0)
    rows[4]["AR"] = 1.9

    result = analyze_movie_measurements(
        filename="movie.tif",
        rows=rows,
        fit_config=FitConfig(fit_start_mode="first_valid", min_fit_r2_warning=0.99),
        um_per_pixel=0.25,
    )

    assert result.status == "warning"
    assert "low_fit_r2" in result.notes
    assert "--fit-start-mode manual" in result.notes


def test_analyze_movie_measurements_recommends_manual_start_when_max_ar_is_last():
    rows = _synthetic_measurement_rows(n_frames=10, tau_s=12.0, area_px=400.0)
    rows[-1]["AR"] = 2.1

    result = analyze_movie_measurements(
        filename="movie.tif",
        rows=rows,
        fit_config=FitConfig(fit_start_mode="max_AR"),
        um_per_pixel=0.25,
    )

    assert result.status == "failed"
    assert "max_ar_at_last_frame" in result.notes
    assert "--manual-fit-starts" in result.notes


def test_pipeline_writes_movie_result_and_summary_csv(tmp_path):
    data_dir = tmp_path / "data"
    write_synthetic_tiff(data_dir / "movie_001.tif", n_frames=10, shape=(64, 72), radius_px=12.0, tau_frames=4.0)
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "output"),
            "--seconds-per-frame",
            "2.0",
            "--um-per-pixel",
            "0.5",
            "--fit-start-mode",
            "first_valid",
            "--no-make-videos",
        ]
    )
    config = config_from_args(args)

    run_pipeline(config)

    movie_result_path = tmp_path / "output" / "movie_001" / "movie_result.json"
    summary_path = tmp_path / "output" / "summary_results.csv"
    assert movie_result_path.exists()
    assert summary_path.exists()

    movie_result = json.loads(movie_result_path.read_text(encoding="utf-8"))
    assert movie_result["filename"] == "movie_001.tif"
    assert movie_result["status"] in {"ok", "warning"}
    assert movie_result["n_fit_frames"] >= 5

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["filename"] == "movie_001.tif"
    assert float(rows[0]["R_um"]) > 0.0
