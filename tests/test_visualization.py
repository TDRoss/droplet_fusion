from __future__ import annotations

import csv
import json
import math

import numpy as np

from droplet_fusion.cli import build_parser, config_from_args
from droplet_fusion.io import write_synthetic_tiff
from droplet_fusion.pipeline import run_pipeline
from droplet_fusion.visualization import (
    write_inverse_capillary_velocity_summary_plot,
    write_overlay_video,
    write_tau_vs_radius_plot,
)


def test_pipeline_writes_visualization_outputs_when_videos_disabled(tmp_path):
    data_dir = tmp_path / "data"
    write_synthetic_tiff(data_dir / "movie_001.tif", n_frames=8, shape=(64, 72), radius_px=12.0)
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

    result = run_pipeline(config_from_args(args))

    movie_dir = tmp_path / "output" / "movie_001"
    assert (movie_dir / "AR_fit.png").stat().st_size > 0
    assert not (movie_dir / "overlay.mp4").exists()
    assert (tmp_path / "output" / "inverse_capillary_velocity_summary.png").stat().st_size > 0
    assert (tmp_path / "output" / "tau_vs_radius.png").stat().st_size > 0

    payload = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert payload["summary_plot_path"].endswith("inverse_capillary_velocity_summary.png")
    assert payload["tau_vs_radius_plot_path"].endswith("tau_vs_radius.png")
    assert payload["dataset_tau_vs_radius_stats"]["n_points"] == 1
    stats = payload["dataset_inverse_capillary_velocity_stats"]
    assert stats["n_successful"] == 1
    assert math.isfinite(stats["mean_inverse_capillary_velocity_s_per_um"])

    with (tmp_path / "output" / "summary_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["inverse_capillary_velocity_s_per_um"]) > 0.0


def test_overlay_video_writer_creates_mp4(tmp_path):
    yy, xx = np.indices((32, 36))
    masks = []
    stack = np.full((3, 32, 36), 1000, dtype=np.uint16)
    rows = []
    for frame in range(3):
        mask = ((xx - 18) / (8 - frame)) ** 2 + ((yy - 16) / (5 + frame * 0.5)) ** 2 <= 1.0
        masks.append(mask)
        stack[frame, mask] = 40000
        rows.append(
            {
                "frame": frame,
                "time_s": float(frame),
                "used_for_fit": True,
                "failure_reason": "",
                "AR": 1.6 - frame * 0.1,
                "major_axis_px": 16.0 - frame,
                "minor_axis_px": 10.0 + frame,
                "ellipse_center_x_px": 18.0,
                "ellipse_center_y_px": 16.0,
                "ellipse_angle_rad": 0.0,
            }
        )

    out_path = write_overlay_video(tmp_path / "overlay.mp4", stack, np.asarray(masks), rows, fps=3.0)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_summary_plot_reports_mean_and_std(tmp_path):
    rows = [
        {"filename": "a.tif", "inverse_capillary_velocity_s_per_um": 2.0},
        {"filename": "b.tif", "inverse_capillary_velocity_s_per_um": 4.0},
        {"filename": "bad.tif", "inverse_capillary_velocity_s_per_um": float("nan")},
    ]

    out_path, stats = write_inverse_capillary_velocity_summary_plot(tmp_path / "summary.png", rows)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert stats["n_successful"] == 2
    assert stats["n_failed_or_nonfinite"] == 1
    assert stats["mean_inverse_capillary_velocity_s_per_um"] == 3.0
    assert stats["std_inverse_capillary_velocity_s_per_um"] == math.sqrt(2.0)


def test_tau_vs_radius_plot_reports_slope_and_correlation(tmp_path):
    rows = [
        {"filename": "a.tif", "R_um": 2.0, "tau_fusion_s": 1.0, "tau_fusion_std_s": 0.1},
        {"filename": "b.tif", "R_um": 4.0, "tau_fusion_s": 2.0, "tau_fusion_std_s": float("nan")},
        {"filename": "bad.tif", "R_um": float("nan"), "tau_fusion_s": float("nan")},
    ]

    out_path, stats = write_tau_vs_radius_plot(tmp_path / "tau_vs_radius.png", rows)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert stats["n_points"] == 2
    assert stats["n_omitted"] == 1
    assert stats["slope_tau_per_radius_s_per_um"] == 0.5
    assert math.isclose(stats["pearson_r"], 1.0)


def test_tau_vs_radius_plot_handles_no_valid_points(tmp_path):
    out_path, stats = write_tau_vs_radius_plot(
        tmp_path / "tau_vs_radius.png",
        [{"filename": "bad.tif", "R_um": float("nan"), "tau_fusion_s": float("nan")}],
    )

    assert out_path.exists()
    assert stats["n_points"] == 0
    assert math.isnan(stats["slope_tau_per_radius_s_per_um"])
