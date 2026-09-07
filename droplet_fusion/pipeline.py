"""Top-level pipeline orchestration."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from skimage import measure
from tqdm import tqdm

from droplet_fusion.config import PipelineConfig
from droplet_fusion.ellipse import fit_ellipse_to_mask_boundary
from droplet_fusion.fitting import (
    SUMMARY_RESULT_COLUMNS,
    analyze_movie_measurements,
    load_manual_fit_starts,
    mark_used_for_fit,
)
from droplet_fusion.io import (
    find_tiff_files,
    load_tiff_stack,
    prepare_movie_output_directory,
    prepare_output_directory,
)
from droplet_fusion.segmentation import segment_stack, write_mask_stack, write_segmentation_qc_csv
from droplet_fusion.visualization import (
    write_ar_fit_plot,
    write_inverse_capillary_velocity_summary_plot,
    write_overlay_video,
    write_tau_vs_radius_plot,
)


FRAME_MEASUREMENT_COLUMNS = [
    "filename",
    "frame",
    "time_s",
    "valid_mask",
    "used_for_fit",
    "failure_reason",
    "area_px",
    "area_um2",
    "centroid_x_px",
    "centroid_y_px",
    "major_axis_px",
    "minor_axis_px",
    "major_axis_um",
    "minor_axis_um",
    "AR",
    "ellipse_center_x_px",
    "ellipse_center_y_px",
    "ellipse_angle_rad",
]


@dataclass(frozen=True)
class PipelineRunResult:
    """Result metadata for a pipeline invocation."""

    output_dir: Path
    config_path: Path
    log_path: Path
    message: str


def run_pipeline(config: PipelineConfig, *, show_progress: bool = True) -> PipelineRunResult:
    """Run the currently implemented partial pipeline.

    When ``show_progress`` is true a live per-movie progress bar is written to the
    terminal so the user can see that the run is advancing and which stage each
    movie is in.
    """

    config.validate()
    prepare_output_directory(config.output_dir, overwrite=config.overwrite)
    tiff_paths = find_tiff_files(config.data_dir)
    manual_fit_starts = load_manual_fit_starts(config.fitting.manual_fit_starts)

    if show_progress:
        print(f"Found {len(tiff_paths)} movie(s) in {config.data_dir}. Analyzing...", flush=True)

    movie_summaries = []
    summary_rows = []
    progress = tqdm(tiff_paths, desc="Analyzing movies", unit="movie", disable=not show_progress)
    for movie_path in progress:
        progress.set_postfix_str(f"{movie_path.name}: loading")
        stack = load_tiff_stack(movie_path)
        movie_dir = prepare_movie_output_directory(config.output_dir, movie_path, overwrite=config.overwrite)
        progress.set_postfix_str(f"{movie_path.name}: segmenting")
        segmentation_result = segment_stack(stack, config.segmentation)
        mask_path = write_mask_stack(movie_dir / "masks.tif", segmentation_result.masks)
        segmentation_qc_path = None
        if config.segmentation.write_segmentation_qc:
            segmentation_qc_path = write_segmentation_qc_csv(movie_dir / "segmentation_qc.csv", segmentation_result.qc_rows)
        progress.set_postfix_str(f"{movie_path.name}: measuring & fitting")
        measurement_rows = build_frame_measurements(
            filename=movie_path.name,
            segmentation_result=segmentation_result,
            seconds_per_frame=config.seconds_per_frame,
            um_per_pixel=config.um_per_pixel,
        )
        movie_fit_result = analyze_movie_measurements(
            filename=movie_path.name,
            rows=measurement_rows,
            fit_config=config.fitting,
            um_per_pixel=config.um_per_pixel,
            manual_fit_starts=manual_fit_starts,
        )
        mark_used_for_fit(measurement_rows, movie_fit_result.used_frame_indices)
        measurements_path = write_frame_measurements_csv(movie_dir / "frame_measurements.csv", measurement_rows)
        movie_result_row = movie_fit_result.to_summary_row()
        movie_result_path = write_movie_result_json(movie_dir / "movie_result.json", movie_result_row)
        ar_fit_plot_path = write_ar_fit_plot(movie_dir / "AR_fit.png", measurement_rows, movie_result_row)
        overlay_video_path = None
        if config.visualization.make_videos:
            progress.set_postfix_str(f"{movie_path.name}: rendering video")
            overlay_video_path = write_overlay_video(
                movie_dir / "overlay.mp4",
                stack,
                segmentation_result.masks,
                measurement_rows,
                fps=config.visualization.video_fps,
            )
        summary_rows.append(movie_result_row)

        n_valid_frames = sum(result.valid for result in segmentation_result.frame_results)
        n_valid_measurements = sum(_row_has_valid_ar(row) for row in measurement_rows)
        movie_summaries.append(
            {
                "filename": movie_path.name,
                "n_frames": int(stack.shape[0]),
                "n_valid_segmentation_frames": int(n_valid_frames),
                "n_valid_measurement_frames": int(n_valid_measurements),
                "status": movie_fit_result.status,
                "notes": movie_fit_result.notes,
                "mask_path": str(mask_path),
                "segmentation_qc_path": str(segmentation_qc_path) if segmentation_qc_path is not None else None,
                "frame_measurements_path": str(measurements_path),
                "movie_result_path": str(movie_result_path),
                "ar_fit_plot_path": str(ar_fit_plot_path),
                "overlay_video_path": str(overlay_video_path) if overlay_video_path is not None else None,
                "frame_failures": [
                    {
                        "frame": frame_index,
                        "failure_reason": measurement_rows[frame_index]["failure_reason"],
                        "background_method": result.background_method,
                        "candidate_count": result.candidate_count,
                        "selected_source": result.selected_source,
                        "score": result.score,
                    }
                    for frame_index, result in enumerate(segmentation_result.frame_results)
                    if measurement_rows[frame_index]["failure_reason"]
                ],
            }
        )

    if show_progress:
        print("Writing dataset summary and plots...", flush=True)

    config_path = config.output_dir / "run_config.json"
    log_path = config.output_dir / "run_log.txt"
    summary_path = write_summary_results_csv(config.output_dir / "summary_results.csv", summary_rows)
    summary_plot_path, dataset_stats = write_inverse_capillary_velocity_summary_plot(
        config.output_dir / "inverse_capillary_velocity_summary.png",
        summary_rows,
    )
    tau_vs_radius_plot_path, tau_vs_radius_stats = write_tau_vs_radius_plot(
        config.output_dir / "tau_vs_radius.png",
        summary_rows,
    )

    run_started_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "run_started_at_utc": run_started_at,
        "pipeline_state": "visualization",
        "config": config.to_dict(),
        "movies": movie_summaries,
        "summary_results_path": str(summary_path),
        "summary_plot_path": str(summary_plot_path),
        "tau_vs_radius_plot_path": str(tau_vs_radius_plot_path),
        "dataset_inverse_capillary_velocity_stats": dataset_stats,
        "dataset_tau_vs_radius_stats": tau_vs_radius_stats,
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    log_lines = [
        "Droplet fusion pipeline visualization run",
        f"Started at UTC: {run_started_at}",
        "State: visualization",
        f"Input TIFF files: {len(tiff_paths)}",
        "Implemented: TIFF discovery/loading, temporally scored segmentation, binary mask stack output, segmentation QC CSV, ellipse fitting, frame measurements, AR fitting, R estimation, movie result JSON, summary CSV, AR plots, overlay videos, dataset summary plot, tau vs radius plot",
        (
            "Dataset inverse capillary velocity: "
            f"n={dataset_stats['n_successful']}, "
            f"mean={dataset_stats['mean_inverse_capillary_velocity_s_per_um']} s/um, "
            f"std={dataset_stats['std_inverse_capillary_velocity_s_per_um']} s/um; "
            f"plot: {summary_plot_path}"
        ),
        (
            "Dataset tau vs radius: "
            f"n={tau_vs_radius_stats['n_points']}, "
            f"slope={tau_vs_radius_stats['slope_tau_per_radius_s_per_um']} s/um, "
            f"pearson_r={tau_vs_radius_stats['pearson_r']}; "
            f"plot: {tau_vs_radius_plot_path}"
        ),
    ]
    for movie_summary in movie_summaries:
        log_lines.append(
            (
                "Movie {filename}: {n_valid_measurement_frames}/{n_frames} valid measurement frames; "
                "status: {status}; masks: {mask_path}; segmentation QC: {segmentation_qc_path}; "
                "measurements: {frame_measurements_path}; "
                "result: {movie_result_path}; AR plot: {ar_fit_plot_path}; overlay: {overlay_video_path}"
            ).format(
                **movie_summary
            )
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return PipelineRunResult(
        output_dir=config.output_dir,
        config_path=config_path,
        log_path=log_path,
        message=(
            "Visualization pipeline completed for "
            f"{len(tiff_paths)} TIFF file(s): wrote masks, frame_measurements.csv, movie_result.json, "
            f"segmentation_qc.csv when enabled, AR_fit.png, overlay videos when enabled, {summary_path}, {summary_plot_path}, "
            f"{tau_vs_radius_plot_path}, "
            f"{config_path}, and {log_path}."
        ),
    )


def build_frame_measurements(
    *,
    filename: str,
    segmentation_result,
    seconds_per_frame: float,
    um_per_pixel: float,
) -> list[dict[str, Any]]:
    """Build frame-level measurement rows from segmentation masks."""

    rows: list[dict[str, Any]] = []
    for frame_index, frame_result in enumerate(segmentation_result.frame_results):
        row = _empty_measurement_row(
            filename=filename,
            frame_index=frame_index,
            seconds_per_frame=seconds_per_frame,
        )
        row["valid_mask"] = bool(frame_result.valid)

        if not frame_result.valid:
            row["failure_reason"] = frame_result.failure_reason or "invalid_mask"
            rows.append(row)
            continue

        mask = np.asarray(frame_result.mask, dtype=bool)
        mask_measurements = _mask_measurements(mask, um_per_pixel=um_per_pixel)
        row.update(mask_measurements)

        ellipse = fit_ellipse_to_mask_boundary(mask)
        if not ellipse.valid:
            row["failure_reason"] = ellipse.failure_reason or "ellipse_fit_failed"
            rows.append(row)
            continue

        row.update(
            {
                "failure_reason": "",
                "major_axis_px": ellipse.major_axis_px,
                "minor_axis_px": ellipse.minor_axis_px,
                "major_axis_um": ellipse.major_axis_px * um_per_pixel,
                "minor_axis_um": ellipse.minor_axis_px * um_per_pixel,
                "AR": ellipse.aspect_ratio,
                "ellipse_center_x_px": ellipse.center_x_px,
                "ellipse_center_y_px": ellipse.center_y_px,
                "ellipse_angle_rad": ellipse.angle_rad,
            }
        )
        rows.append(row)

    return rows


def write_frame_measurements_csv(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    """Write frame-level measurements using the guide's required column order."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_MEASUREMENT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_movie_result_json(path: Path | str, row: dict[str, Any]) -> Path:
    """Write a per-movie result JSON file."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return out_path


def write_summary_results_csv(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    """Write dataset-level summary rows using the guide's required columns."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _empty_measurement_row(*, filename: str, frame_index: int, seconds_per_frame: float) -> dict[str, Any]:
    nan = float("nan")
    return {
        "filename": filename,
        "frame": frame_index,
        "time_s": frame_index * seconds_per_frame,
        "valid_mask": False,
        "used_for_fit": False,
        "failure_reason": "",
        "area_px": nan,
        "area_um2": nan,
        "centroid_x_px": nan,
        "centroid_y_px": nan,
        "major_axis_px": nan,
        "minor_axis_px": nan,
        "major_axis_um": nan,
        "minor_axis_um": nan,
        "AR": nan,
        "ellipse_center_x_px": nan,
        "ellipse_center_y_px": nan,
        "ellipse_angle_rad": nan,
    }


def _mask_measurements(mask: np.ndarray, *, um_per_pixel: float) -> dict[str, float]:
    area_px = float(np.count_nonzero(mask))
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if regions:
        region = max(regions, key=lambda item: item.area)
        centroid_y, centroid_x = (float(value) for value in region.centroid)
    else:
        centroid_x = float("nan")
        centroid_y = float("nan")

    return {
        "area_px": area_px,
        "area_um2": area_px * um_per_pixel**2,
        "centroid_x_px": centroid_x,
        "centroid_y_px": centroid_y,
    }


def _row_has_valid_ar(row: dict[str, Any]) -> bool:
    value = row.get("AR", float("nan"))
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
