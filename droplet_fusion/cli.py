"""Command-line interface for the droplet fusion pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from droplet_fusion.config import (
    FitConfig,
    PipelineConfig,
    SegmentationConfig,
    VisualizationConfig,
)
from droplet_fusion.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="droplet-fusion",
        description="Analyze droplet-fusion TIFF time series.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing input .tif/.tiff stacks.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for pipeline outputs.")
    parser.add_argument("--seconds-per-frame", type=float, default=1.0, help="Frame interval in seconds.")
    parser.add_argument("--um-per-pixel", type=float, default=1.0, help="Pixel size in microns.")
    parser.add_argument("--min-object-area-px", type=int, default=100, help="Minimum segmented object area in pixels.")
    parser.add_argument("--max-object-area-px", type=int, default=None, help="Maximum segmented object area in pixels.")
    parser.add_argument("--background-poly-order", type=int, default=2, help="Polynomial order for background fitting.")
    parser.add_argument(
        "--background-mask-dilation-px",
        type=int,
        default=10,
        help="Rough foreground-mask dilation before background fitting.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["otsu", "yen", "li", "triangle"],
        default="otsu",
        help="Threshold method for corrected frames.",
    )
    parser.add_argument("--snr-threshold", type=float, default=3.0, help="Minimum signal-to-noise threshold.")
    parser.add_argument(
        "--segmentation-mode",
        choices=["per_frame", "temporal"],
        default="temporal",
        help="Mask selection mode: independent per-frame scoring or temporally aware scoring.",
    )
    parser.add_argument(
        "--min-area-ratio-to-reference",
        type=float,
        default=0.35,
        help="Minimum selected-mask area ratio relative to temporal reference before a frame is rejected.",
    )
    parser.add_argument(
        "--max-centroid-jump-px",
        type=float,
        default=None,
        help="Maximum centroid jump from temporal reference; defaults to a fraction of image diagonal.",
    )
    parser.add_argument(
        "--mask-closing-radius-px",
        type=int,
        default=2,
        help="Morphological closing radius applied to candidate masks.",
    )
    parser.add_argument(
        "--boundary-threshold-percentile",
        type=float,
        default=60.0,
        help="Low corrected-intensity percentile used for boundary-growth candidates.",
    )
    parser.add_argument(
        "--outer-envelope-mode",
        choices=["none", "convex_hull"],
        default="none",
        help="Optional candidate-mask repair; convex_hull fills concavities in each candidate mask.",
    )
    parser.add_argument(
        "--growth-floor-mode",
        choices=["image_threshold", "frame_percentile"],
        default="image_threshold",
        help=(
            "How far flood-fill growth candidates may extend below their seed. "
            "image_threshold stops growth at the droplet/background threshold "
            "(correct for sharp-edged droplets); frame_percentile lets growth "
            "reach frame-percentile levels, for droplets with a genuinely dim "
            "outer rim around a much brighter core."
        ),
    )
    parser.add_argument(
        "--write-segmentation-qc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-movie segmentation_qc.csv files with selected-mask diagnostics.",
    )
    parser.add_argument(
        "--fit-start-mode",
        choices=["max_AR", "manual", "first_valid", "decay_onset"],
        default="decay_onset",
        help=(
            "Rule for selecting the first frame used in AR fitting. The default "
            "'decay_onset' smooths AR and starts at the top of the sustained decay, "
            "skipping pre-collapse plateaus so the fit is robust to temporal cropping "
            "and noise."
        ),
    )
    parser.add_argument(
        "--trim-tail-to-floor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the near-circular AR tail from the fit window so it does not dominate the residual.",
    )
    parser.add_argument(
        "--tail-floor-fraction",
        type=float,
        default=0.1,
        help="Keep fit frames down to within this fraction of the descent floor before trimming the tail.",
    )
    parser.add_argument(
        "--float-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit AR(t)=b+A exp(-t/tau) with a free asymptote b>=1 instead of pinning it to 1.",
    )
    parser.add_argument(
        "--manual-fit-starts",
        type=Path,
        default=None,
        help="CSV with filename,fit_start_frame columns for manual fit starts.",
    )
    parser.add_argument(
        "--n-final-frames-for-r",
        type=int,
        default=5,
        help="Number of final valid frames used to estimate the fused radius.",
    )
    parser.add_argument(
        "--min-fit-r2-warning",
        type=float,
        default=0.8,
        help="Warn when a successful AR relaxation fit has R2 below this threshold.",
    )
    parser.add_argument(
        "--fail-max-ar-at-last-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail max_AR fits when the maximum AR occurs at the last valid AR frame.",
    )
    parser.add_argument(
        "--make-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable QC overlay video generation.",
    )
    parser.add_argument("--video-fps", type=float, default=10.0, help="Frames per second for QC overlay videos.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting outputs from a previous run.")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a live progress bar while movies are analyzed (use --no-progress for quiet/log-file runs).",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seconds_per_frame=args.seconds_per_frame,
        um_per_pixel=args.um_per_pixel,
        overwrite=args.overwrite,
        segmentation=SegmentationConfig(
            min_object_area_px=args.min_object_area_px,
            max_object_area_px=args.max_object_area_px,
            background_poly_order=args.background_poly_order,
            background_mask_dilation_px=args.background_mask_dilation_px,
            threshold_method=args.threshold_method,
            snr_threshold=args.snr_threshold,
            segmentation_mode=args.segmentation_mode,
            min_area_ratio_to_reference=args.min_area_ratio_to_reference,
            max_centroid_jump_px=args.max_centroid_jump_px,
            mask_closing_radius_px=args.mask_closing_radius_px,
            boundary_threshold_percentile=args.boundary_threshold_percentile,
            outer_envelope_mode=args.outer_envelope_mode,
            growth_floor_mode=args.growth_floor_mode,
            write_segmentation_qc=args.write_segmentation_qc,
        ),
        fitting=FitConfig(
            fit_start_mode=args.fit_start_mode,
            manual_fit_starts=args.manual_fit_starts,
            n_final_frames_for_r=args.n_final_frames_for_r,
            min_fit_r2_warning=args.min_fit_r2_warning,
            fail_max_ar_at_last_frame=args.fail_max_ar_at_last_frame,
            trim_tail_to_floor=args.trim_tail_to_floor,
            tail_floor_fraction=args.tail_floor_fraction,
            float_baseline=args.float_baseline,
        ),
        visualization=VisualizationConfig(
            make_videos=args.make_videos,
            video_fps=args.video_fps,
        ),
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        result = run_pipeline(config, show_progress=args.progress)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
