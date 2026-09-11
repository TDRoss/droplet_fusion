"""Configuration objects for the droplet fusion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


FitStartMode = Literal["max_AR", "manual", "first_valid", "decay_onset"]
ThresholdMethod = Literal["otsu", "yen", "li", "triangle"]
SegmentationMode = Literal["per_frame", "temporal"]
OuterEnvelopeMode = Literal["none", "convex_hull"]
GrowthFloorMode = Literal["image_threshold", "frame_percentile"]


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters for future per-frame droplet segmentation."""

    min_object_area_px: int = 100
    max_object_area_px: int | None = None
    background_poly_order: int = 2
    background_mask_dilation_px: int = 10
    threshold_method: ThresholdMethod = "otsu"
    snr_threshold: float = 3.0
    segmentation_mode: SegmentationMode = "temporal"
    min_area_ratio_to_reference: float = 0.35
    max_centroid_jump_px: float | None = None
    mask_closing_radius_px: int = 2
    boundary_threshold_percentile: float = 60.0
    outer_envelope_mode: OuterEnvelopeMode = "none"
    growth_floor_mode: GrowthFloorMode = "image_threshold"
    write_segmentation_qc: bool = True
    large_mask_area_fraction: float = 0.85
    percentile_threshold_fallback: bool = True

    def validate_fragment(self) -> None:
        """Validate segmentation settings outside a full PipelineConfig."""

        if self.min_object_area_px < 1:
            raise ValueError("min_object_area_px must be at least 1")
        if self.max_object_area_px is not None and self.max_object_area_px < self.min_object_area_px:
            raise ValueError("max_object_area_px must be at least min_object_area_px")
        if self.background_poly_order < 0:
            raise ValueError("background_poly_order must be non-negative")
        if self.background_mask_dilation_px < 0:
            raise ValueError("background_mask_dilation_px must be non-negative")
        if self.snr_threshold <= 0:
            raise ValueError("snr_threshold must be greater than zero")
        if self.segmentation_mode not in {"per_frame", "temporal"}:
            raise ValueError("segmentation_mode must be 'per_frame' or 'temporal'")
        if not 0.0 < self.min_area_ratio_to_reference <= 1.0:
            raise ValueError("min_area_ratio_to_reference must be in (0, 1]")
        if self.max_centroid_jump_px is not None and self.max_centroid_jump_px <= 0:
            raise ValueError("max_centroid_jump_px must be greater than zero")
        if self.mask_closing_radius_px < 0:
            raise ValueError("mask_closing_radius_px must be non-negative")
        if not 0.0 <= self.boundary_threshold_percentile <= 100.0:
            raise ValueError("boundary_threshold_percentile must be between 0 and 100")
        if self.outer_envelope_mode not in {"none", "convex_hull"}:
            raise ValueError("outer_envelope_mode must be 'none' or 'convex_hull'")
        if self.growth_floor_mode not in {"image_threshold", "frame_percentile"}:
            raise ValueError("growth_floor_mode must be 'image_threshold' or 'frame_percentile'")
        if not 0.0 < self.large_mask_area_fraction <= 1.0:
            raise ValueError("large_mask_area_fraction must be in (0, 1]")


@dataclass(frozen=True)
class FitConfig:
    """Parameters for future post-fusion AR relaxation fitting."""

    fit_start_mode: FitStartMode = "first_valid"
    manual_fit_starts: Path | None = None
    n_final_frames_for_r: int = 5
    min_fit_r2_warning: float = 0.8
    fail_max_ar_at_last_frame: bool = True
    trim_tail_to_floor: bool = True
    tail_floor_fraction: float = 0.1
    float_baseline: bool = False


@dataclass(frozen=True)
class VisualizationConfig:
    """Parameters for future QC plots and overlay videos."""

    make_videos: bool = True
    video_fps: float = 10.0


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration parsed from the CLI."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    seconds_per_frame: float = 1.0
    um_per_pixel: float = 1.0
    overwrite: bool = False
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    fitting: FitConfig = field(default_factory=FitConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def validate(self) -> None:
        """Validate user-facing configuration values."""

        if self.seconds_per_frame <= 0:
            raise ValueError("seconds_per_frame must be greater than zero")
        if self.um_per_pixel <= 0:
            raise ValueError("um_per_pixel must be greater than zero")
        self.segmentation.validate_fragment()
        if self.fitting.fit_start_mode == "manual" and self.fitting.manual_fit_starts is None:
            raise ValueError("manual_fit_starts is required when fit_start_mode is manual")
        if self.fitting.n_final_frames_for_r < 1:
            raise ValueError("n_final_frames_for_r must be at least 1")
        if not 0.0 <= self.fitting.min_fit_r2_warning <= 1.0:
            raise ValueError("min_fit_r2_warning must be between 0 and 1")
        if not 0.0 < self.fitting.tail_floor_fraction < 1.0:
            raise ValueError("tail_floor_fraction must be in (0, 1)")
        if self.visualization.video_fps <= 0:
            raise ValueError("video_fps must be greater than zero")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return _stringify_paths(asdict(self))


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    return value
