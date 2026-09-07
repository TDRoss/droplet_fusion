"""Per-frame and temporally aware droplet segmentation utilities."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from droplet_fusion.config import SegmentationConfig
from droplet_fusion.io import write_tiff_stack


SEGMENTATION_QC_COLUMNS = [
    "frame",
    "valid",
    "failure_reason",
    "background_method",
    "candidate_count",
    "selected_source",
    "selected_threshold",
    "score",
    "area_px",
    "centroid_x_px",
    "centroid_y_px",
    "reference_area_px",
    "area_ratio_to_reference",
    "centroid_jump_px",
    "mask_iou_with_reference",
    "notes",
]


@dataclass(frozen=True)
class SegmentationResult:
    """Segmentation outputs and QC metadata for one frame."""

    mask: np.ndarray
    corrected: np.ndarray
    background: np.ndarray
    valid: bool
    failure_reason: str | None
    background_method: str
    candidate_count: int
    selected_source: str | None = None
    selected_threshold: float = float("nan")
    score: float = float("nan")
    area_px: float = float("nan")
    centroid_x_px: float = float("nan")
    centroid_y_px: float = float("nan")
    reference_area_px: float = float("nan")
    area_ratio_to_reference: float = float("nan")
    centroid_jump_px: float = float("nan")
    mask_iou_with_reference: float = float("nan")
    notes: str = ""


@dataclass(frozen=True)
class StackSegmentationResult:
    """Segmentation outputs for a movie stack."""

    masks: np.ndarray
    frame_results: list[SegmentationResult]
    qc_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class _CandidateMask:
    mask: np.ndarray
    threshold: float
    source: str
    area: float
    centroid_y: float
    centroid_x: float
    solidity: float
    eccentricity: float
    extent: float
    border_distance: float


@dataclass(frozen=True)
class _PreparedFrame:
    corrected: np.ndarray
    background: np.ndarray
    background_method: str
    candidates: list[_CandidateMask]
    failure_reason: str | None


@dataclass(frozen=True)
class _TemporalContext:
    prior_mask: np.ndarray | None
    prior_centroid_y: float
    prior_centroid_x: float
    reference_area: float
    max_centroid_jump_px: float


def estimate_background_masked_poly(
    frame: np.ndarray,
    rough_mask: np.ndarray,
    order: int = 2,
    dilation_px: int = 10,
) -> np.ndarray:
    """Estimate smooth background from pixels outside ``rough_mask``.

    The fit tries the requested polynomial order first and backs off to lower
    orders when too few background pixels are available. If no stable fit can be
    made, a constant median background image is returned.
    """

    background, _method = _estimate_background_masked_poly_with_method(
        frame=frame,
        rough_mask=rough_mask,
        order=order,
        dilation_px=dilation_px,
    )
    return background


def segment_frame(frame: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
    """Segment exactly one non-border droplet object from a frame."""

    prepared = _prepare_frame(frame, config)
    if prepared.failure_reason is not None:
        return _invalid_result_from_prepared(prepared, prepared.failure_reason)

    selected, metrics, rejection_reason = _select_best_candidate(
        prepared.candidates,
        config=config,
        shape=prepared.corrected.shape,
        context=None,
    )
    if selected is None:
        return _invalid_result_from_prepared(prepared, rejection_reason or "no_object_after_threshold")
    if rejection_reason is not None:
        return _invalid_result_from_prepared(prepared, rejection_reason)

    return _result_from_candidate(prepared, selected, metrics)


def segment_stack(stack: np.ndarray, config: SegmentationConfig) -> StackSegmentationResult:
    """Segment all frames in a normalized ``(T, Y, X)`` stack."""

    arr = np.asarray(stack)
    if arr.ndim != 3:
        raise ValueError(f"segment_stack expects a 3D (T, Y, X) stack, got shape {arr.shape}")

    prepared_frames = [_prepare_frame(frame, config) for frame in arr]
    if config.segmentation_mode == "per_frame":
        frame_results = [_select_prepared_frame(prepared, config=config, context=None) for prepared in prepared_frames]
    else:
        frame_results = _select_temporal_stack(prepared_frames, config)

    masks = np.stack([result.mask for result in frame_results], axis=0)
    qc_rows = [_qc_row(frame_index, result) for frame_index, result in enumerate(frame_results)]
    return StackSegmentationResult(masks=masks, frame_results=frame_results, qc_rows=qc_rows)


def write_mask_stack(path: Path | str, masks: np.ndarray) -> Path:
    """Write binary masks as an 8-bit TIFF stack with values 0 and 255."""

    binary = np.asarray(masks, dtype=bool)
    return write_tiff_stack(path, binary.astype(np.uint8) * 255, dtype=np.uint8)


def write_segmentation_qc_csv(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    """Write per-frame segmentation QC diagnostics."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENTATION_QC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _prepare_frame(frame: np.ndarray, config: SegmentationConfig) -> _PreparedFrame:
    frame_float = np.asarray(frame, dtype=np.float64)
    if frame_float.ndim != 2:
        raise ValueError(f"segment_frame expects a 2D frame, got shape {frame_float.shape}")

    finite_frame = frame_float[np.isfinite(frame_float)]
    if finite_frame.size == 0 or float(np.ptp(finite_frame)) == 0.0:
        background = np.full(frame_float.shape, float(finite_frame[0]) if finite_frame.size else 0.0, dtype=np.float64)
        return _PreparedFrame(
            corrected=np.zeros(frame_float.shape, dtype=np.float64),
            background=background,
            background_method="constant_frame",
            candidates=[],
            failure_reason="no_foreground_signal",
        )

    rough_mask = _rough_foreground_mask(frame_float)
    background, background_method = _estimate_background_masked_poly_with_method(
        frame=frame_float,
        rough_mask=rough_mask,
        order=config.background_poly_order,
        dilation_px=config.background_mask_dilation_px,
    )
    corrected = np.clip(frame_float - background, a_min=0.0, a_max=None)

    if not np.any(np.isfinite(corrected)) or float(np.nanmax(corrected)) <= 0.0:
        return _PreparedFrame(
            corrected=corrected,
            background=background,
            background_method=background_method,
            candidates=[],
            failure_reason="no_foreground_signal",
        )

    candidates = _generate_mask_candidates(corrected, config)
    failure_reason = None if candidates else "no_object_after_threshold"
    return _PreparedFrame(
        corrected=corrected,
        background=background,
        background_method=background_method,
        candidates=candidates,
        failure_reason=failure_reason,
    )


def _select_prepared_frame(
    prepared: _PreparedFrame,
    *,
    config: SegmentationConfig,
    context: _TemporalContext | None,
) -> SegmentationResult:
    if prepared.failure_reason is not None:
        return _invalid_result_from_prepared(prepared, prepared.failure_reason)

    selected, metrics, rejection_reason = _select_best_candidate(
        prepared.candidates,
        config=config,
        shape=prepared.corrected.shape,
        context=context,
    )
    if selected is None:
        return _invalid_result_from_prepared(prepared, rejection_reason or "no_object_after_threshold")
    if rejection_reason is not None:
        return _invalid_result_from_prepared(prepared, rejection_reason, metrics=metrics)
    return _result_from_candidate(prepared, selected, metrics)


def _select_temporal_stack(prepared_frames: list[_PreparedFrame], config: SegmentationConfig) -> list[SegmentationResult]:
    if not prepared_frames:
        return []

    independent_results = [
        _select_prepared_frame(prepared, config=config, context=None)
        for prepared in prepared_frames
    ]
    valid_indices = [index for index, result in enumerate(independent_results) if result.valid]
    if not valid_indices:
        return independent_results

    anchor_index = max(valid_indices, key=lambda index: independent_results[index].area_px)
    frame_results: list[SegmentationResult | None] = [None] * len(prepared_frames)
    frame_results[anchor_index] = independent_results[anchor_index]
    accepted_areas = [float(independent_results[anchor_index].area_px)]

    for index in range(anchor_index + 1, len(prepared_frames)):
        prior = frame_results[index - 1]
        context = _context_from_result(prior, accepted_areas, prepared_frames[index].corrected.shape, config)
        result = _select_prepared_frame(prepared_frames[index], config=config, context=context)
        frame_results[index] = result
        if result.valid:
            accepted_areas.append(float(result.area_px))

    for index in range(anchor_index - 1, -1, -1):
        prior = frame_results[index + 1]
        context = _context_from_result(prior, accepted_areas, prepared_frames[index].corrected.shape, config)
        result = _select_prepared_frame(prepared_frames[index], config=config, context=context)
        frame_results[index] = result
        if result.valid:
            accepted_areas.append(float(result.area_px))

    return [result if result is not None else independent_results[index] for index, result in enumerate(frame_results)]


def _context_from_result(
    prior: SegmentationResult | None,
    accepted_areas: list[float],
    shape: tuple[int, int],
    config: SegmentationConfig,
) -> _TemporalContext | None:
    finite_areas = [area for area in accepted_areas if math.isfinite(area) and area > 0.0]
    if prior is None or not prior.valid or not finite_areas:
        return None

    max_jump = config.max_centroid_jump_px
    if max_jump is None:
        max_jump = max(8.0, 0.25 * float(np.hypot(shape[0], shape[1])))

    return _TemporalContext(
        prior_mask=np.asarray(prior.mask, dtype=bool),
        prior_centroid_y=float(prior.centroid_y_px),
        prior_centroid_x=float(prior.centroid_x_px),
        reference_area=float(np.median(finite_areas)),
        max_centroid_jump_px=float(max_jump),
    )


def _invalid_result_from_prepared(
    prepared: _PreparedFrame,
    failure_reason: str,
    *,
    metrics: dict[str, float] | None = None,
) -> SegmentationResult:
    metrics = metrics or {}
    return SegmentationResult(
        mask=np.zeros(prepared.corrected.shape, dtype=bool),
        corrected=prepared.corrected,
        background=prepared.background,
        valid=False,
        failure_reason=failure_reason,
        background_method=prepared.background_method,
        candidate_count=len(prepared.candidates),
        selected_source=None,
        selected_threshold=float("nan"),
        score=float(metrics.get("score", float("nan"))),
        area_px=float(metrics.get("area_px", float("nan"))),
        centroid_x_px=float(metrics.get("centroid_x_px", float("nan"))),
        centroid_y_px=float(metrics.get("centroid_y_px", float("nan"))),
        reference_area_px=float(metrics.get("reference_area_px", float("nan"))),
        area_ratio_to_reference=float(metrics.get("area_ratio_to_reference", float("nan"))),
        centroid_jump_px=float(metrics.get("centroid_jump_px", float("nan"))),
        mask_iou_with_reference=float(metrics.get("mask_iou_with_reference", float("nan"))),
        notes=metrics.get("notes", ""),
    )


def _result_from_candidate(
    prepared: _PreparedFrame,
    candidate: _CandidateMask,
    metrics: dict[str, float],
) -> SegmentationResult:
    return SegmentationResult(
        mask=candidate.mask,
        corrected=prepared.corrected,
        background=prepared.background,
        valid=True,
        failure_reason=None,
        background_method=prepared.background_method,
        candidate_count=len(prepared.candidates),
        selected_source=candidate.source,
        selected_threshold=candidate.threshold,
        score=float(metrics["score"]),
        area_px=candidate.area,
        centroid_x_px=candidate.centroid_x,
        centroid_y_px=candidate.centroid_y,
        reference_area_px=float(metrics.get("reference_area_px", float("nan"))),
        area_ratio_to_reference=float(metrics.get("area_ratio_to_reference", float("nan"))),
        centroid_jump_px=float(metrics.get("centroid_jump_px", float("nan"))),
        mask_iou_with_reference=float(metrics.get("mask_iou_with_reference", float("nan"))),
        notes=metrics.get("notes", ""),
    )


def _qc_row(frame_index: int, result: SegmentationResult) -> dict[str, Any]:
    return {
        "frame": frame_index,
        "valid": bool(result.valid),
        "failure_reason": result.failure_reason or "",
        "background_method": result.background_method,
        "candidate_count": result.candidate_count,
        "selected_source": result.selected_source or "",
        "selected_threshold": result.selected_threshold,
        "score": result.score,
        "area_px": result.area_px,
        "centroid_x_px": result.centroid_x_px,
        "centroid_y_px": result.centroid_y_px,
        "reference_area_px": result.reference_area_px,
        "area_ratio_to_reference": result.area_ratio_to_reference,
        "centroid_jump_px": result.centroid_jump_px,
        "mask_iou_with_reference": result.mask_iou_with_reference,
        "notes": result.notes,
    }


def _rough_foreground_mask(frame: np.ndarray) -> np.ndarray:
    smoothed = filters.gaussian(frame, sigma=1.0, preserve_range=True)
    finite = smoothed[np.isfinite(smoothed)]
    if finite.size == 0 or float(np.ptp(finite)) == 0.0:
        return np.zeros(frame.shape, dtype=bool)

    try:
        seed_threshold = filters.threshold_otsu(finite)
    except ValueError:
        seed_threshold = np.percentile(finite, 99.0)

    seed = smoothed > seed_threshold
    if not np.any(seed):
        seed_threshold = np.percentile(finite, 99.0)
        seed = smoothed > seed_threshold
    if not np.any(seed):
        return np.zeros(frame.shape, dtype=bool)

    grow_threshold = min(float(seed_threshold), float(np.percentile(finite, 70.0)))
    allowed = smoothed > grow_threshold
    grown = ndi.binary_propagation(seed, mask=allowed)
    if np.count_nonzero(grown) > 0.75 * grown.size:
        grown = seed

    return morphology.closing(grown, morphology.disk(2))


def _estimate_background_masked_poly_with_method(
    frame: np.ndarray,
    rough_mask: np.ndarray,
    order: int,
    dilation_px: int,
) -> tuple[np.ndarray, str]:
    frame_float = np.asarray(frame, dtype=np.float64)
    mask = np.asarray(rough_mask, dtype=bool)

    if dilation_px > 0 and np.any(mask):
        mask = morphology.dilation(mask, morphology.disk(dilation_px))

    background_pixels = ~mask & np.isfinite(frame_float)
    for fit_order in range(order, -1, -1):
        n_terms = _polynomial_term_count(fit_order)
        if int(np.count_nonzero(background_pixels)) < max(25, n_terms * 3):
            continue
        try:
            background = _fit_polynomial_background(frame_float, background_pixels, fit_order)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(background)):
            return background, f"poly_order_{fit_order}"

    finite = frame_float[np.isfinite(frame_float)]
    median = float(np.median(finite)) if finite.size else 0.0
    return np.full(frame_float.shape, median, dtype=np.float64), "median_fallback_too_few_background_pixels"


def _fit_polynomial_background(frame: np.ndarray, background_pixels: np.ndarray, order: int) -> np.ndarray:
    yy, xx = np.indices(frame.shape, dtype=np.float64)
    x_norm = _normalize_coordinate(xx)
    y_norm = _normalize_coordinate(yy)
    design = _polynomial_design_matrix(x_norm[background_pixels], y_norm[background_pixels], order)
    values = frame[background_pixels]

    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    full_design = _polynomial_design_matrix(x_norm.ravel(), y_norm.ravel(), order)
    return (full_design @ coefficients).reshape(frame.shape)


def _generate_mask_candidates(corrected: np.ndarray, config: SegmentationConfig) -> list[_CandidateMask]:
    # Primary tier: intensity-based Otsu/SNR thresholds plus flood-fill growth from
    # bright seeds. Otsu/SNR cleanly separate a well-isolated droplet from the
    # corrected background, while growth expands from a bright core out to a dim
    # halo without slicing through the droplet. Raw percentile thresholds computed
    # over the whole frame land inside droplets that fill a large fraction of the
    # crop (cutting off dimmer lobes), so they are only used as a fallback for
    # frames the primary tier cannot resolve (e.g. a threshold that merges the
    # droplet with a border-touching neighbor and is then cleared).
    primary: list[_CandidateMask] = []
    for threshold, source in _primary_threshold_candidates(corrected, config):
        primary.extend(_candidates_from_binary(corrected > threshold, threshold, source, config))
    primary.extend(_growth_candidates(corrected, config))
    primary = _dedupe_candidates(primary)
    if primary or not config.percentile_threshold_fallback:
        return primary

    fallback: list[_CandidateMask] = []
    for threshold, source in _percentile_threshold_candidates(corrected, config):
        fallback.extend(_candidates_from_binary(corrected > threshold, threshold, source, config))
    return _dedupe_candidates(fallback)


def _dedupe_candidates(candidates: list[_CandidateMask]) -> list[_CandidateMask]:
    unique_candidates: list[_CandidateMask] = []
    for candidate in candidates:
        if not any(_mask_iou(candidate.mask, existing.mask) > 0.98 for existing in unique_candidates):
            unique_candidates.append(candidate)
    return unique_candidates


def _threshold_corrected_frame(corrected: np.ndarray, config: SegmentationConfig) -> float:
    finite = corrected[np.isfinite(corrected)]
    if finite.size == 0:
        return np.inf

    image_threshold = _image_threshold(finite, config.threshold_method)

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad
    snr_threshold = median + config.snr_threshold * robust_sigma

    if robust_sigma == 0.0:
        return image_threshold
    return max(image_threshold, snr_threshold)


def _primary_threshold_candidates(corrected: np.ndarray, config: SegmentationConfig) -> list[tuple[float, str]]:
    finite = corrected[np.isfinite(corrected)]
    if finite.size == 0:
        return [(np.inf, "invalid")]

    candidates: list[tuple[float, str]] = [
        (_threshold_corrected_frame(corrected, config), f"{config.threshold_method}_snr"),
        (_image_threshold(finite, config.threshold_method), config.threshold_method),
    ]
    return _dedupe_thresholds(candidates)


def _percentile_threshold_candidates(corrected: np.ndarray, config: SegmentationConfig) -> list[tuple[float, str]]:
    finite = corrected[np.isfinite(corrected)]
    if finite.size == 0:
        return [(np.inf, "invalid")]

    percentiles = [
        config.boundary_threshold_percentile,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        92,
        94,
        95,
        96,
        97,
        98,
        99,
    ]
    candidates: list[tuple[float, str]] = []
    for percentile in percentiles:
        clipped_percentile = min(100.0, max(0.0, float(percentile)))
        candidates.append((float(np.percentile(finite, clipped_percentile)), f"p{clipped_percentile:g}"))
    return _dedupe_thresholds(candidates)


def _dedupe_thresholds(candidates: list[tuple[float, str]]) -> list[tuple[float, str]]:
    unique_candidates: list[tuple[float, str]] = []
    for threshold, source in candidates:
        if not np.isfinite(threshold):
            continue
        if not any(np.isclose(threshold, existing_threshold) for existing_threshold, _ in unique_candidates):
            unique_candidates.append((float(threshold), source))
    return unique_candidates or [(np.inf, "invalid")]


def _image_threshold(finite_values: np.ndarray, method: str) -> float:
    try:
        if method == "otsu":
            return float(filters.threshold_otsu(finite_values))
        if method == "yen":
            return float(filters.threshold_yen(finite_values))
        if method == "li":
            return float(filters.threshold_li(finite_values))
        if method == "triangle":
            return float(filters.threshold_triangle(finite_values))
        raise ValueError(f"unsupported threshold method: {method}")
    except ValueError:
        return float(np.percentile(finite_values, 99.0))


def _growth_candidates(corrected: np.ndarray, config: SegmentationConfig) -> list[_CandidateMask]:
    finite = corrected[np.isfinite(corrected)]
    if finite.size == 0 or float(np.ptp(finite)) == 0.0:
        return []

    # Flood-fill grows a mask from a bright seed down to a low "floor" threshold,
    # so that a droplet whose boundary is dimmer than its core is not sliced off
    # at the core. Which seeds and floor are appropriate depends on how the
    # droplet's intensity profile meets the background; see growth_floor_mode.
    image_threshold = _image_threshold(finite, config.threshold_method)
    low_fractions = (0.3, 0.5, 0.7)

    if config.growth_floor_mode == "image_threshold":
        # Sharp-edged droplets (a flat plateau dropping steeply into background,
        # the usual epi-fluorescence case). Two things must hold.
        #
        # Seeds must sit inside the droplet, so they are percentiles of the
        # foreground rather than of the whole frame. A frame percentile is only a
        # droplet level when the droplet fills much of the crop; for a pair
        # cropped to <10% of the frame, p90/p95 are still background and would
        # seed the flood from noise.
        #
        # The floor is then clamped at the image threshold, which is where the
        # droplet actually ends. Growing below it can only add background: the
        # flood propagates out through the dim optical halo and noise until it
        # happens to stall, producing a ragged envelope well outside the droplet
        # whose excess varies frame to frame. That corrupts the aspect ratio far
        # more than it corrupts the area, since the excess is also asymmetric.
        foreground = finite[finite > image_threshold] if np.isfinite(image_threshold) else finite[:0]
        high_thresholds = [(image_threshold, config.threshold_method)]
        if foreground.size:
            high_thresholds.extend(
                [
                    (float(np.percentile(foreground, 50.0)), "fg50"),
                    (float(np.percentile(foreground, 75.0)), "fg75"),
                    (float(np.percentile(foreground, 90.0)), "fg90"),
                ]
            )
        background_floor = image_threshold
    else:
        # Droplets with a genuinely dim outer region -- a broad faint rim around a
        # much brighter core -- where the image threshold splits the droplet
        # itself and clamping the floor there would discard the rim. Seeds and
        # floor are both taken from the frame, so the flood can reach down to the
        # rim level.
        high_thresholds = [
            (image_threshold, config.threshold_method),
            (float(np.percentile(finite, 90.0)), "p90"),
            (float(np.percentile(finite, 95.0)), "p95"),
        ]
        background_floor = float("nan")

    candidates: list[_CandidateMask] = []
    for high_threshold, high_source in high_thresholds:
        if not np.isfinite(high_threshold) or high_threshold <= 0.0:
            continue
        seed = corrected > high_threshold
        seed = _remove_small_objects(seed, max(3, config.min_object_area_px // 10))
        if not np.any(seed):
            continue
        for fraction in low_fractions:
            low_threshold = float(high_threshold * fraction)
            if np.isfinite(background_floor):
                low_threshold = max(low_threshold, float(background_floor))
            allowed = corrected > low_threshold
            if not np.any(allowed) or low_threshold >= high_threshold:
                continue
            grown = ndi.binary_propagation(seed, mask=allowed)
            source = f"grow_{fraction:g}x_from_{high_source}"
            candidates.extend(_candidates_from_binary(grown, low_threshold, source, config))
    return candidates


def _candidates_from_binary(
    binary: np.ndarray,
    threshold: float,
    source: str,
    config: SegmentationConfig,
) -> list[_CandidateMask]:
    mask = _postprocess_binary_mask(binary, config)
    if not np.any(mask):
        return []

    labeled = measure.label(mask)
    candidates = []
    for region in measure.regionprops(labeled):
        if not _area_within_bounds(region.area, config.max_object_area_px):
            continue
        region_mask = labeled == region.label
        envelope_mask = _apply_outer_envelope(region_mask, config)
        if not np.any(envelope_mask):
            continue
        envelope_area = float(np.count_nonzero(envelope_mask))
        if not _area_within_bounds(envelope_area, config.max_object_area_px):
            continue
        envelope_region = measure.regionprops(measure.label(envelope_mask))[0]
        envelope_source = _source_with_outer_envelope(source, config)
        candidates.append(_candidate_from_region(envelope_mask, envelope_region, threshold, envelope_source))
    return candidates


def _postprocess_binary_mask(binary: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    mask = np.asarray(binary, dtype=bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)

    mask = _remove_small_objects(mask, config.min_object_area_px)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)

    mask = ndi.binary_fill_holes(mask)
    if config.mask_closing_radius_px > 0:
        mask = morphology.closing(mask, morphology.disk(config.mask_closing_radius_px))
        mask = ndi.binary_fill_holes(mask)
    mask = segmentation.clear_border(mask)
    mask = _remove_small_objects(mask, config.min_object_area_px)
    return mask


def _candidate_from_region(
    mask: np.ndarray,
    region,
    threshold: float,
    source: str,
) -> _CandidateMask:
    min_row, min_col, max_row, max_col = region.bbox
    border_distance = float(
        min(
            min_row,
            min_col,
            mask.shape[0] - max_row,
            mask.shape[1] - max_col,
        )
    )
    centroid_y, centroid_x = (float(value) for value in region.centroid)
    return _CandidateMask(
        mask=np.asarray(mask, dtype=bool),
        threshold=float(threshold),
        source=source,
        area=float(region.area),
        centroid_y=centroid_y,
        centroid_x=centroid_x,
        solidity=float(region.solidity) if math.isfinite(float(region.solidity)) else 0.0,
        eccentricity=float(region.eccentricity) if math.isfinite(float(region.eccentricity)) else 0.0,
        extent=float(region.extent) if math.isfinite(float(region.extent)) else 0.0,
        border_distance=border_distance,
    )


def _apply_outer_envelope(mask: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    envelope_mode = config.outer_envelope_mode
    if envelope_mode == "none":
        return np.asarray(mask, dtype=bool)
    if envelope_mode == "convex_hull":
        envelope = morphology.convex_hull_image(np.asarray(mask, dtype=bool))
        envelope = ndi.binary_fill_holes(envelope)
        return segmentation.clear_border(envelope)
    raise ValueError(f"unsupported outer envelope mode: {envelope_mode}")


def _source_with_outer_envelope(source: str, config: SegmentationConfig) -> str:
    if config.outer_envelope_mode == "none":
        return source
    return f"{source}+{config.outer_envelope_mode}"


def _select_best_candidate(
    candidates: list[_CandidateMask],
    *,
    config: SegmentationConfig,
    shape: tuple[int, int],
    context: _TemporalContext | None,
) -> tuple[_CandidateMask | None, dict[str, float], str | None]:
    if not candidates:
        return None, {}, "no_object_after_threshold"

    scored = [(_score_candidate(candidate, shape=shape, context=context, config=config), candidate) for candidate in candidates]
    metrics, selected = max(scored, key=lambda item: item[0]["score"])
    rejection_reason = _candidate_rejection_reason(selected, metrics, config=config, context=context)
    return selected, metrics, rejection_reason


def _score_candidate(
    candidate: _CandidateMask,
    *,
    shape: tuple[int, int],
    context: _TemporalContext | None,
    config: SegmentationConfig,
) -> dict[str, float]:
    image_area = float(shape[0] * shape[1])
    center_y = (shape[0] - 1) / 2.0
    center_x = (shape[1] - 1) / 2.0
    diagonal = float(np.hypot(shape[0], shape[1]))
    center_distance = float(np.hypot(candidate.centroid_y - center_y, candidate.centroid_x - center_x))

    area_score = math.log1p(candidate.area) / max(1.0, math.log1p(image_area * 0.6))
    area_fraction = candidate.area / max(image_area, 1.0)
    large_mask_penalty = max(0.0, area_fraction - config.large_mask_area_fraction) * 12.0
    near_border_penalty = max(0.0, 3.0 - candidate.border_distance) * 0.15
    score = (
        3.0 * area_score
        + 0.9 * candidate.solidity
        + 0.3 * candidate.extent
        - 0.8 * center_distance / max(diagonal, 1.0)
        - 0.15 * candidate.eccentricity
        - large_mask_penalty
        - near_border_penalty
    )

    reference_area = float("nan")
    area_ratio = float("nan")
    centroid_jump = float("nan")
    mask_iou = float("nan")
    notes = ""

    if context is not None and context.reference_area > 0.0:
        reference_area = context.reference_area
        area_ratio = min(candidate.area / reference_area, reference_area / candidate.area)
        centroid_jump = float(np.hypot(candidate.centroid_y - context.prior_centroid_y, candidate.centroid_x - context.prior_centroid_x))
        mask_iou = _mask_iou(candidate.mask, context.prior_mask)
        jump_ratio = centroid_jump / max(context.max_centroid_jump_px, 1.0)
        score += 2.0 * mask_iou + 1.8 * area_ratio - 1.4 * jump_ratio
        if area_ratio < 0.5:
            notes = "low_area_ratio"
        if jump_ratio > 1.0 and mask_iou < 0.05:
            notes = ";".join(filter(None, [notes, "large_centroid_jump"]))

    return {
        "score": float(score),
        "area_px": candidate.area,
        "centroid_x_px": candidate.centroid_x,
        "centroid_y_px": candidate.centroid_y,
        "reference_area_px": reference_area,
        "area_ratio_to_reference": area_ratio,
        "centroid_jump_px": centroid_jump,
        "mask_iou_with_reference": mask_iou,
        "notes": notes,
    }


def _candidate_rejection_reason(
    candidate: _CandidateMask,
    metrics: dict[str, float],
    *,
    config: SegmentationConfig,
    context: _TemporalContext | None,
) -> str | None:
    if context is None:
        return None

    area_ratio = metrics.get("area_ratio_to_reference", float("nan"))
    centroid_jump = metrics.get("centroid_jump_px", float("nan"))
    mask_iou = metrics.get("mask_iou_with_reference", float("nan"))

    if math.isfinite(area_ratio) and area_ratio < config.min_area_ratio_to_reference:
        return "no_temporally_consistent_object"
    if (
        math.isfinite(centroid_jump)
        and centroid_jump > context.max_centroid_jump_px
        and (not math.isfinite(mask_iou) or mask_iou < 0.05)
    ):
        return "no_temporally_consistent_object"
    return None


def _mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
    if mask_a is None or mask_b is None:
        return float("nan")
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        return float("nan")
    union = np.count_nonzero(a | b)
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(a & b) / union)


def _area_within_bounds(area: float, max_object_area_px: int | None) -> bool:
    return max_object_area_px is None or area <= max_object_area_px


def _remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    labeled = measure.label(mask)
    if labeled.max() == 0:
        return np.zeros(mask.shape, dtype=bool)

    counts = np.bincount(labeled.ravel())
    keep_labels = np.flatnonzero(counts >= min_area)
    keep_labels = keep_labels[keep_labels != 0]
    return np.isin(labeled, keep_labels)


def _normalize_coordinate(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    if max_value == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return 2.0 * (values / max_value) - 1.0


def _polynomial_design_matrix(x_values: np.ndarray, y_values: np.ndarray, order: int) -> np.ndarray:
    terms = []
    for total_degree in range(order + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            terms.append((x_values**x_power) * (y_values**y_power))
    return np.column_stack(terms)


def _polynomial_term_count(order: int) -> int:
    return (order + 1) * (order + 2) // 2
