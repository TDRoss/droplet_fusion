"""Ellipse fitting utilities for segmented droplet masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure


@dataclass(frozen=True)
class EllipseResult:
    """Result of fitting an ellipse to a binary mask boundary.

    Axis lengths are full lengths in pixels, matching skimage regionprops
    conventions. They are not semi-axis radii.
    """

    valid: bool
    failure_reason: str | None
    center_x_px: float
    center_y_px: float
    major_axis_px: float
    minor_axis_px: float
    angle_rad: float
    aspect_ratio: float


def fit_ellipse_to_mask_boundary(mask: np.ndarray) -> EllipseResult:
    """Fit an ellipse to the boundary of a single binary mask."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"fit_ellipse_to_mask_boundary expects a 2D mask, got shape {binary.shape}")

    if int(np.count_nonzero(binary)) < 5:
        return _invalid("too_few_foreground_pixels")

    contours = measure.find_contours(binary.astype(np.float32), level=0.5)
    if not contours:
        return _fit_regionprops_fallback(binary, "no_boundary_contour")

    contour = max(contours, key=len)
    if len(contour) < 5:
        return _fit_regionprops_fallback(binary, "too_few_boundary_points")

    # EllipseModel expects coordinates as x, y. find_contours returns row, col.
    points = np.column_stack((contour[:, 1], contour[:, 0]))
    try:
        model = measure.EllipseModel.from_estimate(points)
        params = model.params
    except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError):
        return _fit_regionprops_fallback(binary, "ellipse_model_estimate_failed")

    if params is None:
        return _fit_regionprops_fallback(binary, "ellipse_model_estimate_failed")

    center_x, center_y, axis_a, axis_b, angle = (float(value) for value in params)
    if not np.all(np.isfinite([center_x, center_y, axis_a, axis_b, angle])):
        return _fit_regionprops_fallback(binary, "ellipse_model_nonfinite")

    if axis_a <= 0.0 or axis_b <= 0.0:
        return _fit_regionprops_fallback(binary, "ellipse_axis_nonpositive")

    major_axis = 2.0 * max(axis_a, axis_b)
    minor_axis = 2.0 * min(axis_a, axis_b)
    if axis_b > axis_a:
        angle += np.pi / 2.0

    return _validate_result(
        binary=binary,
        center_x=center_x,
        center_y=center_y,
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle=angle,
        fallback_reason="ellipse_center_not_near_mask",
    )


def _fit_regionprops_fallback(mask: np.ndarray, original_failure: str) -> EllipseResult:
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return _invalid(original_failure)

    region = max(regions, key=lambda item: item.area)
    if region.area < 5:
        return _invalid("too_few_foreground_pixels")

    major_axis = _region_axis_length(region, new_name="axis_major_length", old_name="major_axis_length")
    minor_axis = _region_axis_length(region, new_name="axis_minor_length", old_name="minor_axis_length")
    if major_axis <= 0.0 or minor_axis <= 0.0:
        return _invalid(original_failure)

    center_y, center_x = (float(value) for value in region.centroid)
    # Convert skimage's row-axis orientation into the x/y convention used here.
    angle = float(np.pi / 2.0 - region.orientation)
    return _validate_result(
        binary=mask,
        center_x=center_x,
        center_y=center_y,
        major_axis=major_axis,
        minor_axis=minor_axis,
        angle=angle,
        fallback_reason=original_failure,
    )


def _validate_result(
    *,
    binary: np.ndarray,
    center_x: float,
    center_y: float,
    major_axis: float,
    minor_axis: float,
    angle: float,
    fallback_reason: str,
) -> EllipseResult:
    if minor_axis <= 0.0:
        return _invalid("minor_axis_zero")
    if major_axis < minor_axis:
        major_axis, minor_axis = minor_axis, major_axis
        angle += np.pi / 2.0

    aspect_ratio = major_axis / minor_axis
    if not np.isfinite(aspect_ratio) or aspect_ratio < 1.0:
        return _invalid("invalid_aspect_ratio")

    if not _center_is_inside_or_near_mask(binary, center_x=center_x, center_y=center_y):
        return _invalid(fallback_reason)

    return EllipseResult(
        valid=True,
        failure_reason=None,
        center_x_px=center_x,
        center_y_px=center_y,
        major_axis_px=float(major_axis),
        minor_axis_px=float(minor_axis),
        angle_rad=float(np.mod(angle, np.pi)),
        aspect_ratio=float(aspect_ratio),
    )


def _center_is_inside_or_near_mask(binary: np.ndarray, *, center_x: float, center_y: float) -> bool:
    if not np.all(np.isfinite([center_x, center_y])):
        return False

    height, width = binary.shape
    if center_x < -2.0 or center_y < -2.0 or center_x > width + 1.0 or center_y > height + 1.0:
        return False

    rounded_x = int(round(center_x))
    rounded_y = int(round(center_y))
    if 0 <= rounded_y < height and 0 <= rounded_x < width and binary[rounded_y, rounded_x]:
        return True

    yy, xx = np.nonzero(binary)
    distances = np.hypot(xx.astype(np.float64) - center_x, yy.astype(np.float64) - center_y)
    return bool(distances.size and float(np.min(distances)) <= 2.0)


def _region_axis_length(region, *, new_name: str, old_name: str) -> float:
    if hasattr(region, new_name):
        return float(getattr(region, new_name))
    return float(getattr(region, old_name))


def _invalid(reason: str) -> EllipseResult:
    nan = float("nan")
    return EllipseResult(
        valid=False,
        failure_reason=reason,
        center_x_px=nan,
        center_y_px=nan,
        major_axis_px=nan,
        minor_axis_px=nan,
        angle_rad=nan,
        aspect_ratio=nan,
    )
