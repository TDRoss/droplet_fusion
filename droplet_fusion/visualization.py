"""QC plot and overlay-video generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Ellipse
from skimage import measure

from droplet_fusion.fitting import ar_model_baseline


def write_ar_fit_plot(
    path: Path | str,
    rows: list[dict[str, Any]],
    movie_result: dict[str, Any],
) -> Path:
    """Write a per-movie AR(t) QC plot."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fit_start_frame = _optional_int(movie_result.get("fit_start_frame"))
    fit_start_time = _fit_start_time(rows, fit_start_frame)
    times = np.array([_float_or_nan(row.get("time_s")) - fit_start_time for row in rows], dtype=np.float64)
    ars = np.array([_float_or_nan(row.get("AR")) for row in rows], dtype=np.float64)
    used = np.array([_truthy(row.get("used_for_fit")) for row in rows], dtype=bool)
    finite = np.isfinite(times) & np.isfinite(ars)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    if np.any(finite & ~used):
        ax.scatter(times[finite & ~used], ars[finite & ~used], s=26, color="#4c78a8", label="Measured")
    if np.any(finite & used):
        ax.scatter(times[finite & used], ars[finite & used], s=34, color="#f58518", label="Used for fit", zorder=3)

    A = _float_or_nan(movie_result.get("A"))
    tau = _float_or_nan(movie_result.get("tau_fusion_s"))
    baseline = _float_or_nan(movie_result.get("baseline"))
    if not np.isfinite(baseline):
        baseline = 1.0
    if np.isfinite(A) and np.isfinite(tau) and tau > 0.0 and np.any(finite & used):
        max_time = max(float(np.nanmax(times[finite & used])), 0.0)
        curve_time = np.linspace(0.0, max_time, 200)
        ax.plot(
            curve_time,
            ar_model_baseline(curve_time, A, tau, baseline),
            color="#1f1f1f",
            linewidth=2.0,
            label="Fit",
        )

    ax.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--")
    ax.set_title(str(movie_result.get("filename", "movie")))
    ax.set_xlabel("Time since fit start, s")
    ax.set_ylabel("AR")
    ax.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.7)

    if np.any(finite):
        ymin = min(0.98, float(np.nanmin(ars[finite])) - 0.05)
        ymax = max(1.05, float(np.nanmax(ars[finite])) + 0.05)
        ax.set_ylim(ymin, ymax)

    annotation = "\n".join(
        [
            f"status: {movie_result.get('status', '')}",
            f"A: {_format_float(A, precision=4)}",
            f"baseline: {_format_float(baseline, precision=4)}",
            f"tau_s: {_format_float(tau, precision=4)}",
            f"R_um: {_format_float(movie_result.get('R_um'), precision=4)}",
            "tau/R s/um: "
            f"{_format_float(movie_result.get('inverse_capillary_velocity_s_per_um'), precision=4)}",
            f"R2: {_format_float(movie_result.get('fit_r2'), precision=4)}",
            f"RMSE: {_format_float(movie_result.get('fit_rmse'), precision=4)}",
        ]
    )
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    ax.legend(loc="best", fontsize=8.5)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_inverse_capillary_velocity_summary_plot(
    path: Path | str,
    summary_rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, float | int]]:
    """Write the dataset-level inverse-capillary-velocity summary plot."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    values = []
    labels = []
    failed_labels = []
    for row in summary_rows:
        value = _float_or_nan(row.get("inverse_capillary_velocity_s_per_um"))
        filename = str(row.get("filename", "movie"))
        if np.isfinite(value) and value > 0.0:
            labels.append(filename)
            values.append(float(value))
        else:
            failed_labels.append(filename)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    if values:
        x = np.arange(len(values))
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ax.scatter(x, values, s=42, color="#4c78a8", zorder=3)
        ax.axhline(mean, color="#1f1f1f", linewidth=1.7, label="Mean")
        ax.fill_between(
            [-0.45, len(values) - 0.55],
            [mean - std, mean - std],
            [mean + std, mean + std],
            color="#72b7b2",
            alpha=0.22,
            label="Mean +/- SD",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend(loc="best", fontsize=8.5)
    else:
        mean = float("nan")
        std = float("nan")
        ax.text(0.5, 0.5, "No finite successful values", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])

    ax.set_title("Inverse capillary velocity summary")
    ax.set_ylabel("eta/gamma approx tau_fusion/R (s/um)")
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.7)
    subtitle = f"n={len(values)}, mean={_format_float(mean)}, sd={_format_float(std)}"
    if failed_labels:
        subtitle += f", omitted={len(failed_labels)}"
    ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=9.0)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path, {
        "n_successful": len(values),
        "n_failed_or_nonfinite": len(failed_labels),
        "mean_inverse_capillary_velocity_s_per_um": mean,
        "std_inverse_capillary_velocity_s_per_um": std,
        "mean_inverse_capillary_velocity_s_per_m": mean * 1e6 if math.isfinite(mean) else float("nan"),
        "std_inverse_capillary_velocity_s_per_m": std * 1e6 if math.isfinite(std) else float("nan"),
    }


def _bootstrap_slope_through_origin(
    radii: np.ndarray,
    taus: np.ndarray,
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap the through-origin slope by resampling movies with replacement.

    The per-movie ``tau_fusion_std_s`` values understate the true parameter
    uncertainty, because the AR residuals within a movie are serially correlated
    and so carry less independent information than the frame count suggests.
    Resampling whole movies sidesteps that: the spread of the resampled slopes
    reflects the movie-to-movie scatter actually observed, which is what the
    dataset-level slope is really estimated from.

    Returns ``(std, ci95_low, ci95_high)``, all NaN when there are too few movies.
    """

    n_movies = int(radii.size)
    if n_movies < 3:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    index = rng.integers(0, n_movies, size=(n_resamples, n_movies))
    resampled_radii = radii[index]
    resampled_taus = taus[index]
    denominator = np.sum(resampled_radii**2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slopes = np.sum(resampled_radii * resampled_taus, axis=1) / denominator
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return float("nan"), float("nan"), float("nan")

    low, high = (float(value) for value in np.percentile(slopes, [2.5, 97.5]))
    return float(np.std(slopes, ddof=1)), low, high


def write_tau_vs_radius_plot(
    path: Path | str,
    summary_rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, float | int]]:
    """Write a dataset-level scatter plot of tau_fusion against final droplet radius."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    radii = []
    taus = []
    tau_errors = []
    radius_errors = []
    omitted_labels = []
    for row in summary_rows:
        radius = _float_or_nan(row.get("R_um"))
        tau = _float_or_nan(row.get("tau_fusion_s"))
        filename = str(row.get("filename", "movie"))
        if np.isfinite(radius) and radius > 0.0 and np.isfinite(tau) and tau > 0.0:
            radii.append(float(radius))
            taus.append(float(tau))
            tau_std = _float_or_nan(row.get("tau_fusion_std_s"))
            tau_errors.append(float(tau_std) if np.isfinite(tau_std) and tau_std >= 0.0 else 0.0)
            radius_std = _float_or_nan(row.get("R_um_std"))
            radius_errors.append(float(radius_std) if np.isfinite(radius_std) and radius_std >= 0.0 else 0.0)
        else:
            omitted_labels.append(filename)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    if radii:
        radius_array = np.asarray(radii, dtype=np.float64)
        tau_array = np.asarray(taus, dtype=np.float64)
        error_array = np.asarray(tau_errors, dtype=np.float64)
        radius_error_array = np.asarray(radius_errors, dtype=np.float64)
        slope = float(np.sum(radius_array * tau_array) / np.sum(radius_array**2))
        slope_std, slope_low, slope_high = _bootstrap_slope_through_origin(radius_array, tau_array)
        if np.any(error_array > 0.0) or np.any(radius_error_array > 0.0):
            ax.errorbar(
                radius_array,
                tau_array,
                yerr=error_array if np.any(error_array > 0.0) else None,
                xerr=radius_error_array if np.any(radius_error_array > 0.0) else None,
                fmt="none",
                ecolor="#4c78a8",
                elinewidth=1.0,
                capsize=3.0,
                alpha=0.8,
                zorder=2,
            )
        ax.scatter(radius_array, tau_array, s=42, color="#4c78a8", zorder=3)
        line_radius = np.linspace(0.0, float(np.max(radius_array)) * 1.05, 100)
        ax.plot(
            line_radius,
            slope * line_radius,
            color="#1f1f1f",
            linewidth=1.7,
            label=(
                f"tau = {_format_float(slope)} s/um * R"
                if not np.isfinite(slope_std)
                else f"tau = ({_format_float(slope)} +/- {_format_float(slope_std)}) s/um * R"
            ),
        )
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="best", fontsize=8.5)
        correlation = (
            float(np.corrcoef(radius_array, tau_array)[0, 1]) if len(radii) > 1 else float("nan")
        )
    else:
        slope = float("nan")
        correlation = float("nan")
        slope_std = float("nan")
        slope_low = float("nan")
        slope_high = float("nan")
        ax.text(0.5, 0.5, "No finite successful values", transform=ax.transAxes, ha="center", va="center")

    ax.set_title("Fusion time vs final droplet radius")
    ax.set_xlabel("Final droplet radius R (um)")
    ax.set_ylabel("tau_fusion (s)")
    ax.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.7)
    subtitle = f"n={len(radii)}, slope={_format_float(slope)} s/um, r={_format_float(correlation)}"
    if omitted_labels:
        subtitle += f", omitted={len(omitted_labels)}"
    ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=9.0)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path, {
        "n_points": len(radii),
        "n_omitted": len(omitted_labels),
        "slope_tau_per_radius_s_per_um": slope,
        "slope_bootstrap_std_s_per_um": slope_std,
        "slope_bootstrap_ci95_low_s_per_um": slope_low,
        "slope_bootstrap_ci95_high_s_per_um": slope_high,
        "pearson_r": correlation,
    }


def write_overlay_video(
    path: Path | str,
    stack: np.ndarray,
    masks: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    fps: float,
) -> Path:
    """Write an MP4 QC video with mask boundaries, ellipse fits, and frame labels."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = np.asarray(stack)
    mask_stack = np.asarray(masks, dtype=bool)
    if frames.ndim != 3 or mask_stack.ndim != 3:
        raise ValueError("write_overlay_video expects stack and masks with shape (T, Y, X)")
    if frames.shape[0] != mask_stack.shape[0] or frames.shape[0] != len(rows):
        raise ValueError("stack, masks, and rows must have the same frame count")

    vmin, vmax = _movie_display_limits(frames)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame_index in range(frames.shape[0]):
            image = _scale_to_unit(frames[frame_index], vmin=vmin, vmax=vmax)
            rgb = _render_overlay_frame(image, mask_stack[frame_index], rows[frame_index])
            writer.append_data(rgb)

    return out_path


def _render_overlay_frame(image: np.ndarray, mask: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    for contour in measure.find_contours(mask.astype(np.float32), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color="#00d1ff", linewidth=1.6)

    ellipse_patch = _ellipse_patch_from_row(row)
    if ellipse_patch is not None:
        ax.add_patch(ellipse_patch)

    status_text = f"frame {int(row.get('frame', 0))}  t={_format_float(row.get('time_s'), precision=3)} s"
    ar_value = _float_or_nan(row.get("AR"))
    if np.isfinite(ar_value):
        status_text += f"  AR={ar_value:.3f}"
    else:
        status_text += f"  INVALID: {row.get('failure_reason', 'invalid')}"

    ax.text(
        0.02,
        0.04,
        status_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=9.0,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.65, "pad": 4},
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _ellipse_patch_from_row(row: dict[str, Any]) -> Ellipse | None:
    center_x = _float_or_nan(row.get("ellipse_center_x_px"))
    center_y = _float_or_nan(row.get("ellipse_center_y_px"))
    major = _float_or_nan(row.get("major_axis_px"))
    minor = _float_or_nan(row.get("minor_axis_px"))
    angle = _float_or_nan(row.get("ellipse_angle_rad"))
    if not np.all(np.isfinite([center_x, center_y, major, minor, angle])) or major <= 0.0 or minor <= 0.0:
        return None
    return Ellipse(
        (center_x, center_y),
        width=major,
        height=minor,
        angle=float(np.degrees(angle)),
        fill=False,
        edgecolor="#ffcc33",
        linewidth=1.7,
    )


def _movie_display_limits(stack: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(stack, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _scale_to_unit(frame: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    scaled = (np.asarray(frame, dtype=np.float64) - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0)


def _fit_start_time(rows: list[dict[str, Any]], fit_start_frame: int | None) -> float:
    if fit_start_frame is None:
        return 0.0
    for row in rows:
        if _optional_int(row.get("frame")) == fit_start_frame:
            value = _float_or_nan(row.get("time_s"))
            return float(value) if np.isfinite(value) else 0.0
    return 0.0


def _format_float(value: Any, *, precision: int = 3) -> str:
    number = _float_or_nan(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{precision}g}"


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)
