"""AR relaxation fitting and movie-level summary utilities."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from droplet_fusion.config import FitConfig


SUMMARY_RESULT_COLUMNS = [
    "filename",
    "status",
    "notes",
    "n_frames",
    "n_valid_frames",
    "fit_start_frame",
    "n_fit_frames",
    "A",
    "tau_fusion_s",
    "tau_fusion_std_s",
    "R_px",
    "R_um",
    "inverse_capillary_velocity_s_per_um",
    "inverse_capillary_velocity_s_per_m",
    "fit_rmse",
    "fit_r2",
    "baseline",
]


@dataclass(frozen=True)
class RelaxationFitResult:
    """Result of fitting AR(t) = 1 + A exp(-t / tau)."""

    valid: bool
    failure_reason: str | None
    fit_start_frame: int | None
    n_fit_frames: int
    used_frame_indices: list[int]
    A: float
    tau_fusion_s: float
    tau_fusion_std_s: float
    fit_rmse: float
    fit_r2: float
    baseline: float


@dataclass(frozen=True)
class RadiusResult:
    """Final fused radius estimated from final valid mask areas."""

    valid: bool
    failure_reason: str | None
    n_frames_used: int
    R_px: float
    R_um: float
    note: str


@dataclass(frozen=True)
class MovieFitResult:
    """Movie-level fit, radius, and inverse capillary velocity summary."""

    filename: str
    status: str
    notes: str
    n_frames: int
    n_valid_frames: int
    fit_start_frame: int | None
    n_fit_frames: int
    A: float
    tau_fusion_s: float
    tau_fusion_std_s: float
    R_px: float
    R_um: float
    inverse_capillary_velocity_s_per_um: float
    inverse_capillary_velocity_s_per_m: float
    fit_rmse: float
    fit_r2: float
    baseline: float
    used_frame_indices: list[int]

    def to_summary_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("used_frame_indices")
        return row


def ar_model(time_s: np.ndarray, A: float, tau: float) -> np.ndarray:
    """Exponential post-fusion relaxation model with a fixed AR=1 asymptote."""

    return 1.0 + A * np.exp(-time_s / tau)


def ar_model_baseline(time_s: np.ndarray, A: float, tau: float, baseline: float) -> np.ndarray:
    """Exponential relaxation with a free asymptote (baseline >= 1)."""

    return baseline + A * np.exp(-time_s / tau)


def load_manual_fit_starts(path: Path | str | None) -> dict[str, int]:
    """Load filename -> fit_start_frame mappings from a CSV file."""

    if path is None:
        return {}

    starts: dict[str, int] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "filename" not in reader.fieldnames or "fit_start_frame" not in reader.fieldnames:
            raise ValueError("manual fit starts CSV must contain filename and fit_start_frame columns")
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            starts[filename] = int(row["fit_start_frame"])
    return starts


def fit_ar_relaxation(
    rows: list[dict[str, Any]],
    config: FitConfig,
    *,
    filename: str,
    manual_fit_starts: dict[str, int] | None = None,
    min_fit_frames: int = 5,
) -> RelaxationFitResult:
    """Choose a fit window and fit the AR relaxation model."""

    valid_rows = _valid_ar_rows(rows)
    if not valid_rows:
        return _invalid_fit("too_few_valid_ar_frames")

    manual_fit_starts = manual_fit_starts or {}
    fit_start_frame = _choose_fit_start_frame(valid_rows, config, filename, manual_fit_starts)
    if fit_start_frame is None:
        return _invalid_fit("manual_fit_start_missing")
    if (
        config.fit_start_mode == "max_AR"
        and config.fail_max_ar_at_last_frame
        and _is_last_valid_ar_frame(valid_rows, fit_start_frame)
    ):
        return _invalid_fit("max_ar_at_last_frame", fit_start_frame=fit_start_frame, n_fit_frames=1)

    fit_rows = [row for row in valid_rows if int(row["frame"]) >= fit_start_frame]
    if config.trim_tail_to_floor:
        fit_rows = _trim_tail_to_floor(fit_rows, config.tail_floor_fraction, min_fit_frames)
    if len(fit_rows) < min_fit_frames:
        return _invalid_fit("too_few_fit_frames", fit_start_frame=fit_start_frame, n_fit_frames=len(fit_rows))

    fit_start_time = _time_for_frame(rows, fit_start_frame)
    time_s = np.array([float(row["time_s"]) - fit_start_time for row in fit_rows], dtype=np.float64)
    ar_values = np.array([float(row["AR"]) for row in fit_rows], dtype=np.float64)
    if np.any(time_s < 0) or not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(ar_values)):
        return _invalid_fit("invalid_fit_values", fit_start_frame=fit_start_frame, n_fit_frames=len(fit_rows))

    max_time = float(np.max(time_s))
    positive_deltas = np.diff(np.unique(time_s))
    fallback_tau = float(positive_deltas[0]) if positive_deltas.size else 1.0
    tau0 = max(max_time / 3.0, fallback_tau, 1e-6)

    if config.float_baseline:
        baseline0 = max(float(np.min(ar_values)), 1.0)
        A0 = max(float(np.max(ar_values) - baseline0), 1e-6)
        model = ar_model_baseline
        p0: tuple[float, ...] = (A0, tau0, baseline0)
        bounds = ([0.0, 1e-12, 1.0], [np.inf, np.inf, np.inf])
    else:
        A0 = max(float(np.max(ar_values) - 1.0), 1e-6)
        model = ar_model
        p0 = (A0, tau0)
        bounds = ([0.0, 1e-12], [np.inf, np.inf])

    try:
        params, covariance = curve_fit(
            model,
            time_s,
            ar_values,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return _invalid_fit("ar_fit_failed", fit_start_frame=fit_start_frame, n_fit_frames=len(fit_rows))

    A = float(params[0])
    tau = float(params[1])
    baseline = float(params[2]) if config.float_baseline else 1.0
    if not math.isfinite(tau) or tau <= 0.0:
        return _invalid_fit("tau_fusion_nonpositive", fit_start_frame=fit_start_frame, n_fit_frames=len(fit_rows))

    fitted = model(time_s, *params)
    residuals = ar_values - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((ar_values - np.mean(ar_values)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    tau_std = float("nan")
    if covariance.shape == (2, 2) and np.all(np.isfinite(covariance)) and covariance[1, 1] >= 0.0:
        tau_std = float(np.sqrt(covariance[1, 1]))

    return RelaxationFitResult(
        valid=True,
        failure_reason=None,
        fit_start_frame=fit_start_frame,
        n_fit_frames=len(fit_rows),
        used_frame_indices=[int(row["frame"]) for row in fit_rows],
        A=A,
        tau_fusion_s=tau,
        tau_fusion_std_s=tau_std,
        fit_rmse=rmse,
        fit_r2=r2,
        baseline=baseline,
    )


def estimate_final_radius(
    rows: list[dict[str, Any]],
    *,
    n_final_frames: int,
    um_per_pixel: float,
) -> RadiusResult:
    """Estimate final fused radius from final valid mask areas."""

    valid_area_rows = [
        row
        for row in rows
        if _is_truthy(row.get("valid_mask")) and _is_finite_positive(row.get("area_px"))
    ]
    if not valid_area_rows:
        return RadiusResult(
            valid=False,
            failure_reason="too_few_valid_area_frames",
            n_frames_used=0,
            R_px=float("nan"),
            R_um=float("nan"),
            note="",
        )

    selected_rows = valid_area_rows[-n_final_frames:]
    area_final_px = float(np.median([float(row["area_px"]) for row in selected_rows]))
    R_px = float(math.sqrt(area_final_px / math.pi))
    R_um = R_px * um_per_pixel
    note = ""
    if len(selected_rows) < n_final_frames:
        note = f"R estimated from {len(selected_rows)} final valid frame(s)"

    if not math.isfinite(R_um) or R_um <= 0.0:
        return RadiusResult(
            valid=False,
            failure_reason="R_um_nonpositive",
            n_frames_used=len(selected_rows),
            R_px=R_px,
            R_um=R_um,
            note=note,
        )

    return RadiusResult(
        valid=True,
        failure_reason=None,
        n_frames_used=len(selected_rows),
        R_px=R_px,
        R_um=R_um,
        note=note,
    )


def analyze_movie_measurements(
    *,
    filename: str,
    rows: list[dict[str, Any]],
    fit_config: FitConfig,
    um_per_pixel: float,
    manual_fit_starts: dict[str, int] | None = None,
) -> MovieFitResult:
    """Create the Agent 5 movie-level fit summary."""

    n_frames = len(rows)
    n_valid_frames = sum(1 for row in rows if _is_truthy(row.get("valid_mask")) and _is_finite(row.get("AR")))
    fit = fit_ar_relaxation(rows, fit_config, filename=filename, manual_fit_starts=manual_fit_starts)
    radius = estimate_final_radius(
        rows,
        n_final_frames=fit_config.n_final_frames_for_r,
        um_per_pixel=um_per_pixel,
    )

    notes: list[str] = []
    status = "ok"
    if n_valid_frames < n_frames:
        notes.append(f"{n_frames - n_valid_frames} frame(s) lacked valid AR")
        status = "warning"
    if radius.note:
        notes.append(radius.note)
        status = "warning"

    inverse_s_per_um = float("nan")
    inverse_s_per_m = float("nan")
    if fit.valid and radius.valid:
        inverse_s_per_um = fit.tau_fusion_s / radius.R_um
        inverse_s_per_m = inverse_s_per_um * 1e6
        if _is_finite(fit.fit_r2) and fit.fit_r2 < fit_config.min_fit_r2_warning:
            notes.append(
                "low_fit_r2 "
                f"({fit.fit_r2:.3g} < {fit_config.min_fit_r2_warning:.3g}); "
                "review AR_fit.png/overlay.mp4 and consider --fit-start-mode manual with --manual-fit-starts"
            )
            status = "warning"
        if not math.isfinite(inverse_s_per_um) or inverse_s_per_um <= 0.0:
            notes.append("inverse_capillary_velocity_nonfinite")
            status = "failed"
    else:
        if not fit.valid:
            notes.append(_fit_failure_note(fit.failure_reason))
        if not radius.valid:
            notes.append(radius.failure_reason or "radius_estimation_failed")
        status = "failed"

    return MovieFitResult(
        filename=filename,
        status=status,
        notes="; ".join(notes),
        n_frames=n_frames,
        n_valid_frames=n_valid_frames,
        fit_start_frame=fit.fit_start_frame,
        n_fit_frames=fit.n_fit_frames,
        A=fit.A,
        tau_fusion_s=fit.tau_fusion_s,
        tau_fusion_std_s=fit.tau_fusion_std_s,
        R_px=radius.R_px,
        R_um=radius.R_um,
        inverse_capillary_velocity_s_per_um=inverse_s_per_um,
        inverse_capillary_velocity_s_per_m=inverse_s_per_m,
        fit_rmse=fit.fit_rmse,
        fit_r2=fit.fit_r2,
        baseline=fit.baseline,
        used_frame_indices=fit.used_frame_indices,
    )


def mark_used_for_fit(rows: list[dict[str, Any]], used_frame_indices: list[int]) -> None:
    """Mark rows that were used in the relaxation fit in place."""

    used = set(used_frame_indices)
    for row in rows:
        row["used_for_fit"] = int(row["frame"]) in used


def _choose_fit_start_frame(
    valid_rows: list[dict[str, Any]],
    config: FitConfig,
    filename: str,
    manual_fit_starts: dict[str, int],
) -> int | None:
    if config.fit_start_mode == "manual":
        return manual_fit_starts.get(filename)
    if config.fit_start_mode == "first_valid":
        return min(int(row["frame"]) for row in valid_rows)
    if config.fit_start_mode == "max_AR":
        return int(max(valid_rows, key=lambda row: float(row["AR"]))["frame"])
    if config.fit_start_mode == "decay_onset":
        return _detect_decay_onset(valid_rows)
    raise ValueError(f"unsupported fit start mode: {config.fit_start_mode}")


def _detect_decay_onset(valid_rows: list[dict[str, Any]]) -> int:
    """Return the frame at the top of the sustained post-fusion decay.

    A single ``max_AR`` point is noise-sensitive and, when the movie opens with a
    near-flat pre-collapse plateau, can land anywhere along it. This lightly
    smooths AR, locates the smoothed peak, then walks forward through any plateau
    that stays within a small tolerance of the peak so that t = 0 is placed at the
    onset of the real decay rather than at the start of the plateau.
    """

    frames = [int(row["frame"]) for row in valid_rows]
    ar = [float(row["AR"]) for row in valid_rows]
    smoothed = _moving_average(ar, window=3)
    peak_index = int(np.argmax(smoothed))
    peak = smoothed[peak_index]
    floor = min(smoothed[peak_index:]) if peak_index < len(smoothed) else peak
    amplitude = peak - floor
    if amplitude <= 0.0:
        return frames[peak_index]
    # Onset = first frame at/after the smoothed peak that has dropped a small
    # fraction of the way toward the floor, i.e. where the sustained decay
    # actually begins. This places t = 0 at the end of any near-flat plateau
    # while staying robust to plateau-level measurement scatter.
    threshold = peak - 0.05 * amplitude
    onset_index = next(
        (index for index in range(peak_index, len(smoothed)) if smoothed[index] <= threshold),
        peak_index,
    )
    return frames[onset_index]


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    array = np.asarray(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return list(np.convolve(array, kernel, mode="same"))


def _trim_tail_to_floor(
    fit_rows: list[dict[str, Any]],
    fraction: float,
    min_fit_frames: int,
) -> list[dict[str, Any]]:
    """Drop the long, near-circular tail once AR has descended to the floor.

    The exponential relaxation carries essentially all of its information in the
    descent. Frames where AR has already flattened near its minimum add little
    signal but a lot of measurement scatter, so they dominate the least-squares
    residual and bias tau. This keeps frames down to (and including) the first
    one that reaches within ``fraction`` of the descent floor, then stops.
    """

    if len(fit_rows) <= min_fit_frames:
        return fit_rows
    ar = [float(row["AR"]) for row in fit_rows]
    ar_start = ar[0]
    ar_min = min(ar)
    span = ar_start - ar_min
    if span <= 0.0:
        return fit_rows
    target = ar_min + fraction * span
    cut = next((idx for idx, value in enumerate(ar) if value <= target), None)
    if cut is None:
        return fit_rows
    kept = fit_rows[: cut + 1]
    if len(kept) < min_fit_frames:
        kept = fit_rows[:min_fit_frames]
    return kept


def _valid_ar_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _is_truthy(row.get("valid_mask")) and _is_finite(row.get("AR")) and float(row["AR"]) >= 1.0
    ]


def _is_last_valid_ar_frame(valid_rows: list[dict[str, Any]], frame: int) -> bool:
    return frame == max(int(row["frame"]) for row in valid_rows)


def _fit_failure_note(failure_reason: str | None) -> str:
    if failure_reason == "max_ar_at_last_frame":
        return (
            "max_ar_at_last_frame; automatic max_AR fit start landed on the last valid AR frame; "
            "review AR_fit.png/overlay.mp4 and use --fit-start-mode manual with --manual-fit-starts"
        )
    return failure_reason or "ar_fit_failed"


def _time_for_frame(rows: list[dict[str, Any]], frame: int) -> float:
    for row in rows:
        if int(row["frame"]) == frame and _is_finite(row.get("time_s")):
            return float(row["time_s"])
    if len(rows) >= 2:
        dt = float(rows[1]["time_s"]) - float(rows[0]["time_s"])
        return frame * dt
    return 0.0


def _invalid_fit(
    failure_reason: str,
    *,
    fit_start_frame: int | None = None,
    n_fit_frames: int = 0,
) -> RelaxationFitResult:
    return RelaxationFitResult(
        valid=False,
        failure_reason=failure_reason,
        fit_start_frame=fit_start_frame,
        n_fit_frames=n_fit_frames,
        used_frame_indices=[],
        A=float("nan"),
        tau_fusion_s=float("nan"),
        tau_fusion_std_s=float("nan"),
        fit_rmse=float("nan"),
        fit_r2=float("nan"),
        baseline=float("nan"),
    )


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_finite_positive(value: Any) -> bool:
    return _is_finite(value) and float(value) > 0.0


def _is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)
