"""TIFF I/O and output-path helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}


def normalize_stack_shape(array: np.ndarray) -> np.ndarray:
    """Normalize supported image arrays to ``(T, Y, X)``.

    Supported inputs are:
    - ``(Y, X)``, returned as a one-frame stack;
    - ``(T, Y, X)``, returned unchanged;
    - ``(T, Y, X, 1)``, with the singleton channel axis removed;
    - ``(T, 1, Y, X)``, with the singleton channel axis removed.

    Ambiguous 4D arrays without one of the supported singleton channel axes are
    rejected so callers do not silently analyze the wrong dimension.
    """

    arr = np.asarray(array)

    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        if arr.shape[1] == 1:
            return arr[:, 0, :, :]
        raise ValueError(
            "unsupported 4D TIFF shape; expected (T, Y, X, 1) or (T, 1, Y, X), "
            f"got {arr.shape}"
        )

    raise ValueError(f"unsupported TIFF shape {arr.shape}; expected 2D, 3D, or supported 4D")


def load_tiff_stack(path: Path | str) -> np.ndarray:
    """Load a TIFF image or stack and normalize it to ``(T, Y, X)``."""

    stack_path = Path(path)
    if not stack_path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {stack_path}")

    return normalize_stack_shape(tifffile.imread(stack_path))


def find_tiff_files(data_dir: Path | str) -> list[Path]:
    """Return TIFF files in ``data_dir`` sorted by filename."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory not found: {root}")

    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES)


def prepare_output_directory(output_dir: Path | str, overwrite: bool = False) -> Path:
    """Create the run output directory and protect existing metadata by default."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    protected_paths = (root / "run_config.json", root / "run_log.txt")
    if not overwrite:
        existing = [path for path in protected_paths if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise ValueError(f"output metadata already exists in {root}: {names}; pass --overwrite to replace it")

    return root


def prepare_movie_output_directory(
    output_dir: Path | str,
    movie_path: Path | str,
    overwrite: bool = False,
) -> Path:
    """Create the per-movie output directory for a TIFF stack."""

    root = Path(output_dir)
    movie_dir = root / Path(movie_path).stem

    if movie_dir.exists() and any(movie_dir.iterdir()) and not overwrite:
        raise ValueError(f"movie output directory already exists and is not empty: {movie_dir}")

    movie_dir.mkdir(parents=True, exist_ok=True)
    return movie_dir


def write_tiff_stack(path: Path | str, stack: np.ndarray, dtype: np.dtype | type | None = None) -> Path:
    """Write a stack as TIFF, normalizing supported inputs to ``(T, Y, X)`` first."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_stack_shape(stack)
    if dtype is not None:
        normalized = normalized.astype(dtype, copy=False)

    tifffile.imwrite(out_path, normalized, photometric="minisblack")
    return out_path


def synthetic_relaxing_ellipse_stack(
    *,
    n_frames: int = 12,
    shape: tuple[int, int] = (64, 80),
    radius_px: float = 14.0,
    initial_ar: float = 2.0,
    tau_frames: float = 4.0,
    intensity: int = 40000,
    background: int = 1000,
) -> np.ndarray:
    """Create a deterministic uint16 stack with a relaxing bright ellipse."""

    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must be a positive (Y, X) tuple")
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")

    yy, xx = np.indices(shape)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    area_equivalent = radius_px**2

    stack = np.full((n_frames, shape[0], shape[1]), background, dtype=np.uint16)
    for frame in range(n_frames):
        ar = 1.0 + (initial_ar - 1.0) * np.exp(-frame / tau_frames)
        semi_major = np.sqrt(area_equivalent * ar)
        semi_minor = np.sqrt(area_equivalent / ar)
        mask = ((xx - cx) / semi_major) ** 2 + ((yy - cy) / semi_minor) ** 2 <= 1.0
        stack[frame, mask] = intensity

    return stack


def write_synthetic_tiff(
    path: Path | str,
    *,
    n_frames: int = 12,
    shape: tuple[int, int] = (64, 80),
    radius_px: float = 14.0,
    initial_ar: float = 2.0,
    tau_frames: float = 4.0,
) -> Path:
    """Write a deterministic synthetic relaxing-ellipse TIFF stack for tests."""

    stack = synthetic_relaxing_ellipse_stack(
        n_frames=n_frames,
        shape=shape,
        radius_px=radius_px,
        initial_ar=initial_ar,
        tau_frames=tau_frames,
    )
    return write_tiff_stack(path, stack)
