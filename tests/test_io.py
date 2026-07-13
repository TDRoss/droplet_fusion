from __future__ import annotations

import numpy as np
import pytest
import tifffile

from droplet_fusion.io import (
    find_tiff_files,
    load_tiff_stack,
    normalize_stack_shape,
    prepare_movie_output_directory,
    prepare_output_directory,
    synthetic_relaxing_ellipse_stack,
    write_synthetic_tiff,
    write_tiff_stack,
)


def test_normalize_stack_shape_accepts_single_frame_2d():
    image = np.arange(20, dtype=np.uint16).reshape(4, 5)

    stack = normalize_stack_shape(image)

    assert stack.shape == (1, 4, 5)
    assert stack.dtype == np.uint16
    np.testing.assert_array_equal(stack[0], image)


def test_normalize_stack_shape_keeps_tyx_stack():
    image = np.zeros((3, 4, 5), dtype=np.uint16)

    stack = normalize_stack_shape(image)

    assert stack.shape == (3, 4, 5)
    assert stack.dtype == np.uint16


def test_normalize_stack_shape_removes_trailing_singleton_channel():
    image = np.zeros((3, 4, 5, 1), dtype=np.uint16)

    stack = normalize_stack_shape(image)

    assert stack.shape == (3, 4, 5)


def test_normalize_stack_shape_removes_second_axis_singleton_channel():
    image = np.zeros((3, 1, 4, 5), dtype=np.uint16)

    stack = normalize_stack_shape(image)

    assert stack.shape == (3, 4, 5)


def test_normalize_stack_shape_rejects_ambiguous_4d_shape():
    image = np.zeros((3, 2, 4, 5), dtype=np.uint16)

    with pytest.raises(ValueError, match="unsupported 4D TIFF shape"):
        normalize_stack_shape(image)


def test_load_tiff_stack_preserves_uint16_data(tmp_path):
    original = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, original, photometric="minisblack")

    stack = load_tiff_stack(path)

    assert stack.shape == (3, 4, 5)
    assert stack.dtype == np.uint16
    np.testing.assert_array_equal(stack, original)


def test_find_tiff_files_returns_sorted_tif_and_tiff_only(tmp_path):
    (tmp_path / "b.tiff").write_bytes(b"placeholder")
    (tmp_path / "a.tif").write_bytes(b"placeholder")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    paths = find_tiff_files(tmp_path)

    assert [path.name for path in paths] == ["a.tif", "b.tiff"]


def test_prepare_output_directory_protects_existing_metadata(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "run_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="pass --overwrite"):
        prepare_output_directory(output_dir, overwrite=False)

    assert prepare_output_directory(output_dir, overwrite=True) == output_dir


def test_prepare_movie_output_directory_protects_nonempty_directory(tmp_path):
    output_dir = tmp_path / "output"
    movie_dir = output_dir / "movie_001"
    movie_dir.mkdir(parents=True)
    (movie_dir / "frame_measurements.csv").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        prepare_movie_output_directory(output_dir, tmp_path / "movie_001.tif", overwrite=False)

    assert prepare_movie_output_directory(output_dir, tmp_path / "movie_001.tif", overwrite=True) == movie_dir


def test_synthetic_tiff_writer_round_trips_uint16_stack(tmp_path):
    path = tmp_path / "synthetic.tif"

    written = write_synthetic_tiff(path, n_frames=5, shape=(32, 40), radius_px=7.0)
    stack = load_tiff_stack(written)

    assert stack.shape == (5, 32, 40)
    assert stack.dtype == np.uint16
    assert stack.max() > stack.min()
    assert np.count_nonzero(stack[0] > stack.min()) > 0


def test_write_tiff_stack_normalizes_2d_input(tmp_path):
    path = tmp_path / "single_frame.tif"
    image = np.ones((10, 12), dtype=np.uint16)

    write_tiff_stack(path, image)

    stack = load_tiff_stack(path)
    assert stack.shape == (1, 10, 12)


def test_synthetic_relaxing_ellipse_stack_has_constant_bright_area_approximately():
    stack = synthetic_relaxing_ellipse_stack(n_frames=6, shape=(48, 56), radius_px=9.0)
    bright_counts = np.count_nonzero(stack > stack.min(), axis=(1, 2))

    assert np.ptp(bright_counts) <= 10
