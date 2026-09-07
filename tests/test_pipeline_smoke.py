from __future__ import annotations

import json

from droplet_fusion.cli import build_parser, config_from_args
from droplet_fusion.io import write_synthetic_tiff
from droplet_fusion.pipeline import run_pipeline


def test_cli_help_includes_required_units(capsys):
    parser = build_parser()

    try:
        parser.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    help_text = capsys.readouterr().out
    assert "--seconds-per-frame" in help_text
    assert "--um-per-pixel" in help_text
    assert "--outer-envelope-mode" in help_text
    assert "--min-fit-r2-warning" in help_text
    assert "--fail-max-ar-at-last-frame" in help_text


def test_scaffold_run_writes_config_and_log(tmp_path):
    data_dir = tmp_path / "data"
    write_synthetic_tiff(data_dir / "movie_001.tif", n_frames=3, shape=(48, 56), radius_px=8.0)
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "output"),
            "--seconds-per-frame",
            "0.05",
            "--um-per-pixel",
            "0.108",
            "--no-make-videos",
        ]
    )
    config = config_from_args(args)

    result = run_pipeline(config)

    assert result.config_path.exists()
    assert result.log_path.exists()
    assert (tmp_path / "output" / "movie_001" / "masks.tif").exists()
    assert (tmp_path / "output" / "movie_001" / "segmentation_qc.csv").exists()
    assert (tmp_path / "output" / "movie_001" / "frame_measurements.csv").exists()
    payload = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert payload["pipeline_state"] == "visualization"
    assert payload["config"]["seconds_per_frame"] == 0.05
    assert payload["config"]["um_per_pixel"] == 0.108
    assert payload["config"]["visualization"]["make_videos"] is False
    assert payload["config"]["segmentation"]["segmentation_mode"] == "temporal"
    assert payload["config"]["segmentation"]["outer_envelope_mode"] == "none"
    assert payload["config"]["fitting"]["min_fit_r2_warning"] == 0.8
    assert payload["config"]["fitting"]["fail_max_ar_at_last_frame"] is True
    assert payload["movies"][0]["filename"] == "movie_001.tif"
    assert payload["movies"][0]["segmentation_qc_path"]
    assert payload["movies"][0]["n_valid_segmentation_frames"] == 3
    assert payload["movies"][0]["n_valid_measurement_frames"] == 3
    assert (tmp_path / "output" / "movie_001" / "movie_result.json").exists()
    assert (tmp_path / "output" / "movie_001" / "AR_fit.png").exists()
    assert (tmp_path / "output" / "summary_results.csv").exists()
    assert (tmp_path / "output" / "inverse_capillary_velocity_summary.png").exists()
