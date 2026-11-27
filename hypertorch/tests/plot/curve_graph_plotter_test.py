from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hypertorch.train.plot import CurveGraphPlotter
from hypertorch.types.parsed_metrics import ParsedMetrics


def _make_parsed_metrics(
    metric_frames: dict[str, pd.DataFrame],
    experiment_name: str = "run_0",
    experiment_dir: Path | None = None,
    x_col: str = "epoch",
) -> MagicMock:
    """Create a spec-compliant mock ParsedMetrics instance for unit testing."""
    parsed = MagicMock(spec=ParsedMetrics)
    parsed.x_col = x_col
    parsed.experiment_name = experiment_name
    parsed.experiment_dir = experiment_dir
    parsed.__len__.return_value = len(metric_frames)
    parsed.names.return_value = list(metric_frames.keys())
    parsed.__contains__.side_effect = lambda key: key in metric_frames
    parsed.fetch.side_effect = lambda key: metric_frames[key]
    return parsed


def _make_df(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """Build a metric DataFrame with (epoch, split, value) rows."""
    return pd.DataFrame(rows, columns=pd.Index(["epoch", "split", "value"]))


def _make_threshold_frames(
    num_thresholds: int = 5,
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, pd.DataFrame]:
    """Create thresholded tp/fp/tn/fn DataFrames across requested splits."""
    frames: dict[str, pd.DataFrame] = {}
    positives = 50.0
    negatives = 50.0

    for t in range(num_thresholds):
        frac = t / max(num_thresholds - 1, 1)
        tp = positives * (1.0 - frac * 0.8)
        fn = positives - tp
        fp = negatives * (1.0 - frac)
        tn = negatives - fp

        for key, val in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
            rows: list[tuple[int, str, float]] = []
            for split in splits:
                rows.append((0, split, val * 0.5))
                rows.append((1, split, val))
            frames[f"{key}_t{t}"] = _make_df(rows)

    return frames


def _make_calibration_frames(
    num_bins: int = 4,
    splits: tuple[str, ...] = ("val", "test"),
    include_ece: bool = True,
) -> dict[str, pd.DataFrame]:
    """Create calibration bin DataFrames across requested splits."""
    frames: dict[str, pd.DataFrame] = {}
    for b in range(num_bins):
        conf = (b + 0.5) / num_bins
        acc = min(1.0, conf * 0.95)
        frames[f"cal_conf_{b}"] = _make_df([(0, s, conf) for s in splits])
        frames[f"cal_acc_{b}"] = _make_df([(0, s, acc) for s in splits])

    if include_ece:
        frames["ece"] = _make_df([(0, s, 0.042) for s in splits])

    return frames


def _make_curve_metrics(
    tmp_path: Path,
    experiment_name: str = "gcn_curves",
    splits: tuple[str, ...] = ("val", "test"),
    include_thresholds: bool = True,
    include_calibration: bool = True,
    include_ece: bool = True,
    num_thresholds: int = 5,
    num_bins: int = 4,
) -> MagicMock:
    """Create a ParsedMetrics with thresholded and/or calibration curve metrics."""
    frames: dict[str, pd.DataFrame] = {}
    if include_thresholds:
        frames.update(_make_threshold_frames(num_thresholds=num_thresholds, splits=splits))
    if include_calibration:
        frames.update(
            _make_calibration_frames(
                num_bins=num_bins,
                splits=splits,
                include_ece=include_ece,
            )
        )

    exp_dir = tmp_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return _make_parsed_metrics(frames, experiment_name=experiment_name, experiment_dir=exp_dir)


"""Initialization & detection validation"""


def test_init_raises_when_all_curves_disabled() -> None:
    with pytest.raises(ValueError, match="At least one curve type"):
        CurveGraphPlotter()


def test_detect_threshold_count_valid(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(tmp_path, num_thresholds=5, include_calibration=False)
    assert CurveGraphPlotter._detect_threshold_count(parsed) == 5


def test_detect_threshold_count_raises_on_incomplete_or_missing(tmp_path: Path) -> None:
    # Missing fn_t1
    frames = {
        "tp_t0": _make_df([(0, "test", 10.0)]),
        "fp_t0": _make_df([(0, "test", 2.0)]),
        "tn_t0": _make_df([(0, "test", 8.0)]),
        "fn_t0": _make_df([(0, "test", 1.0)]),
        "tp_t1": _make_df([(0, "test", 8.0)]),
        "fp_t1": _make_df([(0, "test", 1.0)]),
        "tn_t1": _make_df([(0, "test", 9.0)]),
    }
    parsed = _make_parsed_metrics(frames, experiment_name="broken_thresh", experiment_dir=tmp_path)
    with pytest.raises(ValueError, match="does not contain complete thresholded"):
        CurveGraphPlotter._detect_threshold_count(parsed)

    # No threshold keys at all
    unrelated = _make_parsed_metrics(
        {"loss": _make_df([(0, "test", 0.5)])},
        experiment_name="no_thresh",
        experiment_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="does not contain complete thresholded"):
        CurveGraphPlotter._detect_threshold_count(unrelated)


def test_detect_calibration_bins_valid(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(tmp_path, num_bins=4, include_thresholds=False)
    assert CurveGraphPlotter._detect_calibration_bins(parsed) == 4


def test_detect_calibration_bins_raises_on_incomplete_or_missing(tmp_path: Path) -> None:
    frames = {
        "cal_conf_0": _make_df([(0, "test", 0.25)]),
        "cal_acc_0": _make_df([(0, "test", 0.20)]),
        "cal_conf_1": _make_df([(0, "test", 0.75)]),
        # missing cal_acc_1
    }
    parsed = _make_parsed_metrics(frames, experiment_name="broken_cal", experiment_dir=tmp_path)
    with pytest.raises(ValueError, match="does not contain complete calibration"):
        CurveGraphPlotter._detect_calibration_bins(parsed)

    unrelated = _make_parsed_metrics(
        {"accuracy": _make_df([(0, "test", 0.9)])},
        experiment_name="no_cal",
        experiment_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="does not contain complete calibration"):
        CurveGraphPlotter._detect_calibration_bins(unrelated)


def test_fetch_last_split_value_returns_none_when_filtered_df_empty(tmp_path: Path) -> None:
    class _SplitWrapper:
        def __str__(self) -> str:
            return "test"

    frames = {
        "tp_t0": pd.DataFrame(
            [(0, _SplitWrapper(), 10.0)],
            columns=pd.Index(["epoch", "split", "value"]),
        )
    }
    parsed = _make_parsed_metrics(frames, experiment_dir=tmp_path)
    assert CurveGraphPlotter._fetch_last_split_value(parsed, "tp_t0", "test") is None


"""Rendering & split resolution"""


def test_plot_all_curves_prefers_test_over_val(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(tmp_path, experiment_name="gcn_all", splits=("val", "test"))
    plotter = CurveGraphPlotter(roc=True, toc=True, det=True, pr=True, calibration=True, cumulative=True)

    saved = plotter.plot(parsed)

    assert len(saved) == 6
    names = {p.name for p in saved}
    assert names == {
        "CurveGraph_ROC_test_gcn_all.png",
        "CurveGraph_PR_test_gcn_all.png",
        "CurveGraph_DET_test_gcn_all.png",
        "CurveGraph_TOC_test_gcn_all.png",
        "CurveGraph_Calibration_test_gcn_all.png",
        "CurveGraph_Cumulative_test_gcn_all.png"
    }
    for path in saved:
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_falls_back_to_val_and_handles_no_eval_splits(tmp_path: Path) -> None:
    plotter = CurveGraphPlotter(roc=True, calibration=True)

    val_only = _make_curve_metrics(tmp_path, experiment_name="val_only", splits=("val",))
    saved_val = plotter.plot(val_only)
    assert len(saved_val) == 2
    assert {p.name for p in saved_val} == {
        "CurveGraph_ROC_val_val_only.png",
        "CurveGraph_Calibration_val_val_only.png",
    }

    train_only = _make_curve_metrics(tmp_path, experiment_name="train_only", splits=("train",))
    saved_none = plotter.plot(train_only)
    assert saved_none == []


def test_plot_calibration_only_without_ece(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(
        tmp_path,
        experiment_name="cal_no_ece",
        include_thresholds=False,
        include_calibration=True,
        include_ece=False,
    )
    plotter = CurveGraphPlotter(calibration=True)

    saved = plotter.plot(parsed)

    assert len(saved) == 1
    assert saved[0].name == "CurveGraph_Calibration_test_cal_no_ece.png"
    assert saved[0].exists()


def test_plot_pr_only_without_roc(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(tmp_path, experiment_name="pr_only", include_calibration=False)
    plotter = CurveGraphPlotter(pr=True)

    saved = plotter.plot(parsed)

    assert len(saved) == 1
    assert saved[0].name == "CurveGraph_PR_test_pr_only.png"
    assert saved[0].exists()


def test_plot_multi_run_comparison(tmp_path: Path) -> None:
    runs = [_make_curve_metrics(tmp_path, experiment_name=f"model_{i}") for i in range(3)]
    plotter = CurveGraphPlotter(roc=True, pr=True, calibration=True)

    saved = plotter.plot(runs, output_dir=tmp_path)

    assert len(saved) == 3
    for path in saved:
        assert path.parent == tmp_path / "plots" / "comparison"
        assert path.name.endswith("_comparison.png")
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_without_subfolder(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(tmp_path, experiment_name="flat_run", include_calibration=False)
    out_dir = tmp_path / "custom_out"
    out_dir.mkdir()

    plotter = CurveGraphPlotter(roc=True)
    saved = plotter.plot(parsed, output_dir=out_dir, create_subfolder=False)

    assert len(saved) == 1
    assert saved[0].parent == out_dir


def test_plot_handles_zero_denominators_safely(tmp_path: Path) -> None:
    frames = {
        f"{k}_t{i}": _make_df([(0, "test", 0.0)])
        for i in range(2)
        for k in ("tp", "fp", "tn", "fn")
    }
    parsed = _make_parsed_metrics(frames, experiment_name="all_zeros", experiment_dir=tmp_path)
    plotter = CurveGraphPlotter(roc=True, pr=True, det=True, toc=True)

    saved = plotter.plot(parsed)
    assert len(saved) == 4
    assert all(p.exists() for p in saved)


def test_plot_metric_names_validation(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(tmp_path, num_thresholds=2, num_bins=2)
    valid_thresh = [f"{k}_t{i}" for i in range(2) for k in ("tp", "fp", "tn", "fn")]
    valid_cal = [f"cal_{k}_{b}" for b in range(2) for k in ("conf", "acc")]

    plotter = CurveGraphPlotter(roc=True, calibration=True)
    saved = plotter.plot(parsed, metric_names=valid_thresh + valid_cal)
    assert len(saved) == 2

    with pytest.raises(ValueError, match="metric_names must include all required thresholded"):
        plotter.plot(parsed, metric_names=valid_cal)

    with pytest.raises(ValueError, match="metric_names must include all required calibration"):
        plotter.plot(parsed, metric_names=valid_thresh)


def test_plot_raises_import_error_without_matplotlib(tmp_path: Path) -> None:
    parsed = _make_curve_metrics(tmp_path)
    plotter = CurveGraphPlotter(roc=True)

    with (
        patch.object(CurveGraphPlotter, "_is_plotting_available", return_value=False),
        pytest.raises(ImportError, match="Plotting dependencies are not available"),
    ):
        plotter.plot(parsed)
