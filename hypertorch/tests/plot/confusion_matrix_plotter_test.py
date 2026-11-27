from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from hypertorch.train.plot import ConfusionMatrixPlotter
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


def _make_binary_metrics(
    tmp_path: Path,
    experiment_name: str = "gcn_binary",
    splits: tuple[str, ...] = ("val", "test"),
    counts: dict[str, float] | None = None,
) -> MagicMock:
    """Create a binary ParsedMetrics with tp, fp, tn, fn across requested splits."""
    vals = counts or {"tp": 40.0, "fp": 5.0, "tn": 50.0, "fn": 10.0}
    frames: dict[str, pd.DataFrame] = {}
    for key in ("tp", "fp", "tn", "fn"):
        rows: list[tuple[int, str, float]] = []
        for split in splits:
            # Include two epochs to verify the plotter uses the final logged value
            rows.append((0, split, vals[key] / 2.0))
            rows.append((1, split, vals[key]))
        frames[key] = _make_df(rows)

    exp_dir = tmp_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return _make_parsed_metrics(frames, experiment_name=experiment_name, experiment_dir=exp_dir)


def _make_multiclass_metrics(
    tmp_path: Path,
    num_classes: int = 3,
    experiment_name: str = "gcn_multiclass",
    splits: tuple[str, ...] = ("val", "test"),
) -> MagicMock:
    """Create a multiclass ParsedMetrics with cm_{i}_{j} cells across requested splits."""
    frames: dict[str, pd.DataFrame] = {}
    for row in range(num_classes):
        for col in range(num_classes):
            val = float(20 if row == col else 2)
            rows = [(0, split, val) for split in splits]
            frames[f"cm_{row}_{col}"] = _make_df(rows)

    exp_dir = tmp_path / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return _make_parsed_metrics(frames, experiment_name=experiment_name, experiment_dir=exp_dir)


"""Mode detection & validation"""


def test_detect_matrix_mode_binary(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(tmp_path)
    assert ConfusionMatrixPlotter._detect_matrix_mode(parsed) == ("binary", 2)


@pytest.mark.parametrize("num_classes", [3, 7])
def test_detect_matrix_mode_multiclass(tmp_path: Path, num_classes: int) -> None:
    parsed = _make_multiclass_metrics(tmp_path, num_classes=num_classes)
    assert ConfusionMatrixPlotter._detect_matrix_mode(parsed) == ("multiclass", num_classes)


def test_detect_matrix_mode_raises_on_incomplete_binary(tmp_path: Path) -> None:
    frames = {
        "tp": _make_df([(0, "val", 10.0)]),
        "fp": _make_df([(0, "val", 2.0)]),
        "tn": _make_df([(0, "val", 15.0)]),
    }
    parsed = _make_parsed_metrics(frames, experiment_name="broken_binary", experiment_dir=tmp_path)
    with pytest.raises(ValueError, match="does not contain complete binary"):
        ConfusionMatrixPlotter._detect_matrix_mode(parsed)


def test_detect_matrix_mode_raises_on_incomplete_multiclass(tmp_path: Path) -> None:
    frames = {
        "cm_0_0": _make_df([(0, "val", 10.0)]),
        "cm_0_1": _make_df([(0, "val", 1.0)]),
        "cm_1_0": _make_df([(0, "val", 2.0)]),
    }
    parsed = _make_parsed_metrics(frames, experiment_name="broken_mc", experiment_dir=tmp_path)
    with pytest.raises(ValueError, match="does not contain complete binary"):
        ConfusionMatrixPlotter._detect_matrix_mode(parsed)


def test_detect_matrix_mode_raises_on_unrelated_metrics(tmp_path: Path) -> None:
    frames = {
        "loss": _make_df([(0, "val", 0.5)]),
        "accuracy": _make_df([(0, "val", 0.8)]),
    }
    parsed = _make_parsed_metrics(frames, experiment_name="no_cm", experiment_dir=tmp_path)
    with pytest.raises(ValueError, match="does not contain complete binary"):
        ConfusionMatrixPlotter._detect_matrix_mode(parsed)


"""Grid dimensions & matrix extraction"""


@pytest.mark.parametrize(
    ("n_plots", "expected"),
    [
        (1, (1, 1)),
        (2, (2, 1)),
        (3, (3, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
        (7, (3, 3)),
    ],
)
def test_calculate_grid_dims(n_plots: int, expected: tuple[int, int]) -> None:
    assert ConfusionMatrixPlotter._calculate_grid_dims(n_plots) == expected


def test_extract_split_matrix_binary_uses_last_value(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(
        tmp_path,
        counts={"tn": 50.0, "fp": 5.0, "fn": 10.0, "tp": 40.0},
    )
    matrix = ConfusionMatrixPlotter._extract_split_matrix(parsed, "test", "binary", 2)
    assert matrix is not None
    expected = np.array([[50.0, 5.0], [10.0, 40.0]])
    np.testing.assert_array_equal(matrix, expected)


def test_extract_split_matrix_multiclass(tmp_path: Path) -> None:
    parsed = _make_multiclass_metrics(tmp_path, num_classes=3)
    matrix = ConfusionMatrixPlotter._extract_split_matrix(parsed, "val", "multiclass", 3)
    assert matrix is not None
    expected = np.array(
        [
            [20.0, 2.0, 2.0],
            [2.0, 20.0, 2.0],
            [2.0, 2.0, 20.0],
        ]
    )
    np.testing.assert_array_equal(matrix, expected)


def test_extract_split_matrix_returns_none_when_split_missing(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(tmp_path, splits=("train",))
    assert ConfusionMatrixPlotter._extract_split_matrix(parsed, "test", "binary", 2) is None


"""Rendering"""


def test_plot_single_binary_run_prefers_test(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(tmp_path, experiment_name="gcn_bin", splits=("val", "test"))
    plotter = ConfusionMatrixPlotter()

    saved = plotter.plot(parsed)

    assert len(saved) == 1
    assert saved[0].name == "ConfusionMatrix_test_gcn_bin.png"
    assert saved[0].exists()
    assert saved[0].stat().st_size > 0


@pytest.mark.parametrize("num_classes", [3, 7])
def test_plot_single_multiclass_run(tmp_path: Path, num_classes: int) -> None:
    parsed = _make_multiclass_metrics(
        tmp_path,
        num_classes=num_classes,
        experiment_name=f"gcn_mc_{num_classes}",
    )
    plotter = ConfusionMatrixPlotter()

    saved = plotter.plot(parsed)

    assert len(saved) == 1
    assert saved[0].name == f"ConfusionMatrix_test_gcn_mc_{num_classes}.png"
    assert saved[0].exists()
    assert saved[0].stat().st_size > 0


def test_plot_multi_run_comparison_grid(tmp_path: Path) -> None:
    runs = [_make_binary_metrics(tmp_path, experiment_name=f"model_{i}") for i in range(5)]
    plotter = ConfusionMatrixPlotter()

    saved = plotter.plot(runs, output_dir=tmp_path)

    assert len(saved) == 1
    assert saved[0].parent == tmp_path / "plots" / "comparison"
    assert saved[0].name == "ConfusionMatrix_test_comparison.png"
    assert saved[0].exists()
    assert saved[0].stat().st_size > 0


def test_plot_without_subfolder(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(tmp_path, experiment_name="flat_run")
    out_dir = tmp_path / "custom_out"
    out_dir.mkdir()

    plotter = ConfusionMatrixPlotter()
    saved = plotter.plot(parsed, output_dir=out_dir, create_subfolder=False)

    assert len(saved) == 1
    assert saved[0].parent == out_dir


def test_plot_falls_back_to_val_and_handles_no_eval_splits(tmp_path: Path) -> None:
    plotter = ConfusionMatrixPlotter()

    val_only = _make_binary_metrics(tmp_path, experiment_name="val_only", splits=("val",))
    saved_val = plotter.plot(val_only)
    assert len(saved_val) == 1
    assert saved_val[0].name == "ConfusionMatrix_val_val_only.png"

    train_only = _make_binary_metrics(tmp_path, experiment_name="train_only", splits=("train",))
    saved_none = plotter.plot(train_only)
    assert saved_none == []


def test_plot_handles_zero_row_sums_safely(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(
        tmp_path,
        experiment_name="zero_row",
        counts={"tn": 0.0, "fp": 0.0, "fn": 5.0, "tp": 15.0},
    )
    plotter = ConfusionMatrixPlotter()
    saved = plotter.plot(parsed)
    assert len(saved) == 1
    assert saved[0].exists()


def test_plot_metric_names_validation(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(tmp_path)
    plotter = ConfusionMatrixPlotter()

    saved = plotter.plot(parsed, metric_names=["tp", "fp", "tn", "fn"])
    assert len(saved) == 1

    with pytest.raises(ValueError, match="metric_names must include all required"):
        plotter.plot(parsed, metric_names=["tp", "fp"])


def test_plot_raises_import_error_without_matplotlib(tmp_path: Path) -> None:
    parsed = _make_binary_metrics(tmp_path)
    plotter = ConfusionMatrixPlotter()

    with (
        patch.object(ConfusionMatrixPlotter, "_is_plotting_available", return_value=False),
        pytest.raises(ImportError, match="Plotting dependencies are not available"),
    ):
        plotter.plot(parsed)


def test_extract_split_matrix_multiclass_returns_none_when_split_missing(tmp_path: Path) -> None:
    parsed = _make_multiclass_metrics(tmp_path, num_classes=3, splits=("train",))
    assert ConfusionMatrixPlotter._extract_split_matrix(parsed, "test", "multiclass", 3) is None


def test_fetch_last_split_value_returns_none_when_filtered_df_empty(tmp_path: Path) -> None:
    class _SplitWrapper:
        def __str__(self) -> str:
            return "val"

    frames = {
        "tp": pd.DataFrame(
            [(0, _SplitWrapper(), 10.0)],
            columns=pd.Index(["epoch", "split", "value"]),
        )
    }
    parsed = _make_parsed_metrics(frames, experiment_dir=tmp_path)
    assert ConfusionMatrixPlotter._fetch_last_split_value(parsed, "tp", "val") is None
