from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock
import warnings

import numpy as np
import pandas as pd
import pytest

from hypertorch.train.plot import BarChartPlotter
from hypertorch.types import ParsedMetrics


@pytest.fixture(autouse=True)
def mock_plotting_backend(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock matplotlib in sys.modules and importlib to run headless."""
    mock_matplotlib = MagicMock()
    mock_plt = MagicMock()

    def fake_barh(*args: Any, **kwargs: Any) -> list[MagicMock]:
        widths = args[1] if len(args) > 1 else kwargs.get("width", [])
        return [MagicMock() for _ in range(len(widths))]

    def fake_subplots(*args: Any, **kwargs: Any) -> tuple[MagicMock, Any]:
        fig = MagicMock()
        if kwargs.get("squeeze") is False or (args and len(args) >= 2):
            axes_list = []
            for _ in range(12):
                ax = MagicMock()
                ax.barh.side_effect = fake_barh
                axes_list.append(ax)
            ax_container = MagicMock()
            ax_container.flatten.return_value = axes_list
            return fig, ax_container

        ax = MagicMock()
        ax.barh.side_effect = fake_barh
        return fig, ax

    mock_plt.subplots.side_effect = fake_subplots

    def fake_savefig(path: Path | str, *args: Any, **kwargs: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    mock_plt.savefig.side_effect = fake_savefig
    mock_matplotlib.pyplot = mock_plt

    monkeypatch.setattr(
        "hypertorch.train.plot.plotter.importlib.util.find_spec",
        lambda name: MagicMock() if name == "matplotlib" else None,
    )
    monkeypatch.setitem(sys.modules, "matplotlib", mock_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", mock_plt)

    return {
        "matplotlib": mock_matplotlib,
        "plt": mock_plt,
    }


""" Grid Dimension & Axis Limit Tests """


@pytest.mark.parametrize(
    "n_plots, expected_dims",
    [
        (1, (1, 1)),
        (2, (2, 1)),
        (3, (3, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
        (7, (3, 3)),
    ],
)
def test_barchart_calculate_grid_dims(n_plots: int, expected_dims: tuple[int, int]) -> None:
    assert BarChartPlotter._calculate_grid_dims(n_plots) == expected_dims


def test_barchart_resolve_x_limits_empty_and_zero_baseline() -> None:
    plotter_zero = BarChartPlotter(zero_baseline=True)
    assert plotter_zero._resolve_x_limits([]) == (0.0, 1.0)

    left_pos, right_pos = plotter_zero._resolve_x_limits([0.631, 0.694])
    assert left_pos == 0.0
    assert right_pos == pytest.approx(0.694 * 1.18)

    left_neg, right_neg = plotter_zero._resolve_x_limits([-0.5, -0.2])
    assert left_neg == pytest.approx(-0.5 * 1.15)
    assert right_neg == 1.0

    # Zero span (identical scores) falls back to zero-baseline logic even if zero_baseline=False
    plotter_zoom = BarChartPlotter(zero_baseline=False)
    left_flat, right_flat = plotter_zoom._resolve_x_limits([0.75, 0.75])
    assert left_flat == 0.0
    assert right_flat == pytest.approx(0.75 * 1.18)


def test_barchart_resolve_x_limits_dynamic_zoom() -> None:
    plotter = BarChartPlotter(zero_baseline=False)
    x_left, x_right = plotter._resolve_x_limits([0.631, 0.691, 0.694])
    assert 0.50 < x_left < 0.631
    assert 0.694 < x_right < 0.85

    neg_left, neg_right = plotter._resolve_x_limits([-0.40, -0.35])
    assert neg_left < -0.40
    assert neg_right > -0.35


""" Annotation & Strict Zip Validation Tests """


def test_barchart_annotate_bars_skips_nan() -> None:
    mock_ax = MagicMock()
    bars = [MagicMock(), MagicMock()]
    BarChartPlotter._annotate_bars(
        ax=mock_ax,
        bars=bars,
        values=[0.8123, np.nan],
        offset=0.01,
    )
    mock_ax.text.assert_called_once()


def test_barchart_annotate_bars_mismatch_raises_value_error() -> None:
    mock_ax = MagicMock()
    with pytest.raises(ValueError, match=r"Bar count .* does not match value count"):
        BarChartPlotter._annotate_bars(
            ax=mock_ax,
            bars=[MagicMock()],
            values=[0.85, 0.90],
            offset=0.01,
        )


""" Data Collection & Sorting Tests """


def test_barchart_collect_metric_scores_orders_descending_and_handles_missing() -> None:
    m1 = ParsedMetrics(x_col="epoch", experiment_name="model_b")
    m1.add("auc", pd.DataFrame({"epoch": [0, 1], "split": ["val", "val"], "value": [0.5, 0.631]}))

    m2 = ParsedMetrics(x_col="epoch", experiment_name="model_a")
    m2.add("auc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.694]}))

    m3 = ParsedMetrics(x_col="epoch", experiment_name="no_auc")
    m3.add("loss", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.3]}))

    m4 = ParsedMetrics(x_col="epoch", experiment_name="train_only")
    m4.add("auc", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.9]}))

    scores = BarChartPlotter._collect_metric_scores([m1, m2, m3, m4], "auc", "val")
    assert scores == [
        ("model_a", 0.694),
        ("model_b", 0.631),
    ]


def test_barchart_collect_metric_scores_skips_empty_split_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="empty_split")
    m.add("auc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.8]}))

    mock_df = MagicMock()
    mock_df.__getitem__.side_effect = lambda key: (
        pd.Series(["val"]) if isinstance(key, str) and key == "split" else pd.DataFrame()
    )
    monkeypatch.setattr(m, "fetch", lambda var: mock_df)

    assert BarChartPlotter._collect_metric_scores([m], "auc", "val") == []


def test_barchart_collect_composite_data_handles_partial_metrics() -> None:
    m1 = ParsedMetrics(x_col="epoch", experiment_name="model_1")
    m1.add("auc", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.88]}))
    m1.add("acc", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.81]}))

    # Model 2 only has 'auc' for 'test' ('acc' is only on 'train')
    m2 = ParsedMetrics(x_col="epoch", experiment_name="model_2")
    m2.add("auc", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.84]}))
    m2.add("acc", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.90]}))

    # Model 3 has no test data at all
    m3 = ParsedMetrics(x_col="epoch", experiment_name="model_3")
    m3.add("auc", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.75]}))

    comp = BarChartPlotter._collect_composite_data([m1, m2, m3], ["auc", "acc"], "test")
    assert comp == [
        ("model_1", {"auc": 0.88, "acc": 0.81}),
        ("model_2", {"auc": 0.84}),
    ]


""" End-to-End Plot Generation & Routing Tests """


def test_barchart_missing_dependencies_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hypertorch.train.plot.plotter.Plotter._is_plotting_available",
        staticmethod(lambda: False),
    )
    plotter = BarChartPlotter()
    metrics = ParsedMetrics(x_col="epoch")

    with pytest.raises(ImportError, match="Plotting dependencies are not available"):
        plotter.plot(metrics)


def test_barchart_empty_targets_or_no_val_test_returns_empty(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_train_only"
    exp_dir.mkdir()

    m = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="0")
    m.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.4]}))

    plotter = BarChartPlotter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert plotter.plot(m, metric_names=["non_existent"]) == []
        assert plotter.plot(m) == []


def test_barchart_single_run_warning_does_not_abort(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_single"
    exp_dir.mkdir()

    m = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="single")
    m.add("auc", pd.DataFrame({"epoch": [0, 0], "split": ["val", "test"], "value": [0.85, 0.82]}))

    plotter = BarChartPlotter()
    with pytest.warns(UserWarning, match="only 1 experiment run was provided"):
        saved = plotter.plot(m)

    assert len(saved) == 2
    assert {p.name for p in saved} == {"BarChart_val_single.png", "BarChart_test_single.png"}
    for path in saved:
        assert path.is_file()


def test_barchart_plot_multi_run_generates_collections_and_comprehensive(
    tmp_path: Path,
) -> None:
    """Verify multi-run comparisons generate both collection and comprehensive charts."""
    logs_root = tmp_path / "logs"
    exp_1 = logs_root / "exp_1"
    exp_2 = logs_root / "exp_2"
    exp_1.mkdir(parents=True)
    exp_2.mkdir(parents=True)

    m1 = ParsedMetrics(x_col="epoch", experiment_dir=exp_1, experiment_name="mean_m")
    m1.add("auc", pd.DataFrame({"epoch": [0, 0], "split": ["val", "test"], "value": [0.70, 0.691]}))
    m1.add("acc", pd.DataFrame({"epoch": [0, 0], "split": ["val", "test"], "value": [0.80, 0.78]}))

    # m2 lacks 'acc' in test to also exercise NaN handling in composite rendering
    m2 = ParsedMetrics(x_col="epoch", experiment_dir=exp_2, experiment_name="max_m")
    m2.add("auc", pd.DataFrame({"epoch": [0, 0], "split": ["val", "test"], "value": [0.65, 0.694]}))
    m2.add("acc", pd.DataFrame({"epoch": [0], "split": ["val"], "value": [0.85]}))

    plotter = BarChartPlotter(zero_baseline=False)
    saved_plots = plotter.plot([m1, m2])

    expected_files = {
        logs_root / "plots" / "comparison" / "BarChart_val_comparison.png",
        logs_root / "plots" / "comparison" / "BarChart_test_comparison.png",
        logs_root / "plots" / "comparison" / "BarChart_comprehensive_val_comparison.png",
        logs_root / "plots" / "comparison" / "BarChart_comprehensive_test_comparison.png",
    }

    assert set(saved_plots) == expected_files
    for path in expected_files:
        assert path.is_file()


def test_barchart_plot_custom_output_dir_without_subfolder(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_barchart_dest"
    custom_dir.mkdir()

    m1 = ParsedMetrics(x_col="epoch", experiment_name="run_a")
    m1.add("auc", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.81]}))

    m2 = ParsedMetrics(x_col="epoch", experiment_name="run_b")
    m2.add("auc", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.86]}))

    plotter = BarChartPlotter()
    saved = plotter.plot([m1, m2], output_dir=custom_dir, create_subfolder=False)

    expected = custom_dir / "BarChart_test_comparison.png"
    assert saved == [expected]
    assert expected.is_file()
    assert not (custom_dir / "plots").exists()


def test_barchart_collect_composite_data_skips_empty_split_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = ParsedMetrics(x_col="epoch", experiment_name="empty_split")
    m.add("auc", pd.DataFrame({"epoch": [0], "split": ["test"], "value": [0.8]}))

    mock_df = MagicMock()
    mock_df.__getitem__.side_effect = lambda key: (
        pd.Series(["test"]) if isinstance(key, str) and key == "split" else pd.DataFrame()
    )
    monkeypatch.setattr(m, "fetch", lambda var: mock_df)

    assert BarChartPlotter._collect_composite_data([m], ["auc"], "test") == []


def test_barchart_skips_composite_when_multi_metric_split_is_empty(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp_multi_train_only"
    exp_dir.mkdir()

    m = ParsedMetrics(x_col="epoch", experiment_dir=exp_dir, experiment_name="train_only")
    m.add("loss", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.4]}))
    m.add("auc", pd.DataFrame({"epoch": [0], "split": ["train"], "value": [0.8]}))

    plotter = BarChartPlotter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert plotter.plot(m) == []
