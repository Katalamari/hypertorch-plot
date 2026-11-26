from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np

from hypertorch.train.plot.plotter import Plotter
from hypertorch.types.parsed_metrics import ParsedMetrics


class BarChartPlotter(Plotter):
    """
    Generates consolidated horizontal bar chart collections and multi-metric vertical
    column charts for Val and Test splits, with persistent model colors and sorted leaderboards.
    """

    def __init__(self, zero_baseline: bool = False) -> None:
        """
        Args:
            zero_baseline: Whether to force metric axes to start at 0.0.
        """
        super().__init__()
        self.zero_baseline = zero_baseline

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """
        Render consolidated metric collections and comprehensive vertical column charts.

        Args:
            metrics: Single ParsedMetrics or sequence of metrics to plot.
            metric_names: Optional subset of metric variables to plot.
            output_dir: Optional destination path for plot outputs.
            create_subfolder: Whether to save inside categorized subfolders.

        Returns:
            saved_plots: List of file paths to generated PNG images.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        plt = self._init_charting()
        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)

        if len(metrics_list) == 1:
            warnings.warn(
                "BarChartPlotter is designed for multi-model comparisons, but only 1 "
                "experiment run was provided.",
                category=UserWarning,
                stacklevel=2,
            )

        targets = self._resolve_target_metrics(metrics_list, metric_names)
        if not targets:
            return []

        plots_dir = self._resolve_plots_dir(metrics_list, output_dir, create_subfolder)
        is_multi = len(metrics_list) > 1
        first_run_name = metrics_list[0].experiment_name or "0"
        _, suffix = self._resolve_run_naming(is_multi, first_run_name)
        dest_dir = self._resolve_dest_dir(
            plots_dir=plots_dir,
            is_multi=is_multi,
            total_runs=len(metrics_list),
            create_subfolder=create_subfolder,
        )

        # Persistent color assignment per model
        model_colors = {
            (m.experiment_name or f"run_{i}"): self.colourblind_palette[
                i % len(self.colourblind_palette)
            ]
            for i, m in enumerate(metrics_list)
        }

        saved_plots: list[Path] = []

        for split_name in ("val", "test"):
            # 1. Consolidated Collection Plot (Horizontal subplots per metric, sorted individually)
            coll_path = self._render_split_collection(
                split_name=split_name,
                targets=targets,
                metrics_list=metrics_list,
                model_colors=model_colors,
                dest_dir=dest_dir,
                suffix=suffix,
                plt=plt,
            )
            if coll_path is not None:
                saved_plots.append(coll_path)

            # 2. Comprehensive Multi-Metric Plot (Vertical grouped columns per model)
            if len(targets) > 1:
                comp_data = self._collect_composite_data(metrics_list, targets, split_name)
                if comp_data:
                    comp_path = self._render_composite_split_chart(
                        split_name=split_name,
                        targets=targets,
                        comp_data=comp_data,
                        dest_dir=dest_dir,
                        suffix=suffix,
                        plt=plt,
                    )
                    saved_plots.append(comp_path)

        return saved_plots

    @staticmethod
    def _calculate_grid_dims(n_plots: int) -> tuple[int, int]:
        """Calculate dynamic grid dimensions based on subplot count."""
        match n_plots:
            case 1:
                return 1, 1
            case 2:
                return 2, 1
            case 3:
                return 3, 1
            case 4:
                return 2, 2
            case _:
                return 3, math.ceil(n_plots / 3)

    @classmethod
    def _collect_metric_scores(
        cls,
        metrics_list: list[ParsedMetrics],
        var_name: str,
        split_name: str,
    ) -> list[tuple[str, float]]:
        """Extract and sort final scores descending for one metric in a split."""
        scores: list[tuple[str, float]] = []

        for run_idx, m in enumerate(metrics_list):
            if var_name not in m:
                continue

            df = m.fetch(var_name)
            splits = [str(s) for s in df["split"].unique()]
            matched = next((s for s in splits if split_name in s.lower()), None)
            if matched is None:
                continue

            split_df = df[df["split"] == matched]
            if split_df.empty:
                continue

            final_val = float(split_df["value"].iloc[-1])
            run_label = m.experiment_name or f"run_{run_idx}"
            scores.append((run_label, final_val))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    @classmethod
    def _collect_composite_data(
        cls,
        metrics_list: list[ParsedMetrics],
        targets: list[str],
        split_name: str,
    ) -> list[tuple[str, dict[str, float]]]:
        """Collect all target metric values for one split per model."""
        comp_data: list[tuple[str, dict[str, float]]] = []

        for run_idx, m in enumerate(metrics_list):
            run_label = m.experiment_name or f"run_{run_idx}"
            model_metrics: dict[str, float] = {}

            for var_name in targets:
                if var_name not in m:
                    continue

                df = m.fetch(var_name)
                splits = [str(s) for s in df["split"].unique()]
                matched = next((s for s in splits if split_name in s.lower()), None)
                if matched is None:
                    continue

                split_df = df[df["split"] == matched]
                if not split_df.empty:
                    model_metrics[var_name] = float(split_df["value"].iloc[-1])

            if model_metrics:
                comp_data.append((run_label, model_metrics))

        return comp_data

    def _resolve_axis_limits(self, values: list[float]) -> tuple[float, float]:
        """Calculate dynamic zoom or zero-anchored metric axis bounds."""
        if not values:
            return 0.0, 1.0

        min_v, max_v = min(values), max(values)
        span = max_v - min_v

        if self.zero_baseline or span == 0:
            low = 0.0 if min_v >= 0 else min_v * 1.15
            high = max_v * 1.18 if max_v > 0 else 1.0
            return low, high

        padding = max(span * 0.25, 0.015)
        low = max(0.0, min_v - padding) if min_v >= 0 else min_v - padding
        high = max_v + padding * 1.5
        return low, high

    @staticmethod
    def _annotate_bars(
        ax: Any,
        bars: Any,
        values: Sequence[float],
        offset: float,
        fontsize: float = 8.5,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
    ) -> None:
        """Annotate horizontal bars or vertical columns with formatted metric values."""
        try:
            for bar, val in zip(bars, values, strict=True):
                if np.isnan(val):
                    continue
                if orientation == "horizontal":
                    ax.text(
                        val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}",
                        va="center",
                        ha="left",
                        fontsize=fontsize,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + offset,
                        f"{val:.4f}",
                        va="bottom",
                        ha="center",
                        fontsize=fontsize,
                        fontweight="bold",
                    )
        except ValueError as exc:
            raise ValueError(
                f"Bar count ({len(bars)}) does not match value count ({len(values)})."
            ) from exc

    def _render_split_collection(
        self,
        split_name: str,
        targets: list[str],
        metrics_list: list[ParsedMetrics],
        model_colors: dict[str, str],
        dest_dir: Path,
        suffix: str,
        plt: Any,
    ) -> Path | None:
        """Render a collection figure with horizontal metric subplots for one split."""
        valid_metric_data: list[tuple[str, list[tuple[str, float]]]] = []
        for var_name in targets:
            scores = self._collect_metric_scores(metrics_list, var_name, split_name)
            if scores:
                valid_metric_data.append((var_name, scores))

        if not valid_metric_data:
            return None

        n_plots = len(valid_metric_data)
        n_cols, n_rows = self._calculate_grid_dims(n_plots)
        max_models = max(len(s) for _, s in valid_metric_data)

        fig_width = max(6.5, 4.8 * n_cols)
        fig_height = max(3.8, (max_models * 0.45 + 1.2) * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        flat_axes = axes.flatten()

        for idx, (var_name, scores) in enumerate(valid_metric_data):
            ax = flat_axes[idx]
            labels = [item[0] for item in scores]
            values = [item[1] for item in scores]
            colors = [model_colors.get(lbl, self.colourblind_palette[0]) for lbl in labels]

            x_left, x_right = self._resolve_axis_limits(values)
            span = x_right - x_left
            indices = np.arange(len(scores))

            bars = ax.barh(
                indices,
                [v - x_left for v in values],
                left=x_left,
                height=0.52,
                color=colors,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )

            self._annotate_bars(
                ax,
                bars,
                values,
                offset=span * 0.02,
                fontsize=8.5,
                orientation="horizontal",
            )

            ax.set_yticks(indices)
            ax.set_yticklabels(labels, fontsize=9.5)
            ax.invert_yaxis()
            ax.set_xlim(x_left, x_right)
            ax.set_title(self._format_label(var_name), fontsize=11, pad=8)
            ax.set_xlabel("Score", fontsize=9.5)

        for empty_idx in range(n_plots, len(flat_axes)):
            flat_axes[empty_idx].set_visible(False)

        fig.suptitle(split_name.capitalize(), fontsize=13, fontweight="bold", y=1.01)

        out_path = dest_dir / f"BarChart_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _render_composite_split_chart(
        self,
        split_name: str,
        targets: list[str],
        comp_data: list[tuple[str, dict[str, float]]],
        dest_dir: Path,
        suffix: str,
        plt: Any,
    ) -> Path:
        """Render comprehensive vertical grouped column chart comparing all metrics per model."""
        num_models = len(comp_data)
        num_metrics = len(targets)
        labels = [item[0] for item in comp_data]

        all_values = [v for _, m in comp_data for v in m.values()]
        y_bottom, y_top = self._resolve_axis_limits(all_values)
        span = y_top - y_bottom

        fig_width = max(7.5, num_models * (num_metrics * 0.45 + 0.8) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, 5.2))
        ax.set_axisbelow(True)

        group_indices = np.arange(num_models)
        bar_width = min(0.26, 0.75 / num_metrics)
        offsets = np.linspace(
            -(num_metrics - 1) / 2 * bar_width,
            (num_metrics - 1) / 2 * bar_width,
            num_metrics,
        )

        offset_text = span * 0.015
        for m_idx, var_name in enumerate(targets):
            color = self.colourblind_palette[m_idx % len(self.colourblind_palette)]
            vals = [m.get(var_name, np.nan) for _, m in comp_data]

            bars = ax.bar(
                group_indices + offsets[m_idx],
                [v - y_bottom if not np.isnan(v) else np.nan for v in vals],
                bottom=y_bottom,
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=self._format_label(var_name),
                zorder=2,
            )

            self._annotate_bars(
                ax,
                bars,
                vals,
                offset=offset_text,
                fontsize=7.5,
                orientation="vertical",
            )

        ax.set_xticks(group_indices)
        ax.set_xticklabels(
            labels,
            fontsize=9.5,
        )

        ax.set_ylim(y_bottom, y_top)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xlabel("Model Version", fontsize=10)

        fig.suptitle(split_name.capitalize(), fontsize=12, fontweight="bold", y=0.98)
        ax.legend(
            title="Metrics",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0.0,
            frameon=True,
        )

        out_path = dest_dir / f"BarChart_comprehensive_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
