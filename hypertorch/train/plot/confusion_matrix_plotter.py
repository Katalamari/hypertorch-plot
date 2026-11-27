from collections.abc import Sequence
import math
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np

from hypertorch.train.plot.plotter import Plotter
from hypertorch.types.parsed_metrics import ParsedMetrics


class ConfusionMatrixPlotter(Plotter):
    """Renders 2x2 binary or CxC multiclass confusion matrix heatmaps from logged counts."""

    BINARY_METRICS: tuple[str, ...] = ("tp", "fp", "tn", "fn")
    CELL_PATTERN: re.Pattern[str] = re.compile(r"^cm_(\d+)_(\d+)$")
    PRIMARY_SPLIT: str = "test"
    FALLBACK_SPLIT: str = "val"

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """
        Render and save confusion matrices for the test split, falling back to val.
        ...
        """
        plt = self._init_charting()
        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)

        modes: list[tuple[Literal["binary", "multiclass"], int]] = []
        for m in metrics_list:
            mode, num_classes = self._detect_matrix_mode(m)
            modes.append((mode, num_classes))

            if metric_names is not None:
                requested = set(metric_names)
                required = self._required_keys(mode, num_classes)
                if not required.issubset(requested):
                    raise ValueError(
                        f"metric_names must include all required {mode} confusion matrix "
                        f"keys to render a confusion matrix."
                    )

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

        split_name, valid_matrices = self._resolve_split_matrices(metrics_list, modes)
        if not valid_matrices:
            return []

        out_path = self._render_split_matrices(
            split_name=split_name,
            valid_matrices=valid_matrices,
            dest_dir=dest_dir,
            suffix=suffix,
            plt=plt,
        )
        return [out_path]

    @classmethod
    def _resolve_split_matrices(
        cls,
        metrics_list: list[ParsedMetrics],
        modes: list[tuple[Literal["binary", "multiclass"], int]],
    ) -> tuple[str, list[tuple[str, np.ndarray]]]:
        """Collect confusion matrices from the primary split, falling back to val."""
        test_matrices = cls._collect_split_matrices(metrics_list, modes, cls.PRIMARY_SPLIT)
        if test_matrices:
            return cls.PRIMARY_SPLIT, test_matrices

        val_matrices = cls._collect_split_matrices(metrics_list, modes, cls.FALLBACK_SPLIT)
        return cls.FALLBACK_SPLIT, val_matrices

    @classmethod
    def _collect_split_matrices(
        cls,
        metrics_list: list[ParsedMetrics],
        modes: list[tuple[Literal["binary", "multiclass"], int]],
        split_name: str,
    ) -> list[tuple[str, np.ndarray]]:
        """Extract confusion matrices for all runs containing the given split."""
        valid_matrices: list[tuple[str, np.ndarray]] = []
        for idx, (m, (mode, num_classes)) in enumerate(zip(metrics_list, modes, strict=True)):
            matrix = cls._extract_split_matrix(m, split_name, mode, num_classes)
            if matrix is not None:
                label = m.experiment_name or f"run_{idx}"
                valid_matrices.append((label, matrix))
        return valid_matrices

    @classmethod
    def _required_keys(
        cls,
        mode: Literal["binary", "multiclass"],
        num_classes: int,
    ) -> set[str]:
        """Return the set of metric keys required for the given mode and class count."""
        if mode == "binary":
            return set(cls.BINARY_METRICS)
        return {f"cm_{r}_{c}" for r in range(num_classes) for c in range(num_classes)}

    @classmethod
    def _detect_matrix_mode(
        cls,
        parsed: ParsedMetrics,
    ) -> tuple[Literal["binary", "multiclass"], int]:
        """Detect whether ParsedMetrics contains binary counts or multiclass cm_i_j cells."""
        names = set(parsed.names())
        if set(cls.BINARY_METRICS).issubset(names):
            return "binary", 2

        cell_coords = [
            (int(match.group(1)), int(match.group(2)))
            for name in names
            if (match := cls.CELL_PATTERN.match(name)) is not None
        ]
        if cell_coords:
            num_classes = max(max(r, c) for r, c in cell_coords) + 1
            expected = cls._required_keys("multiclass", num_classes)
            if num_classes >= 2 and expected.issubset(names):
                return "multiclass", num_classes

        run_id = parsed.experiment_name or "unknown"
        raise ValueError(
            f"ParsedMetrics '{run_id}' does not contain complete binary "
            "(tp, fp, tn, fn) or multiclass (cm_i_j) confusion matrix metrics."
        )

    @staticmethod
    def _calculate_grid_dims(n_plots: int) -> tuple[int, int]:
        """Calculate dynamic grid dimensions based on model count."""
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

    @staticmethod
    def _fetch_last_split_value(
        parsed: ParsedMetrics,
        key: str,
        split_name: str,
    ) -> float | None:
        """Fetch the final logged value of a metric for a given split."""
        df = parsed.fetch(key)
        splits = [str(s) for s in df["split"].unique()]
        matched = next((s for s in splits if split_name in s.lower()), None)
        if matched is None:
            return None

        split_df = df[df["split"] == matched]
        if split_df.empty:
            return None

        return float(split_df["value"].iloc[-1])

    @classmethod
    def _extract_split_matrix(
        cls,
        parsed: ParsedMetrics,
        split_name: str,
        mode: Literal["binary", "multiclass"],
        num_classes: int,
    ) -> np.ndarray | None:
        """
        Extract final confusion matrix for a given split.

        Returns:
            matrix: 2x2 ([[TN, FP], [FN, TP]]) or CxC numpy float array, or None if split absent.
        """
        if mode == "binary":
            counts: dict[str, float] = {}
            for key in cls.BINARY_METRICS:
                val = cls._fetch_last_split_value(parsed, key, split_name)
                if val is None:
                    return None
                counts[key] = val

            return np.array(
                [
                    [counts["tn"], counts["fp"]],
                    [counts["fn"], counts["tp"]],
                ],
                dtype=float,
            )

        matrix = np.zeros((num_classes, num_classes), dtype=float)
        for row in range(num_classes):
            for col in range(num_classes):
                val = cls._fetch_last_split_value(parsed, f"cm_{row}_{col}", split_name)
                if val is None:
                    return None
                matrix[row, col] = val
        return matrix

    def _render_split_matrices(
        self,
        split_name: str,
        valid_matrices: list[tuple[str, np.ndarray]],
        dest_dir: Path,
        suffix: str,
        plt: Any,
    ) -> Path:
        """Render a grid of confusion matrix heatmaps for one evaluation split."""
        n_plots = len(valid_matrices)
        n_cols, n_rows = self._calculate_grid_dims(n_plots)
        max_classes = max(matrix.shape[0] for _, matrix in valid_matrices)

        per_plot_w = max(4.6, 0.9 * max_classes)
        per_plot_h = max(4.2, 0.85 * max_classes)
        fig_width = max(5.2, per_plot_w * n_cols)
        fig_height = max(4.4, per_plot_h * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        flat_axes = axes.flatten()

        for idx, (run_label, matrix) in enumerate(valid_matrices):
            ax = flat_axes[idx]
            num_classes = matrix.shape[0]
            row_sums = matrix.sum(axis=1, keepdims=True)
            norm_matrix = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix),
                where=row_sums != 0,
            )

            ax.imshow(norm_matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)

            cell_fontsize = 10.0 if num_classes <= 3 else (8.5 if num_classes <= 6 else 7.5)
            for row in range(num_classes):
                for col in range(num_classes):
                    count_val: int = round(matrix[row, col])
                    pct_val: float = norm_matrix[row, col] * 100.0
                    text_color: str = "white" if norm_matrix[row, col] > 0.55 else "black"
                    ax.text(
                        col,
                        row,
                        f"{count_val}\n({pct_val:.1f}%)",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=cell_fontsize,
                        fontweight="bold",
                    )

            ticks = list(range(num_classes))
            if num_classes == 2:
                x_labels = ["Pred 0", "Pred 1"]
                y_labels = ["Actual 0", "Actual 1"]
            else:
                x_labels = [str(i) for i in ticks]
                y_labels = [str(i) for i in ticks]

            ax.set_xticks(ticks)
            ax.set_xticklabels(x_labels, fontsize=9.5)
            ax.set_yticks(ticks)
            ax.set_yticklabels(y_labels, fontsize=9.5)
            ax.set_xlabel("Predicted Label", fontsize=9.5)
            ax.set_ylabel("True Label", fontsize=9.5)
            ax.set_title(run_label, fontsize=11, pad=8)

        for empty_idx in range(n_plots, len(flat_axes)):
            flat_axes[empty_idx].set_visible(False)

        fig.suptitle(split_name.capitalize(), fontsize=13, fontweight="bold", y=1.01)

        out_path = dest_dir / f"ConfusionMatrix_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
