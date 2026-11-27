from collections.abc import Sequence
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any

import numpy as np

from hypertorch.train.plot.plotter import Plotter
from hypertorch.types.parsed_metrics import ParsedMetrics


class CurveGraphPlotter(Plotter):
    THRESHOLD_PATTERN: re.Pattern[str] = re.compile(r"^(tp|fp|tn|fn)_t(\d+)$")
    CALIBRATION_PATTERN: re.Pattern[str] = re.compile(r"^cal_(conf|acc)_(\d+)$")
    PRIMARY_SPLIT: str = "test"
    FALLBACK_SPLIT: str = "val"

    def __init__(
        self,
        roc: bool = False,
        toc: bool = False,
        det: bool = False,
        pr: bool = False,
        calibration: bool = False,
    ) -> None:
        """
        Renders evaluation curves from logged metrics.
        Currently supports ROC, TOC, DET, Precision-Recall, Calibratio.

        Args:
            roc: Whether to render Receiver Operating Characteristic (ROC) curves.
            toc: Whether to render Total Operating Characteristic (TOC) curves.
            det: Whether to render Detection Error Tradeoff (DET) curves.
            pr: Whether to render Precision-Recall (PR) curves.
            calibration: Whether to render Calibration (Reliability Diagram) curves.

        Raises:
            ValueError: If all curve flags are False.
        """
        if not any((roc, toc, det, pr, calibration)):
            raise ValueError(
                "At least one curve type (roc, toc, det, pr, calibration) must be enabled."
            )

        self.roc: bool = roc
        self.toc: bool = toc
        self.det: bool = det
        self.pr: bool = pr
        self.calibration: bool = calibration

    @property
    def _needs_threshold_metrics(self) -> bool:
        """Return True if any threshold-based curve (ROC, TOC, DET, PR) is enabled."""
        return any((self.roc, self.toc, self.det, self.pr))

    def _validate_curve_requirements(
        self,
        metrics_list: list[ParsedMetrics],
        metric_names: list[str] | None,
    ) -> tuple[list[int], list[int]]:
        """Validate required threshold and calibration keys across all runs."""
        threshold_counts: list[int] = []
        calibration_bins: list[int] = []
        requested = set(metric_names) if metric_names is not None else None

        for m in metrics_list:
            if self._needs_threshold_metrics:
                n_thresh = self._detect_threshold_count(m)
                threshold_counts.append(n_thresh)
                if requested is not None:
                    req = {f"{k}_t{i}" for i in range(n_thresh) for k in ("tp", "fp", "tn", "fn")}
                    if not req.issubset(requested):
                        raise ValueError(
                            "metric_names must include all required thresholded "
                            "(tp_t*, fp_t*, tn_t*, fn_t*) keys."
                        )

            if self.calibration:
                n_bins = self._detect_calibration_bins(m)
                calibration_bins.append(n_bins)
                if requested is not None:
                    req = {f"cal_{k}_{b}" for b in range(n_bins) for k in ("conf", "acc")}
                    if not req.issubset(requested):
                        raise ValueError(
                            "metric_names must include all required calibration "
                            "(cal_conf_*, cal_acc_*) keys."
                        )

        return threshold_counts, calibration_bins

    def plot(
        self,
        metrics: ParsedMetrics | Sequence[ParsedMetrics],
        metric_names: list[str] | None = None,
        output_dir: str | Path | None = None,
        create_subfolder: bool = True,
    ) -> list[Path]:
        """
        Render and save enabled evaluation curves for test splits.
        Fallback on latest val otherwise.
        """
        plt = self._init_plotting()
        metrics_list = self._normalize_metrics(metrics)
        self._validate_metrics_list(metrics_list)

        threshold_counts, calibration_bins = self._validate_curve_requirements(
            metrics_list, metric_names
        )

        plots_dir = self._resolve_plots_dir(metrics_list, output_dir, create_subfolder)
        is_multi = len(metrics_list) > 1
        first_run_name = metrics_list[0].experiment_name or "0"
        title_prefix, suffix = self._resolve_run_naming(is_multi, first_run_name)
        dest_dir = self._resolve_dest_dir(
            plots_dir=plots_dir,
            is_multi=is_multi,
            total_runs=len(metrics_list),
            create_subfolder=create_subfolder,
        )

        saved_plots: list[Path] = []

        if self._needs_threshold_metrics:
            thresh_split, thresh_data = self._resolve_threshold_data(metrics_list, threshold_counts)
            if thresh_data:
                threshold_renderers = (
                    (self.roc, self._render_roc),
                    (self.pr, self._render_pr),
                    (self.det, self._render_det),
                    (self.toc, self._render_toc),
                )
                saved_plots.extend(
                    render_fn(thresh_data, thresh_split, title_prefix, suffix, dest_dir, plt)
                    for enabled, render_fn in threshold_renderers
                    if enabled
                )

        if self.calibration:
            cal_split, cal_data = self._resolve_calibration_data(metrics_list, calibration_bins)
            if cal_data:
                saved_plots.append(
                    self._render_calibration(
                        cal_data, cal_split, title_prefix, suffix, dest_dir, plt
                    )
                )

        return saved_plots

    @classmethod
    def _resolve_threshold_data(
        cls,
        metrics_list: list[ParsedMetrics],
        threshold_counts: list[int],
    ) -> tuple[str, list[tuple[str, dict[str, np.ndarray]]]]:
        """Collect threshold metrics from the primary (test) split, falling back to val."""
        test_data = cls._collect_threshold_data(metrics_list, threshold_counts, cls.PRIMARY_SPLIT)
        if test_data:
            return cls.PRIMARY_SPLIT, test_data

        val_data = cls._collect_threshold_data(metrics_list, threshold_counts, cls.FALLBACK_SPLIT)
        return cls.FALLBACK_SPLIT, val_data

    @classmethod
    def _resolve_calibration_data(
        cls,
        metrics_list: list[ParsedMetrics],
        calibration_bins: list[int],
    ) -> tuple[str, list[tuple[str, np.ndarray, np.ndarray, float | None]]]:
        """Collect calibration metrics from the primary (test) split, falling back to val."""
        test_data = cls._collect_calibration_data(metrics_list, calibration_bins, cls.PRIMARY_SPLIT)
        if test_data:
            return cls.PRIMARY_SPLIT, test_data

        val_data = cls._collect_calibration_data(metrics_list, calibration_bins, cls.FALLBACK_SPLIT)
        return cls.FALLBACK_SPLIT, val_data

    @classmethod
    def _detect_threshold_count(cls, parsed: ParsedMetrics) -> int:
        """Detect and validate the number of threshold steps logged in ParsedMetrics."""
        names = set(parsed.names())
        indices = [
            int(match.group(2))
            for name in names
            if (match := cls.THRESHOLD_PATTERN.match(name)) is not None
        ]
        if indices:
            n_thresh = max(indices) + 1
            expected = {f"{k}_t{i}" for i in range(n_thresh) for k in ("tp", "fp", "tn", "fn")}
            if n_thresh >= 2 and expected.issubset(names):
                return n_thresh

        run_id = parsed.experiment_name or "unknown"
        raise ValueError(
            f"ParsedMetrics '{run_id}' does not contain complete thresholded confusion "
            "metrics (tp_t{k}, fp_t{k}, tn_t{k}, fn_t{k} for at least 2 thresholds)."
        )

    @classmethod
    def _detect_calibration_bins(cls, parsed: ParsedMetrics) -> int:
        """Detect and validate the number of calibration bins logged in ParsedMetrics."""
        names = set(parsed.names())
        indices = [
            int(match.group(2))
            for name in names
            if (match := cls.CALIBRATION_PATTERN.match(name)) is not None
        ]
        if indices:
            n_bins = max(indices) + 1
            expected = {f"cal_{k}_{b}" for b in range(n_bins) for k in ("conf", "acc")}
            if n_bins >= 2 and expected.issubset(names):
                return n_bins

        run_id = parsed.experiment_name or "unknown"
        raise ValueError(
            f"ParsedMetrics '{run_id}' does not contain complete calibration metrics "
            "(cal_conf_{b}, cal_acc_{b} for at least 2 bins)."
        )

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
    def _collect_threshold_data(
        cls,
        metrics_list: list[ParsedMetrics],
        threshold_counts: list[int],
        split_name: str,
    ) -> list[tuple[str, dict[str, np.ndarray]]]:
        """Extract thresholded TP/FP/TN/FN arrays for each run containing the split."""
        collected: list[tuple[str, dict[str, np.ndarray]]] = []
        for idx, (m, n_thresh) in enumerate(zip(metrics_list, threshold_counts, strict=True)):
            arrays: dict[str, np.ndarray] = {
                k: np.zeros(n_thresh, dtype=float) for k in ("tp", "fp", "tn", "fn")
            }
            complete = True
            for i in range(n_thresh):
                for k in ("tp", "fp", "tn", "fn"):
                    val = cls._fetch_last_split_value(m, f"{k}_t{i}", split_name)
                    if val is None:
                        complete = False
                        break
                    arrays[k][i] = val
                if not complete:
                    break

            if complete:
                label = m.experiment_name or f"run_{idx}"
                collected.append((label, arrays))

        return collected

    @classmethod
    def _collect_calibration_data(
        cls,
        metrics_list: list[ParsedMetrics],
        calibration_bins: list[int],
        split_name: str,
    ) -> list[tuple[str, np.ndarray, np.ndarray, float | None]]:
        """Extract calibration (confidence, accuracy, optional ECE) per run for the split."""
        collected: list[tuple[str, np.ndarray, np.ndarray, float | None]] = []
        for idx, (m, n_bins) in enumerate(zip(metrics_list, calibration_bins, strict=True)):
            confs = np.zeros(n_bins, dtype=float)
            accs = np.zeros(n_bins, dtype=float)
            complete = True
            for b in range(n_bins):
                conf_val = cls._fetch_last_split_value(m, f"cal_conf_{b}", split_name)
                acc_val = cls._fetch_last_split_value(m, f"cal_acc_{b}", split_name)
                if conf_val is None or acc_val is None:
                    complete = False
                    break
                confs[b] = conf_val
                accs[b] = acc_val

            if complete:
                ece = cls._fetch_last_split_value(m, "ece", split_name) if "ece" in m else None
                label = m.experiment_name or f"run_{idx}"
                collected.append((label, confs, accs, ece))

        return collected

    @staticmethod
    def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
        """Compute area under curve via the trapezoidal rule (NumPy 1.x & 2.x safe)."""
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))

    def _render_roc(
        self,
        data: list[tuple[str, dict[str, np.ndarray]]],
        split_name: str,
        title_prefix: str,
        suffix: str,
        dest_dir: Path,
        plt: Any,
    ) -> Path:
        """Render and save the ROC curve."""
        fig, ax = plt.subplots(figsize=(8.0, 5.5))

        for idx, (run_label, arr) in enumerate(data):
            tp, fp, tn, fn = arr["tp"], arr["fp"], arr["tn"], arr["fn"]
            fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) != 0)
            tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)

            order = np.argsort(fpr)
            fpr_sorted, tpr_sorted = fpr[order], tpr[order]
            auc_val = self._trapezoid_area(tpr_sorted, fpr_sorted)

            color = self.colourblind_palette[idx % len(self.colourblind_palette)]
            marker = self.marker_succession[idx % len(self.marker_succession)]
            ax.plot(
                fpr_sorted,
                tpr_sorted,
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.8,
                label=f"{run_label} (AUC = {auc_val:.4f})",
            )

        ax.plot(
            [0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.2, label="Random"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title(f"{title_prefix} — ROC Curve ({split_name.capitalize()})", fontsize=12, pad=10)
        ax.legend(loc="lower right", framealpha=0.85)

        out_path = dest_dir / f"CurveGraph_ROC_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _render_pr(
        self,
        data: list[tuple[str, dict[str, np.ndarray]]],
        split_name: str,
        title_prefix: str,
        suffix: str,
        dest_dir: Path,
        plt: Any,
    ) -> Path:
        """Render and save the Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8.0, 5.5))

        for idx, (run_label, arr) in enumerate(data):
            tp, fp, fn = arr["tp"], arr["fp"], arr["fn"]
            recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
            precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) != 0)

            order = np.argsort(recall)
            rec_sorted, prec_sorted = recall[order], precision[order]
            ap_val = self._trapezoid_area(prec_sorted, rec_sorted)

            color = self.colourblind_palette[idx % len(self.colourblind_palette)]
            marker = self.marker_succession[idx % len(self.marker_succession)]
            ax.plot(
                rec_sorted,
                prec_sorted,
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.8,
                label=f"{run_label} (AP = {ap_val:.4f})",
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(
            f"{title_prefix} — Precision-Recall Curve ({split_name.capitalize()})",
            fontsize=12,
            pad=10,
        )
        ax.legend(loc="lower left", framealpha=0.85)

        out_path = dest_dir / f"CurveGraph_PR_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _render_det(
        self,
        data: list[tuple[str, dict[str, np.ndarray]]],
        split_name: str,
        title_prefix: str,
        suffix: str,
        dest_dir: Path,
        plt: Any,
    ) -> Path:
        """Render and save the Detection Error Tradeoff (DET) curve on normal-deviate axes."""
        norm = NormalDist()
        fig, ax = plt.subplots(figsize=(8.0, 5.5))

        for idx, (run_label, arr) in enumerate(data):
            tp, fp, tn, fn = arr["tp"], arr["fp"], arr["tn"], arr["fn"]
            fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) != 0)
            fnr = np.divide(fn, tp + fn, out=np.zeros_like(fn), where=(tp + fn) != 0)

            fpr_clipped = np.clip(fpr, 1e-4, 1.0 - 1e-4)
            fnr_clipped = np.clip(fnr, 1e-4, 1.0 - 1e-4)

            x_det = np.array([norm.inv_cdf(float(v)) for v in fpr_clipped])
            y_det = np.array([norm.inv_cdf(float(v)) for v in fnr_clipped])

            order = np.argsort(x_det)
            color = self.colourblind_palette[idx % len(self.colourblind_palette)]
            marker = self.marker_succession[idx % len(self.marker_succession)]
            ax.plot(
                x_det[order],
                y_det[order],
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.8,
                label=run_label,
            )

        tick_probs = [0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
        tick_locs = [norm.inv_cdf(p) for p in tick_probs]
        tick_labels = [f"{p * 100:g}%" for p in tick_probs]

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("False Negative Rate (FNR)")
        ax.set_title(f"{title_prefix} — DET Curve ({split_name.capitalize()})", fontsize=12, pad=10)
        ax.legend(loc="upper right", framealpha=0.85)

        out_path = dest_dir / f"CurveGraph_DET_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _render_toc(
        self,
        data: list[tuple[str, dict[str, np.ndarray]]],
        split_name: str,
        title_prefix: str,
        suffix: str,
        dest_dir: Path,
        plt: Any,
    ) -> Path:
        """Render and save the Total Operating Characteristic (TOC) curve."""
        fig, ax = plt.subplots(figsize=(8.0, 5.5))

        ref_arr = data[0][1]
        positives = float((ref_arr["tp"] + ref_arr["fn"])[0])
        negatives = float((ref_arr["fp"] + ref_arr["tn"])[0])
        total = positives + negatives

        # TOC parallelogram bounds and uniform baseline
        ax.plot(
            [0.0, positives, total],
            [0.0, positives, positives],
            linestyle=":",
            color="#444444",
            linewidth=1.2,
            label="Maximum",
        )
        ax.plot(
            [0.0, negatives, total],
            [0.0, 0.0, positives],
            linestyle=":",
            color="#888888",
            linewidth=1.2,
            label="Minimum",
        )
        ax.plot(
            [0.0, total],
            [0.0, positives],
            linestyle="--",
            color="#666666",
            linewidth=1.2,
            label="Uniform",
        )

        for idx, (run_label, arr) in enumerate(data):
            hits = arr["tp"]
            hits_plus_fa = arr["tp"] + arr["fp"]

            order = np.argsort(hits_plus_fa)
            color = self.colourblind_palette[idx % len(self.colourblind_palette)]
            marker = self.marker_succession[idx % len(self.marker_succession)]
            ax.plot(
                hits_plus_fa[order],
                hits[order],
                color=color,
                marker=marker,
                markersize=4.0,
                linewidth=1.8,
                label=run_label,
            )

        ax.set_xlim(0.0, max(total, 1.0))
        ax.set_ylim(0.0, max(positives * 1.05, 1.0))
        ax.set_xlabel("Hits + False Alarms (TP + FP)")
        ax.set_ylabel("Hits (TP)")
        ax.set_title(f"{title_prefix} — TOC Curve ({split_name.capitalize()})", fontsize=12, pad=10)
        ax.legend(loc="lower right", framealpha=0.85)

        out_path = dest_dir / f"CurveGraph_TOC_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _render_calibration(
        self,
        data: list[tuple[str, np.ndarray, np.ndarray, float | None]],
        split_name: str,
        title_prefix: str,
        suffix: str,
        dest_dir: Path,
        plt: Any,
    ) -> Path:
        """Render and save the Calibration Curve (Reliability Diagram)."""
        fig, ax = plt.subplots(figsize=(8.0, 5.5))

        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            color="#666666",
            linewidth=1.2,
            label="Perfectly Calibrated",
        )

        for idx, (run_label, confs, accs, ece) in enumerate(data):
            order = np.argsort(confs)
            color = self.colourblind_palette[idx % len(self.colourblind_palette)]
            marker = self.marker_succession[idx % len(self.marker_succession)]
            legend_label = f"{run_label} (ECE = {ece:.4f})" if ece is not None else run_label
            ax.plot(
                confs[order],
                accs[order],
                color=color,
                marker=marker,
                markersize=5.0,
                linewidth=1.8,
                label=legend_label,
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title(
            f"{title_prefix} — Calibration Curve ({split_name.capitalize()})",
            fontsize=12,
            pad=10,
        )
        ax.legend(loc="upper left", framealpha=0.85)

        out_path = dest_dir / f"CurveGraph_Calibration_{split_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
