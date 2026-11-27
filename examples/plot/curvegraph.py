from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCalibrationError,
    MulticlassF1Score,
)

from hypertorch.data import (
    CoraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
)
from hypertorch.node_classification import GCNClassifier
from hypertorch.train import MultiModelTrainer
from hypertorch.train.plot import LogParser, CurveGraphPlotter
from hypertorch.types import ModelConfig


class MulticlassThresholdStatCell(Metric):
    """Logs micro-averaged OvR scalar (tp, fp, tn, or fn) at a fixed probability threshold."""

    full_state_update: bool | None = False
    count: Tensor

    def __init__(self, num_classes: int, threshold: float, stat: str) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.stat = stat  # "tp", "fp", "tn", or "fn"
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        probs = F.softmax(preds.detach().float(), dim=-1)
        one_hot = F.one_hot(target.detach().long(), num_classes=self.num_classes).bool()
        pred_pos = probs >= self.threshold

        if self.stat == "tp":
            val = (pred_pos & one_hot).sum()
        elif self.stat == "fp":
            val = (pred_pos & ~one_hot).sum()
        elif self.stat == "tn":
            val = (~pred_pos & ~one_hot).sum()
        else:
            val = (~pred_pos & one_hot).sum()

        self.count += val.float()

    def compute(self) -> Tensor:
        return self.count


class MulticlassCalibrationBinMetric(Metric):
    """Logs mean top-label confidence or empirical accuracy for one probability bin."""

    full_state_update: bool | None = False
    total_conf: Tensor
    total_correct: Tensor
    count: Tensor

    def __init__(self, bin_idx: int, num_bins: int, mode: str) -> None:
        super().__init__()
        self.bin_idx = bin_idx
        self.num_bins = num_bins
        self.mode = mode  # "conf" or "acc"
        self.add_state("total_conf", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        probs = F.softmax(preds.detach().float(), dim=-1)
        confs, pred_classes = probs.max(dim=-1)
        correct = (pred_classes == target.detach().long()).float()

        low = self.bin_idx / self.num_bins
        high = (self.bin_idx + 1) / self.num_bins
        if self.bin_idx == self.num_bins - 1:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs >= low) & (confs < high)

        if mask.any():
            self.total_conf += confs[mask].sum()
            self.total_correct += correct[mask].sum()
            self.count += mask.sum().float()

    def compute(self) -> Tensor:
        if self.count == 0:
            midpoint = (self.bin_idx + 0.5) / self.num_bins
            return torch.tensor(midpoint, device=self.count.device)
        if self.mode == "conf":
            return self.total_conf / self.count
        return self.total_correct / self.count


def build_multiclass_curve_metrics(
    num_classes: int,
    num_thresholds: int = 11,
    num_bins: int = 10,
) -> MetricCollection:
    """Create a MetricCollection with thresholded OvR stats and calibration bin scalars."""
    metric_dict: dict[str, Metric | MetricCollection] = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "auc": MulticlassAUROC(num_classes=num_classes, average="macro"),
        "ece": MulticlassCalibrationError(num_classes=num_classes, n_bins=num_bins),
    }

    thresholds = torch.linspace(0.0, 1.0, steps=num_thresholds).tolist()
    for t_idx, threshold in enumerate(thresholds):
        for stat in ("tp", "fp", "tn", "fn"):
            metric_dict[f"{stat}_t{t_idx}"] = MulticlassThresholdStatCell(
                num_classes=num_classes,
                threshold=float(threshold),
                stat=stat,
            )

    for b_idx in range(num_bins):
        metric_dict[f"cal_conf_{b_idx}"] = MulticlassCalibrationBinMetric(
            bin_idx=b_idx,
            num_bins=num_bins,
            mode="conf",
        )
        metric_dict[f"cal_acc_{b_idx}"] = MulticlassCalibrationBinMetric(
            bin_idx=b_idx,
            num_bins=num_bins,
            mode="acc",
        )

    return MetricCollection(metric_dict)


def main() -> None:
    num_workers = 4
    pe_features = 32

    print("Loading and preparing multiclass node-classification dataset (cora)...")
    dataset = CoraDataset(sampling_strategy="node", task="node-classification")
    num_classes = int(dataset.hdata.y.unique().numel())
    print(f"Detected {num_classes} unique class(es).")
    metrics = build_multiclass_curve_metrics(
        num_classes=num_classes,
        num_thresholds=11,
        num_bins=10,
    )

    # Split nodes into train, val, and test (70/10/20)
    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        node_space_setting="transductive",
        shuffle=True,
        seed=42,
    )

    print("Enriching node features...")
    train_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=pe_features,
            num_nodes=dataset.hdata.num_nodes,
        ),
        enrichment_mode="concatenate",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    in_channels = int(train_dataset.hdata.x.shape[1])
    print(f"Total node feature dimension: {in_channels}")

    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = GCNClassifier(
        classifier_config={
            "in_channels": in_channels,
            "hidden_channels": 64,
            "out_channels": num_classes,
            "num_layers": 2,
            "drop_rate": 0.5,
            "graph_reduction_strategy": "clique_expansion",
            "num_nodes": dataset.hdata.num_nodes,
        },
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="gcn",
            version="node-classification",
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting training and evaluation...")
    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=30,
        accelerator="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
        devices=1,
        test_devices=1,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            verbose=True,
        )
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("\nParsing experiment metrics...")
    parser = LogParser()
    parser.discover_from_latest_dir(num=1)
    parsed_runs = parser.parse_all()

    print("Generating curve plots (ROC, TOC, DET, PR, Calibration)...")
    plotter = CurveGraphPlotter(
        roc=True,
        toc=True,
        det=True,
        pr=True,
        calibration=True,
    )
    saved_plots = plotter.plot(parsed_runs)

    print(f"\nGenerated {len(saved_plots)} plot(s):")
    for plot_path in saved_plots:
        print(f" -> {Path(plot_path).resolve()}")


if __name__ == "__main__":
    main()
