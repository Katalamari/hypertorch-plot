from pathlib import Path

from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from hypertorch.data import (
    CoraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
)
from hypertorch.node_classification import GCNClassifier
from hypertorch.train import MultiModelTrainer
from hypertorch.train.plot import LogParser
from hypertorch.train.plot.confusionmatrixplotter import ConfusionMatrixPlotter
from hypertorch.types import ModelConfig


class MulticlassCMCell(MulticlassConfusionMatrix):
    """Logs a single scalar cell (row, col) of a multiclass confusion matrix."""

    def __init__(self, num_classes: int, row: int, col: int) -> None:
        super().__init__(num_classes=num_classes)
        self.row = row
        self.col = col

    def compute(self) -> Tensor:
        return super().compute()[self.row, self.col].float()


def build_multiclass_metrics(num_classes: int) -> MetricCollection:
    """Create a MetricCollection with standard metrics and individual cm_{i}_{j} cells."""
    metric_dict: dict[str, object] = {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "auc": MulticlassAUROC(num_classes=num_classes, average="macro"),
    }
    for i in range(num_classes):
        for j in range(num_classes):
            metric_dict[f"cm_{i}_{j}"] = MulticlassCMCell(
                num_classes=num_classes,
                row=i,
                col=j,
            )
    return MetricCollection(metric_dict)  # type: ignore[arg-type]


def main() -> None:
    num_workers = 4
    num_features = 32

    print("Loading and preparing multiclass node-classification dataset (cora)...")
    dataset = CoraDataset(sampling_strategy="node", task="node-classification")
    num_classes = int(dataset.hdata.y.unique().numel())
    print(f"Detected {num_classes} unique class(es).")
    metrics = build_multiclass_metrics(num_classes=num_classes)

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
            num_features=num_features,
            num_nodes=dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

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
            "in_channels": num_features,
            "hidden_channels": 32,
            "out_channels": num_classes,
            "num_layers": 2,
            "drop_rate": 0.1,
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
        max_epochs=10,
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

    print("Generating multiclass confusion matrix plots...")
    plotter = ConfusionMatrixPlotter()
    saved_plots = plotter.plot(parsed_runs)

    print(f"\nGenerated {len(saved_plots)} plot(s):")
    for plot_path in saved_plots:
        print(f" -> {Path(plot_path).resolve()}")


if __name__ == "__main__":
    main()