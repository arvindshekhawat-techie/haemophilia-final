"""Evaluation metrics for the elbow form classifier."""

from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

from src.features.elbow_features import INPUT_SIZE
from src.models.elbow_lstm import ElbowLSTM, load_checkpoint
from src.training.dataset import ElbowDataset
from src.training.splitting import grouped_train_validation_split


def evaluate(processed_dir: str | Path, model_path: str | Path) -> dict:
    dataset = ElbowDataset(processed_dir)
    labels = [int(dataset[index][1].item()) for index in range(len(dataset))]
    _, validation_indices = grouped_train_validation_split(
        dataset.files, labels, dataset.groups
    )
    loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=8,
    )
    model = ElbowLSTM(input_size=INPUT_SIZE)
    load_checkpoint(model, str(model_path))
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in loader:
            logits = model(features).squeeze(1)
            predictions.extend((logits >= 0.0).long().tolist())
            labels.extend(batch_labels.tolist())
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "validation_samples": len(labels),
        "validation_groups": len({dataset.groups[index] for index in validation_indices}),
        "subject_disjoint": bool(dataset.subject_ids) and all(dataset.subject_ids),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    processed_dir = (
        root / "processed_rebuilt_subjects"
        if (root / "processed_rebuilt_subjects").is_dir()
        else root / "processed_rebuilt_v2"
    )
    print(
        evaluate(
            processed_dir,
            root / "models" / "elbow_lstm_candidate_v2.pth",
        )
    )
