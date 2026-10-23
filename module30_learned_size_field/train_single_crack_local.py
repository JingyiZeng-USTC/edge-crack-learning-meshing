"""Train the straight-only, segment-count-invariant local size-field model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import DirectNormalizedLogSizeMLP
from .single_crack_local_features import (
    SINGLE_CRACK_LOCAL_FEATURE_NAMES,
    SINGLE_CRACK_LOCAL_MODEL_FAMILY,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_split(root: Path, rows: list[dict], split: str):
    collected = ([], [], [], [])
    for row in rows:
        if row["split"] != split:
            continue
        with np.load(root / row["file"], allow_pickle=False) as data:
            collected[0].append(np.asarray(data["features"], dtype=np.float32))
            collected[1].append(np.asarray(data["target_normalized_log_h"], dtype=np.float32))
            collected[2].append(np.asarray(data["sample_weight"], dtype=np.float32))
            collected[3].append(np.asarray(data["target_h"], dtype=np.float32))
    if not collected[0]:
        raise ValueError(f"No {split} samples in {root}")
    return tuple(np.concatenate(values, axis=0) for values in collected)


def _evaluate(model, loader, device, target_h, h_min: float, h_max: float) -> dict:
    model.eval()
    loss_sum = 0.0
    count = 0
    predictions = []
    criterion = nn.SmoothL1Loss(reduction="none")
    with torch.no_grad():
        for features, target, weight in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            predicted = model(features)
            loss_sum += float((criterion(predicted, target) * weight).sum().item())
            count += len(target)
            predictions.append(predicted.cpu().numpy())
    normalized_raw = np.concatenate(predictions).astype(np.float64, copy=False)
    normalized = np.clip(normalized_raw, 0.0, 1.0)
    log_min = float(np.log(h_min))
    log_range = float(np.log(h_max) - np.log(h_min))
    predicted_h = np.exp(log_min + normalized * log_range)
    target_values = np.asarray(target_h, dtype=np.float64)
    refined = target_values <= float(np.sqrt(h_min * h_max))
    refined_relative = float(
        np.linalg.norm(predicted_h[refined] - target_values[refined])
        / max(np.linalg.norm(target_values[refined]), 1.0e-30)
    ) if np.any(refined) else float("nan")
    return {
        "loss": loss_sum / max(count, 1),
        "h_relative_l2": float(
            np.linalg.norm(predicted_h - target_values)
            / max(np.linalg.norm(target_values), 1.0e-30)
        ),
        "log_mae": float(np.mean(np.abs(np.log(predicted_h / target_values)))),
        "refined_h_relative_l2": refined_relative,
        "predicted_h_min": float(predicted_h.min()),
        "predicted_h_max": float(predicted_h.max()),
        "predicted_size_contrast": float(
            predicted_h.max() / max(predicted_h.min(), 1.0e-30)
        ),
        "raw_output_min": float(normalized_raw.min()),
        "raw_output_max": float(normalized_raw.max()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train single-crack local size field v4")
    parser.add_argument("--dataset", default="datasets/module30_single_straight_local_v4")
    parser.add_argument("--output", default="module30_checkpoints/single_straight_crack_local_v4.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument(
        "--refinement-weight", type=float, default=12.0,
        help="Extra loss weight alpha in 1 + alpha*(1-y)^2 for refined regions",
    )
    parser.add_argument(
        "--minimum-size-contrast", type=float, default=1.5,
        help="Fail training if test max(h)/min(h) is below this value",
    )
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=63042)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    root = Path(args.dataset)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_family") != SINGLE_CRACK_LOCAL_MODEL_FAMILY:
        raise ValueError("Dataset is not single-crack-local-size-field-v4")
    if list(metadata.get("feature_names", ())) != SINGLE_CRACK_LOCAL_FEATURE_NAMES:
        raise ValueError("Dataset feature schema does not match local v4")
    if metadata.get("feature_normalization") != "fixed_physical_bounded_no_dataset_zscore":
        raise ValueError("Dataset still requests empirical z-score normalization")
    if metadata.get("family_counts") != {"straight": metadata.get("meshes")}:
        raise ValueError("Training dataset must contain straight cracks only")
    with (root / metadata["manifest"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    train = _load_split(root, rows, "train")
    validation = _load_split(root, rows, "validation")
    test = _load_split(root, rows, "test")

    def loader(data, shuffle: bool):
        # Features are already dimensionless and bounded by fixed definitions.
        effective_weight = data[2] * (
            1.0 + float(args.refinement_weight) * (1.0 - data[1]) ** 2
        )
        effective_weight = effective_weight / max(
            float(effective_weight.mean()), 1.0e-30,
        )
        tensors = TensorDataset(
            torch.from_numpy(data[0]),
            torch.from_numpy(data[1]),
            torch.from_numpy(effective_weight.astype(np.float32)),
        )
        return DataLoader(
            tensors,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

    train_loader = loader(train, True)
    validation_loader = loader(validation, False)
    test_loader = loader(test, False)
    h_min = float(metadata["h_min"])
    h_max = float(metadata["h_max"])
    model = DirectNormalizedLogSizeMLP(
        len(SINGLE_CRACK_LOCAL_FEATURE_NAMES),
        args.hidden,
        args.layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(args.patience // 4, 5),
    )
    criterion = nn.SmoothL1Loss(reduction="none")
    best_state = None
    best_epoch = 0
    best_loss = float("inf")
    stale = 0
    history = []
    print(f"Device: {device}")
    print(f"Points: train={len(train[0])}, validation={len(validation[0])}, test={len(test[0])}")
    print("Input: 14 fixed-scale local features | empirical z-score: none")
    print("Training shapes: straight single cracks only")
    print(f"Refinement loss weight: 1 + {args.refinement_weight:g}*(1-y)^2")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for features, target, weight in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(features)
            loss = (criterion(predicted, target) * weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.item()) * len(target)
            count += len(target)
        validation_metrics = _evaluate(
            model, validation_loader, device, validation[3], h_min, h_max,
        )
        scheduler.step(validation_metrics["loss"])
        train_loss = total / max(count, 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_h_relative_l2": validation_metrics["h_relative_l2"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | train={train_loss:.6e} "
                f"val={validation_metrics['loss']:.6e} "
                f"h_rel={validation_metrics['h_relative_l2']:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    validation_metrics = _evaluate(
        model, validation_loader, device, validation[3], h_min, h_max,
    )
    test_metrics = _evaluate(model, test_loader, device, test[3], h_min, h_max)
    if test_metrics["predicted_size_contrast"] < args.minimum_size_contrast:
        raise RuntimeError(
            "Collapsed size-field model: test max(h)/min(h)="
            f"{test_metrics['predicted_size_contrast']:.6g} is below "
            f"{args.minimum_size_contrast:.6g}. No checkpoint was saved."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_family": SINGLE_CRACK_LOCAL_MODEL_FAMILY,
        "model_state_dict": model.state_dict(),
        "model_config": model.config(),
        "feature_names": SINGLE_CRACK_LOCAL_FEATURE_NAMES,
        "feature_normalization": "fixed_physical_bounded_no_dataset_zscore",
        "feature_train_min": np.asarray(metadata["feature_train_min"], dtype=np.float32),
        "feature_train_max": np.asarray(metadata["feature_train_max"], dtype=np.float32),
        "h_min": h_min,
        "h_max": h_max,
        "output_parameterization": "linear_normalized_log_h_clipped_at_runtime",
        "runtime_uses_analytic_size_field": False,
        "epoch": best_epoch,
        "validation_loss": best_loss,
        "target_metadata": metadata,
        "training_args": vars(args),
    }
    torch.save(checkpoint, output)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "model_family": SINGLE_CRACK_LOCAL_MODEL_FAMILY,
        "runtime_uses_analytic_size_field": False,
        "training_scope": "straight_single_cracks_only",
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "checkpoint": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Validation: {validation_metrics}")
    print(f"Test: {test_metrics}")
    print(f"Saved: {output}, {output.with_suffix('.csv')}, {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
