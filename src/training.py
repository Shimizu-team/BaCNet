"""Reusable BaCNet training loop."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = predictions.astype(np.float64).reshape(-1)
    targets = targets.astype(np.float64).reshape(-1)
    residual = targets - predictions
    mse = float(np.mean(residual**2))
    target_ss = float(np.sum((targets - targets.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / target_ss) if target_ss > 0 else float("nan")
    pcc = float(np.corrcoef(predictions, targets)[0, 1]) if len(targets) > 1 else float("nan")
    pred_rank = pd.Series(predictions).rank(method="average").to_numpy()
    target_rank = pd.Series(targets).rank(method="average").to_numpy()
    scc = float(np.corrcoef(pred_rank, target_rank)[0, 1]) if len(targets) > 1 else float("nan")
    return {"mse": mse, "r2": r2, "pcc": pcc, "scc": scc}


def _evaluate(model, loader, accelerator, criterion):
    model.eval()
    predictions = []
    targets = []
    indices = []
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for features, labels, row_indices in loader:
            labels = labels.float().view(-1, 1)
            outputs = model(features).float()
            loss = criterion(outputs, labels)
            gathered_outputs, gathered_labels, gathered_indices = accelerator.gather_for_metrics(
                (outputs, labels, row_indices)
            )
            predictions.append(gathered_outputs.detach().cpu())
            targets.append(gathered_labels.detach().cpu())
            indices.append(gathered_indices.detach().cpu())
            loss_sum += float(loss.detach()) * features.size(0)
            sample_count += features.size(0)
    pred_array = torch.cat(predictions).numpy().reshape(-1)
    target_array = torch.cat(targets).numpy().reshape(-1)
    index_array = torch.cat(indices).numpy().reshape(-1)
    totals = torch.tensor([loss_sum, sample_count], device=accelerator.device, dtype=torch.float32)
    totals = accelerator.reduce(totals, reduction="sum").cpu().numpy()
    metrics = regression_metrics(pred_array, target_array)
    metrics["loss"] = float(totals[0] / totals[1])
    return metrics, pred_array, target_array, index_array


def _save_checkpoint(path: Path, model, optimizer, epoch: int, best_loss: float, config: dict[str, Any], accelerator):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = {
            "model_state_dict": accelerator.get_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_loss,
            "config": config,
        }
        accelerator.save(state, path)


def train_model(
    model,
    loaders,
    optimizer,
    accelerator,
    output_dir: Path,
    config: dict[str, Any],
    epochs: int,
    patience: int,
    resume_path: str | None = None,
):
    criterion = torch.nn.MSELoss()
    model, optimizer, train_loader, validation_loader, test_loader = accelerator.prepare(
        model,
        optimizer,
        loaders["train"],
        loaders["validation"],
        loaders["test"],
    )
    start_epoch = 0
    best_loss = math.inf
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])

    history: list[dict[str, float | int | str]] = []
    epochs_without_improvement = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for features, labels, _ in train_loader:
            labels = labels.float().view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features).float()
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            loss_sum += float(loss.detach()) * features.size(0)
            sample_count += features.size(0)

        totals = torch.tensor([loss_sum, sample_count], device=accelerator.device, dtype=torch.float32)
        totals = accelerator.reduce(totals, reduction="sum").cpu().numpy()
        train_loss = float(totals[0] / totals[1])
        validation_metrics, _, _, _ = _evaluate(model, validation_loader, accelerator, criterion)
        history.append({"epoch": epoch, "split": "train", "loss": train_loss})
        history.append({"epoch": epoch, "split": "validation", **validation_metrics})

        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            _save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, epoch, best_loss, config, accelerator)
        else:
            epochs_without_improvement += 1
        _save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, best_loss, config, accelerator)

        if accelerator.is_main_process:
            pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
            accelerator.print(
                f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} "
                f"validation_loss={validation_metrics['loss']:.6f}"
            )

        stop = torch.tensor(
            int(epochs_without_improvement >= patience), device=accelerator.device, dtype=torch.int32
        )
        stop = accelerator.reduce(stop, reduction="max")
        if int(stop.item()):
            break

    accelerator.wait_for_everyone()
    best_checkpoint_path = output_dir / "checkpoint_best.pt"
    if not best_checkpoint_path.is_file() and resume_path:
        # This occurs when a checkpoint is resumed into a new output directory
        # and no further epoch is scheduled. In that case, evaluate the supplied
        # checkpoint rather than failing after a successful state restoration.
        best_checkpoint_path = Path(resume_path)
    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    accelerator.unwrap_model(model).load_state_dict(best_checkpoint["model_state_dict"])
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.save(accelerator.get_state_dict(model), output_dir / "model_best_state_dict.pt")
    test_metrics, predictions, targets, indices = _evaluate(model, test_loader, accelerator, criterion)
    if accelerator.is_main_process:
        predictions_frame = pd.DataFrame(
            {"dataset_row": indices.astype(int), "target_bct": targets, "prediction": predictions}
        ).sort_values("dataset_row")
        predictions_frame.to_csv(output_dir / "test_predictions.csv", index=False)
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2, allow_nan=True)
    return history, test_metrics
