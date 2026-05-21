from __future__ import annotations

"""Train a simple ResNet18 stage1 reward model from labeled LIBERO images."""

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm

REPO_PARENT = Path(__file__).resolve().parents[4]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from serl_torch.examples.libero.runtime.stage_reward_model import (
    create_stage_reward_classifier,
)
from serl_torch.examples.libero.runtime.stage_reward_model import (
    observations_to_reward_model_obs,
)


_VIEW_COLUMNS = {
    "wrist": "wrist_image",
    "wrist_image": "wrist_image",
    "image_rgb_1": "wrist_image",
    "agentview": "agentview_image",
    "image": "agentview_image",
    "image_rgb_0": "agentview_image",
}


def _model_obs_key_for_view(view: str) -> str:
    if str(view) in ("wrist", "wrist_image", "image_rgb_1"):
        return "image_rgb_1"
    if str(view) in ("agentview", "image", "image_rgb_0"):
        return "image_rgb_0"
    return str(view)


@dataclass(frozen=True)
class RewardModelExample:
    demo: str
    step: int
    label: int
    image_paths: tuple[str, ...]


class StageRewardImageDataset(Dataset[tuple[dict[str, torch.Tensor], torch.Tensor]]):
    def __init__(
        self,
        examples: Sequence[RewardModelExample],
        *,
        image_size: int,
        views: Sequence[str],
    ) -> None:
        self.examples = tuple(examples)
        self.image_size = int(image_size)
        self.views = tuple(str(view) for view in views)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        example = self.examples[int(index)]
        obs: dict[str, np.ndarray] = {}
        for view, image_path in zip(self.views, example.image_paths):
            image = Image.open(image_path).convert("RGB")
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size),
                    resample=Image.BILINEAR,
                )
            obs[_model_obs_key_for_view(str(view))] = np.asarray(image, dtype=np.uint8)
        model_obs = observations_to_reward_model_obs(
            [obs],
            views=self.views,
            image_size=self.image_size,
        )
        tensor_obs = {
            key: torch.as_tensor(value[0], dtype=torch.uint8)
            for key, value in model_obs.items()
        }
        label = torch.tensor(float(example.label), dtype=torch.float32)
        return tensor_obs, label


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-csv",
        default=(
            "/Users/n/Documents/codex/datasets/LIBERO-datasets/"
            "libero_spatial/labled/labels.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", nargs="+", default=["wrist"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--resnet-model-name",
        default="pretrained_models/microsoft--resnet-18",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pooling-method", default="spatial_learned_embeddings")
    parser.add_argument("--num-spatial-blocks", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _column_for_view(view: str) -> str:
    try:
        return _VIEW_COLUMNS[str(view)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported reward-model view {view!r}; expected one of "
            f"{sorted(_VIEW_COLUMNS)}"
        ) from exc


def _resolve_labeled_image_path(
    path_text: str,
    *,
    labels_root: Path,
    demo: str,
    column: str,
) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    parts = path.parts
    if "labled" in parts:
        relative = Path(*parts[parts.index("labled") + 1 :])
        candidate = labels_root / relative
        if candidate.exists():
            return candidate

    view_dir = "wrist" if column == "wrist_image" else "agentview"
    candidate = labels_root / view_dir / str(demo) / path.name
    if candidate.exists():
        return candidate
    return path


def _load_examples(
    labels_csv: Path,
    *,
    views: Sequence[str],
    max_samples: int | None,
) -> list[RewardModelExample]:
    view_columns = tuple(_column_for_view(view) for view in views)
    examples: list[RewardModelExample] = []
    labels_root = labels_csv.parent
    with labels_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            raw_label = int(row["label"])
            if raw_label not in {-1, 1}:
                continue
            image_paths = tuple(
                str(
                    _resolve_labeled_image_path(
                        row[column],
                        labels_root=labels_root,
                        demo=str(row["demo"]),
                        column=column,
                    )
                )
                for column in view_columns
            )
            missing_paths = [path for path in image_paths if not Path(path).exists()]
            if missing_paths:
                raise FileNotFoundError(
                    f"Missing labeled image path(s): {missing_paths}"
                )
            examples.append(
                RewardModelExample(
                    demo=str(row["demo"]),
                    step=int(row["step"]),
                    label=1 if raw_label == 1 else 0,
                    image_paths=image_paths,
                )
            )
    if max_samples is not None:
        examples = examples[: max(0, int(max_samples))]
    if not examples:
        raise ValueError(f"No labeled examples found in {labels_csv}")
    return examples


def _split_by_demo(
    examples: Sequence[RewardModelExample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[RewardModelExample], list[RewardModelExample], list[str], list[str]]:
    demos = sorted({example.demo for example in examples})
    rng = random.Random(int(seed))
    rng.shuffle(demos)
    val_count = max(1, int(round(len(demos) * float(val_fraction)))) if len(demos) > 1 else 0
    val_demos = set(demos[:val_count])
    train_examples = [example for example in examples if example.demo not in val_demos]
    val_examples = [example for example in examples if example.demo in val_demos]
    if not train_examples or not val_examples:
        raise ValueError(
            "Demo-level train/val split produced an empty split; adjust --val-fraction"
        )
    return train_examples, val_examples, sorted(set(demos) - val_demos), sorted(val_demos)


def _label_counts(examples: Sequence[RewardModelExample]) -> dict[str, int]:
    positives = sum(int(example.label == 1) for example in examples)
    negatives = sum(int(example.label == 0) for example in examples)
    return {"negative": int(negatives), "positive": int(positives)}


def _metrics_for_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= float(threshold)
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.logical_and(predictions, positives).sum())
    tn = int(np.logical_and(~predictions, negatives).sum())
    fp = int(np.logical_and(predictions, negatives).sum())
    fn = int(np.logical_and(~predictions, positives).sum())
    accuracy = float((tp + tn) / max(1, int(labels.size)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_best_f1_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    candidates = [
        _metrics_for_threshold(probabilities, labels, threshold=float(threshold))
        for threshold in np.linspace(0.1, 0.9, 17)
    ]
    return max(candidates, key=lambda item: (float(item["f1"]), float(item["precision"])))


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for observations, labels in loader:
            observations = _observations_to_device(observations, device=device)
            labels = labels.to(device)
            logits = model(observations, train=False)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    return (
        float(np.mean(losses)) if losses else 0.0,
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0).astype(np.int64),
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    losses: list[float] = []
    progress = tqdm(loader, desc=f"train epoch {int(epoch)}", leave=False)
    for observations, labels in progress:
        observations = _observations_to_device(observations, device=device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(observations, train=True)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")
    return float(np.mean(losses)) if losses else 0.0


def _observations_to_device(
    observations: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in observations.items()
    }


def main() -> None:
    args = _parse_args()
    labels_csv = Path(args.labels_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    device = torch.device(
        str(args.device)
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    examples = _load_examples(
        labels_csv,
        views=tuple(args.views),
        max_samples=args.max_samples,
    )
    train_examples, val_examples, train_demos, val_demos = _split_by_demo(
        examples,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    train_loader = DataLoader(
        StageRewardImageDataset(
            train_examples,
            image_size=int(args.image_size),
            views=tuple(args.views),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )
    val_loader = DataLoader(
        StageRewardImageDataset(
            val_examples,
            image_size=int(args.image_size),
            views=tuple(args.views),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    resnet_kwargs = {
        "model_name": str(args.resnet_model_name),
        "pretrained": bool(args.pretrained),
        "freeze_backbone": bool(args.freeze_backbone),
        "pooling_method": str(args.pooling_method),
        "num_spatial_blocks": int(args.num_spatial_blocks),
        "bottleneck_dim": int(args.bottleneck_dim),
    }
    model = create_stage_reward_classifier(
        views=tuple(args.views),
        resnet_kwargs=resnet_kwargs,
        device=device,
        image_size=int(args.image_size),
    )
    split_payload = {
        "train_demos": train_demos,
        "val_demos": val_demos,
        "train_label_counts": _label_counts(train_examples),
        "val_label_counts": _label_counts(val_examples),
        "num_train_examples": int(len(train_examples)),
        "num_val_examples": int(len(val_examples)),
    }
    print(
        json.dumps(
            {
                "labels_csv": str(labels_csv),
                "views": list(args.views),
                "device": str(device),
                "image_size": int(args.image_size),
                "resnet_kwargs": resnet_kwargs,
                "split": split_payload,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        val_loss, val_probs, val_labels = _evaluate(model, val_loader, device=device)
        metrics_05 = _metrics_for_threshold(val_probs, val_labels, threshold=0.5)
        metrics_08 = _metrics_for_threshold(val_probs, val_labels, threshold=0.8)
        best_f1 = _select_best_f1_threshold(val_probs, val_labels)
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "metrics_at_0p5": metrics_05,
            "metrics_at_0p8": metrics_08,
            "best_f1_threshold": best_f1,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    final_metrics = history[-1]
    model_config = {
        "architecture": "serl_reward_classifier",
        "views": list(args.views),
        "image_size": int(args.image_size),
        "resnet_kwargs": resnet_kwargs,
    }
    train_config = {
        **vars(args),
        "labels_csv": str(labels_csv),
        "output_dir": str(output_dir),
        "device": str(device),
    }
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "train_config": train_config,
        "split": split_payload,
        "metrics": final_metrics,
        "history": history,
    }
    checkpoint_path = output_dir / "stage1_reward_model.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model_config": model_config,
                "train_config": train_config,
                "split": split_payload,
                "metrics": final_metrics,
                "history": history,
                "checkpoint_path": str(checkpoint_path),
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    print(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
                "split": split_payload,
                "final_metrics": final_metrics,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
