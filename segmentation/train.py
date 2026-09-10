"""Train 7-class semantic change segmentation on SECOND (build-order step 5).

Two outputs per pair: the class map at t1 and at t2, where class 0 means
"unchanged" and is identical in both by construction. A shared encoder sees
both dates; each decoder head sees its own date's features alongside the
difference, because whether a pixel carries a class at all depends on both
images, not one.

Checkpoint selection uses CDVQA **Val** only. Test is never read here, and
the dataset layer asserts that no Test or Val scene reaches training.

Run the pilot before any full run::

    python -m segmentation.train --pilot --steps 50

which reports peak VRAM and per-step time so the full-run cost is a measured
extrapolation rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segmentation.dataset import (
    SECONDChangeDataset,
    assert_split_disjoint,
    cdvqa_split_scenes,
)
from segmentation.validate_cdvqa import CDVQAValidator, format_validation
from tools.change_analysis.cdvqa import N_CLASSES

__all__ = ["SiameseChangeNet", "TrainConfig", "run_pilot", "train"]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SiameseChangeNet(nn.Module):
    """Shared ResNet-18 encoder, two UNet decoders, 7 classes each.

    Siamese rather than early-fusion: the encoder sees each date separately
    with shared weights, so the pretrained ImageNet statistics stay valid for
    3-channel input. Each decoder is fed its own date's skip features
    concatenated with the temporal difference, which is what lets a head
    decide "unchanged" (class 0) rather than guessing land cover.
    """

    def __init__(self, n_classes: int = N_CLASSES, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2, 64
        self.pool = net.maxpool
        self.layer1 = net.layer1   # /4,  64
        self.layer2 = net.layer2   # /8,  128
        self.layer3 = net.layer3   # /16, 256
        self.layer4 = net.layer4   # /32, 512

        # Each skip carries [own features, difference] -> doubled channels.
        self.decoders = nn.ModuleList()
        for _ in range(2):
            self.decoders.append(
                nn.ModuleList([
                    _DecoderBlock(1024, 512, 256),  # /32 -> /16
                    _DecoderBlock(256, 256, 128),   # /16 -> /8
                    _DecoderBlock(128, 128, 64),    # /8  -> /4
                    _DecoderBlock(64, 128, 64),     # /4  -> /2
                ])
            )
        self.heads = nn.ModuleList(
            [nn.Conv2d(64, n_classes, 1) for _ in range(2)]
        )

    def _encode(self, x: torch.Tensor):
        s0 = self.stem(x)            # /2,  64
        s1 = self.layer1(self.pool(s0))  # /4,  64
        s2 = self.layer2(s1)         # /8,  128
        s3 = self.layer3(s2)         # /16, 256
        s4 = self.layer4(s3)         # /32, 512
        return s0, s1, s2, s3, s4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self._encode(x[:, :3])
        b = self._encode(x[:, 3:])

        outputs = []
        for index, (decoder, head) in enumerate(zip(self.decoders, self.heads)):
            own, other = (a, b) if index == 0 else (b, a)
            # Bottleneck and skips both carry own-vs-difference context.
            y = torch.cat([own[4], own[4] - other[4]], dim=1)
            skips = [
                torch.cat([own[3], own[3] - other[3]], dim=1),
                torch.cat([own[2], own[2] - other[2]], dim=1),
                torch.cat([own[1], own[1] - other[1]], dim=1),
                torch.cat([own[0], own[0] - other[0]], dim=1),
            ]
            for block, skip in zip(decoder, skips):
                y = block(y, skip)
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            outputs.append(head(y))
        return outputs[0], outputs[1]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def confusion(pred: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    k = (target * n + pred).view(-1)
    return torch.bincount(k, minlength=n * n).view(n, n)


def miou_from_confusion(matrix: torch.Tensor) -> Tuple[float, list]:
    matrix = matrix.double()
    intersection = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    per_class = (intersection / union.clamp(min=1)).tolist()
    present = union > 0
    if present.sum() == 0:
        return float("nan"), per_class
    valid = (intersection[present] / union[present]).mean().item()
    return valid, per_class


# --------------------------------------------------------------------------
# Config and training
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    cdvqa_root: str = "datasets/CDVQA"
    second_root: str = "datasets/SECOND_raw/train"
    batch_size: int = 4
    accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    workers: int = 4
    amp: bool = True
    class_weight_power: float = 0.0
    out_dir: str = "outputs/segmentation"
    seed: int = 0


def _build_loaders(config: TrainConfig, pilot: bool = False):
    train_scenes = cdvqa_split_scenes(config.cdvqa_root, "Train")
    val_scenes = cdvqa_split_scenes(config.cdvqa_root, "Val")
    test_scenes = cdvqa_split_scenes(config.cdvqa_root, "Test")
    test2_scenes = cdvqa_split_scenes(config.cdvqa_root, "Test2")

    # CLAUDE.md section 2: fail loudly, never warn quietly.
    assert_split_disjoint(
        train_scenes, val=val_scenes, test=test_scenes, test2=test2_scenes
    )

    if pilot:
        train_scenes = train_scenes[: max(config.batch_size * 8, 32)]
        val_scenes = val_scenes[:16]

    train_set = SECONDChangeDataset(
        config.second_root, train_scenes, augment=True, seed=config.seed
    )
    val_set = SECONDChangeDataset(config.second_root, val_scenes, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True, drop_last=True,
        persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True,
    )
    return train_loader, val_loader, train_scenes, val_scenes


def _criterion(config: TrainConfig, device) -> nn.Module:
    if config.class_weight_power <= 0:
        return nn.CrossEntropyLoss()
    # Inverse-frequency weights, softened by an exponent so the rare classes
    # do not dominate outright. Frequencies are measured, not assumed.
    from segmentation.dataset import class_pixel_counts

    scenes = cdvqa_split_scenes(config.cdvqa_root, "Train")
    counts = class_pixel_counts(config.second_root, scenes, limit=200)
    total = sum(counts.values())
    weights = torch.tensor(
        [(total / max(counts[c], 1)) ** config.class_weight_power
         for c in range(N_CLASSES)],
        dtype=torch.float32, device=device,
    )
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=weights)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> Dict[str, float]:
    model.eval()
    matrix = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=device)
    for x, y1, y2 in loader:
        x, y1, y2 = x.to(device, non_blocking=True), y1.to(device), y2.to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            o1, o2 = model(x)
        matrix += confusion(o1.argmax(1), y1, N_CLASSES)
        matrix += confusion(o2.argmax(1), y2, N_CLASSES)
    miou, per_class = miou_from_confusion(matrix.cpu())
    correct = matrix.diag().sum().item()
    return {
        "miou": miou,
        "pixel_accuracy": correct / max(matrix.sum().item(), 1),
        "per_class_iou": per_class,
    }


def run_pilot(config: TrainConfig, steps: int = 50) -> Dict[str, object]:
    """Measure peak VRAM and per-step time, then extrapolate the full run."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("pilot requires CUDA; no GPU is visible to torch")

    train_loader, _, _, _ = _build_loaders(config, pilot=True)
    full_train_scenes = len(cdvqa_split_scenes(config.cdvqa_root, "Train"))

    model = SiameseChangeNet().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    criterion = _criterion(config, device)

    torch.cuda.reset_peak_memory_stats()
    model.train()

    done = 0
    timings = []
    start_all = time.time()
    while done < steps:
        for x, y1, y2 in train_loader:
            if done >= steps:
                break
            step_start = time.time()
            x = x.to(device, non_blocking=True)
            y1, y2 = y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config.amp):
                o1, o2 = model(x)
                loss = criterion(o1, y1) + criterion(o2, y2)
            scaler.scale(loss / config.accum_steps).backward()
            if (done + 1) % config.accum_steps == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            timings.append(time.time() - step_start)
            done += 1

    # Discard the first few steps: cuDNN autotuning and allocator warmup
    # make them unrepresentative of steady state.
    steady = timings[5:] if len(timings) > 10 else timings
    per_step = float(np.median(steady))
    steps_per_epoch = full_train_scenes // config.batch_size
    epoch_seconds = per_step * steps_per_epoch

    return {
        "device": torch.cuda.get_device_name(0),
        "total_vram_mib": torch.cuda.get_device_properties(0).total_memory / 2 ** 20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
        "steps_measured": done,
        "median_step_seconds": per_step,
        "wall_seconds": time.time() - start_all,
        "batch_size": config.batch_size,
        "amp": config.amp,
        "train_scenes_full": full_train_scenes,
        "steps_per_epoch": steps_per_epoch,
        "projected_epoch_minutes": epoch_seconds / 60.0,
        "projected_full_run_hours": epoch_seconds * config.epochs / 3600.0,
        "epochs_planned": config.epochs,
        "params_millions": sum(p.numel() for p in model.parameters()) / 1e6,
    }


def train(config: TrainConfig) -> Dict[str, object]:
    """Full training run.

    Checkpoints are selected on CDVQA Val **average accuracy** -- the actual
    objective -- not on mIoU, which is a proxy for it. mIoU is logged as a
    diagnostic so the correlation between segmentation quality and answer
    accuracy is measurable rather than assumed.

    Test is never read here.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_scenes, val_scenes = _build_loaders(config)
    os.makedirs(config.out_dir, exist_ok=True)

    # The validator walks val_scenes in the same order the loader yields them,
    # so the loader must not shuffle.
    validator = CDVQAValidator(
        config.cdvqa_root, config.second_root, "Val", scenes=val_scenes
    )

    model = SiameseChangeNet().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.epochs * max(len(train_loader), 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    criterion = _criterion(config, device)

    best = {"average_accuracy": -1.0, "epoch": -1}
    history = []
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for step, (x, y1, y2) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y1, y2 = y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config.amp):
                o1, o2 = model(x)
                loss = criterion(o1, y1) + criterion(o2, y2)
            scaler.scale(loss / config.accum_steps).backward()
            if (step + 1) % config.accum_steps == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        report = validator.run(model, val_loader, device, config.amp)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_average_accuracy": report["average_accuracy"],
            "val_overall_accuracy": report["overall_accuracy"],
            "val_miou": report["miou"],
            "val_pixel_accuracy": report["pixel_accuracy"],
            "head_disagreement": report["head_disagreement"],
            "unanswered": report["unanswered_total"],
            "per_class_iou": report["per_class_iou"],
        }
        history.append(entry)
        print(
            f"epoch {epoch:3d}  loss {train_loss:.4f}  "
            f"val AA {report['average_accuracy'] * 100:6.2f}%  "
            f"OA {report['overall_accuracy'] * 100:6.2f}%  "
            f"mIoU {report['miou'] * 100:5.2f}%  "
            f"head-disagree {report['head_disagreement'] * 100:.3f}%",
            flush=True,
        )

        if report["average_accuracy"] > best["average_accuracy"]:
            best = {
                "average_accuracy": report["average_accuracy"],
                "overall_accuracy": report["overall_accuracy"],
                "miou": report["miou"],
                "epoch": epoch,
            }
            torch.save(
                {"model": model.state_dict(), "config": asdict(config),
                 "epoch": epoch, "val_report": report},
                os.path.join(config.out_dir, "best.pt"),
            )
            with open(os.path.join(config.out_dir, "best_val_report.json"),
                      "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

    print()
    print(format_validation(report))
    with open(os.path.join(config.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "best": best,
                   "n_train": len(train_scenes), "n_val": len(val_scenes)},
                  f, indent=2)
    return {"best": best, "history": history}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="measure VRAM and step time, then stop")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--out-dir", default="outputs/segmentation")
    args = parser.parse_args(argv)

    config = TrainConfig(
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        workers=args.workers,
        amp=not args.no_amp,
        class_weight_power=args.class_weight_power,
        out_dir=args.out_dir,
    )

    if args.pilot:
        report = run_pilot(config, args.steps)
        width = max(len(k) for k in report)
        print("PILOT")
        print("-" * (width + 30))
        for key, value in report.items():
            rendered = f"{value:.2f}" if isinstance(value, float) else value
            print(f"  {key:<{width}}  {rendered}")
        os.makedirs(config.out_dir, exist_ok=True)
        with open(os.path.join(config.out_dir, "pilot.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return 0

    train(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
