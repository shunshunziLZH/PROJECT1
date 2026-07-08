from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_prediction.data import TBDataset, describe_dataset, save_rgb
from tb_prediction.checkpoint import load_checkpoint_file
from tb_prediction.model import TBResUNet, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def diff_x(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[:, :, :, 1:] - tensor[:, :, :, :-1]


def diff_y(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[:, :, 1:, :] - tensor[:, :, :-1, :]


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(diff_x(pred) - diff_x(target))) + torch.mean(torch.abs(diff_y(pred) - diff_y(target)))


def tv_loss(tensor: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(diff_x(tensor))) + torch.mean(torch.abs(diff_y(tensor)))


def tb_loss(
    outputs: Dict[str, torch.Tensor],
    target_t: torch.Tensor,
    target_b: torch.Tensor,
    grad_weight: float = 0.1,
    tv_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    l1 = nn.functional.l1_loss
    loss_t = l1(outputs["T"], target_t)
    loss_b = l1(outputs["B"], target_b)
    loss_t_grad = gradient_l1(outputs["T"], target_t)
    loss_b_tv = tv_loss(outputs["B"])
    total = loss_t + loss_b + grad_weight * loss_t_grad + tv_weight * loss_b_tv
    return total, {
        "loss": total.detach(),
        "T_MAE": loss_t.detach(),
        "B_MAE": loss_b.detach(),
        "loss_t_grad": loss_t_grad.detach(),
        "loss_b_tv": loss_b_tv.detach(),
    }


def resolve_amp_dtype(device: torch.device, amp_enabled: bool, amp_dtype: str) -> torch.dtype:
    if not amp_enabled:
        return torch.float16
    if device.type != "cuda":
        return torch.float16
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 AMP was requested, but this CUDA device does not report BF16 support.")
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def amp_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype)


def move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    channels_last: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    return (
        batch["I"].to(device, non_blocking=True, memory_format=memory_format),
        batch["T"].to(device, non_blocking=True, memory_format=memory_format),
        batch["B"].to(device, non_blocking=True, memory_format=memory_format),
    )


def average_dict(values: Dict[str, float], count: int) -> Dict[str, float]:
    return {key: val / max(1, count) for key, val in values.items()}


def run_epoch(
    model: TBResUNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    grad_weight: float,
    tv_weight: float,
    amp: bool,
    amp_dtype: torch.dtype,
    clip_grad: float,
    channels_last: bool,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: Dict[str, float] = {}
    sample_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, leave=False, desc="train" if is_train else "val")
        for batch in pbar:
            image_i, target_t, target_b = move_batch(batch, device, channels_last=channels_last)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                outputs = model(image_i)
            outputs_for_loss = {key: value.float() for key, value in outputs.items()}
            loss, loss_items = tb_loss(
                outputs_for_loss,
                target_t.float(),
                target_b.float(),
                grad_weight=grad_weight,
                tv_weight=tv_weight,
            )

            if not torch.isfinite(loss):
                batch_ids = batch.get("id", [])
                diagnostics = {
                    "ids": list(batch_ids) if isinstance(batch_ids, (list, tuple)) else batch_ids,
                    "I_finite": bool(torch.isfinite(image_i).all().item()),
                    "T_finite": bool(torch.isfinite(target_t).all().item()),
                    "B_finite": bool(torch.isfinite(target_b).all().item()),
                    "T_hat_finite": bool(torch.isfinite(outputs["T"]).all().item()),
                    "B_hat_finite": bool(torch.isfinite(outputs["B"]).all().item()),
                }
                raise FloatingPointError(f"Non-finite loss detected: {loss.item()} | {diagnostics}")

            if is_train:
                scaler.scale(loss).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()

            batch_size = image_i.shape[0]
            sample_count += batch_size
            for key, value in loss_items.items():
                totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
            pbar.set_postfix({key: f"{value / sample_count:.4f}" for key, value in totals.items() if key in {"loss", "T_MAE", "B_MAE"}})

    return average_dict(totals, sample_count)


def save_validation_visuals(
    model: TBResUNet,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    max_images: int,
    amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
) -> None:
    if max_images <= 0:
        return

    model.eval()
    visual_dir = output_dir / "visualization" / f"epoch_{epoch:04d}"
    saved = 0

    with torch.no_grad():
        for batch in loader:
            image_i, target_t, target_b = move_batch(batch, device, channels_last=channels_last)
            with autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                outputs = model(image_i)

            image_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else str(saved)
            save_rgb(target_t[0], visual_dir / f"{image_id}_T_gt.png")
            save_rgb(outputs["T"][0], visual_dir / f"{image_id}_T_hat.png")
            save_rgb(torch.abs(outputs["T"][0] - target_t[0]), visual_dir / f"{image_id}_T_abs_error.png")
            save_rgb(target_b[0], visual_dir / f"{image_id}_B_gt.png")
            save_rgb(outputs["B"][0], visual_dir / f"{image_id}_B_hat.png")
            save_rgb(torch.abs(outputs["B"][0] - target_b[0]), visual_dir / f"{image_id}_B_abs_error.png")

            saved += 1
            if saved >= max_images:
                break

    print(f"Saved validation visuals to {visual_dir}")


def create_dataloaders(
    args: argparse.Namespace,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    train_set = TBDataset(
        root=args.data_root,
        split="train",
        val_ratio=args.val_ratio,
        patch_size=patch_size,
        use_flip=not args.no_flip,
        use_rot=not args.no_rot,
        limit=args.limit_train,
        cache_data=args.cache_data,
    )
    val_set = TBDataset(
        root=args.data_root,
        split="val",
        val_ratio=args.val_ratio,
        patch_size=None,
        use_flip=False,
        use_rot=False,
        limit=args.limit_val,
        cache_data=args.cache_data,
    )

    train_loader_kwargs = {}
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_lr_at_epoch(base_lr: float, eta_min: float, total_epochs: int, completed_epochs: int) -> float:
    if total_epochs <= 0:
        return base_lr
    completed_epochs = min(max(0, completed_epochs), total_epochs)
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * completed_epochs / total_epochs)) / 2


def get_stage_monitor(
    monitor_state: Dict[str, object],
    stage_name: str,
    initial_best: float,
) -> Dict[str, object]:
    stages = monitor_state.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        monitor_state["stages"] = stages
    entry = stages.setdefault(stage_name, {})
    if not isinstance(entry, dict):
        entry = {}
        stages[stage_name] = entry
    if "best_val" not in entry:
        entry["best_val"] = float(initial_best) if math.isfinite(float(initial_best)) else math.inf
    entry.setdefault("best_epoch", 0)
    entry.setdefault("bad_validations", 0)
    entry.setdefault("last_val", math.inf)
    entry.setdefault("notified", False)
    return entry


def update_training_monitor(
    stage_name: str,
    stage_epoch: int,
    global_epoch: int,
    val_loss: float,
    monitor_entry: Dict[str, object],
    patience: int,
    min_delta: float,
    output_dir: Path,
) -> Tuple[bool, Optional[str], str]:
    best_stage_val = float(monitor_entry.get("best_val", math.inf))
    improved = val_loss < best_stage_val - min_delta
    if improved:
        monitor_entry["best_val"] = val_loss
        monitor_entry["best_epoch"] = global_epoch
        monitor_entry["bad_validations"] = 0
        monitor_entry["notified"] = False
        monitor_entry["last_val"] = val_loss
        return True, None, ""

    bad_validations = int(monitor_entry.get("bad_validations", 0)) + 1
    monitor_entry["bad_validations"] = bad_validations
    monitor_entry["last_val"] = val_loss

    if patience <= 0 or bad_validations < patience:
        return False, None, ""

    if bool(monitor_entry.get("notified", False)):
        action = "advance" if stage_name.startswith("stage1") else "pause"
        return False, action, ""

    monitor_entry["notified"] = True
    best_epoch = int(monitor_entry.get("best_epoch", 0))
    best_path = output_dir / "best.pth"
    if stage_name.startswith("stage1"):
        message = (
            f"[monitor] {stage_name}: val loss has not improved by at least {min_delta:g} "
            f"for {bad_validations} validation checks. Best stage val={best_stage_val:.6f} "
            f"at global epoch {best_epoch}. Advancing to the next stage."
        )
        return False, "advance", message

    message = (
        f"[monitor] {stage_name}: val loss has not improved by at least {min_delta:g} "
        f"for {bad_validations} validation checks. Best stage val={best_stage_val:.6f} "
        f"at global epoch {best_epoch}. Consider pausing training and using {best_path}."
    )
    return False, "pause", message


def save_checkpoint(
    path: Path,
    model: TBResUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
    monitor_state: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": model.config.__dict__,
        "args": vars(args),
    }
    if monitor_state is not None:
        payload["monitor_state"] = monitor_state
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: TBResUNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: object = "cpu",
    return_monitor_state: bool = False,
):
    checkpoint = load_checkpoint_file(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1
    best_val = float(checkpoint.get("best_val", math.inf)) if isinstance(checkpoint, dict) else math.inf
    monitor_state = checkpoint.get("monitor_state", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(monitor_state, dict):
        monitor_state = {}
    if return_monitor_state:
        return start_epoch, best_val, monitor_state
    return start_epoch, best_val


def train_stage(
    stage_name: str,
    model: TBResUNet,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    best_val: float,
    patch_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    global_epoch_offset: int,
    total_epochs: int,
    amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    monitor_state: Dict[str, object],
    start_stage_epoch: int = 1,
) -> Tuple[int, float]:
    if epochs <= 0:
        return global_epoch_offset, best_val
    if start_stage_epoch > epochs:
        print(f"Stage {stage_name}: already complete, skipping.")
        return global_epoch_offset + epochs, best_val
    if start_stage_epoch < 1:
        raise ValueError("start_stage_epoch must be >= 1")

    completed_stage_epochs = start_stage_epoch - 1
    current_stage_lr = cosine_lr_at_epoch(lr, args.eta_min, epochs, completed_stage_epochs)
    set_optimizer_lr(optimizer, current_stage_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.eta_min)
    scheduler.base_lrs = [lr for _ in scheduler.base_lrs]
    scheduler.last_epoch = completed_stage_epochs
    scheduler._last_lr = [current_stage_lr for _ in optimizer.param_groups]
    train_loader, val_loader = create_dataloaders(args, patch_size=patch_size, batch_size=batch_size, device=device)
    stage_monitor = get_stage_monitor(monitor_state, stage_name, best_val)
    monitor_patience = args.stage1_patience if stage_name.startswith("stage1") else args.stage2_patience

    print(
        f"Stage {stage_name}: patch_size={patch_size}, batch_size={batch_size}, "
        f"epochs={epochs}, start_epoch={start_stage_epoch}, lr={lr:.6g}, scheduler=cosine"
    )

    for stage_epoch in range(start_stage_epoch, epochs + 1):
        global_epoch = global_epoch_offset + stage_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {global_epoch}/{total_epochs} | stage={stage_name} {stage_epoch}/{epochs} | lr={current_lr:.6g}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            grad_weight=args.grad_weight,
            tv_weight=args.tv_weight,
            amp=amp,
            amp_dtype=amp_dtype,
            clip_grad=args.clip_grad,
            channels_last=channels_last,
        )
        should_validate = args.val_freq > 0 and (global_epoch % args.val_freq == 0 or stage_epoch == epochs)
        if should_validate:
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                scaler=scaler,
                grad_weight=args.grad_weight,
                tv_weight=args.tv_weight,
                amp=amp,
                amp_dtype=amp_dtype,
                clip_grad=0,
                channels_last=channels_last,
            )
        else:
            val_metrics = {}
        monitor_action: Optional[str] = None
        monitor_message = ""
        global_improved = False
        if should_validate:
            val_loss = float(val_metrics["loss"])
            _, monitor_action, monitor_message = update_training_monitor(
                stage_name,
                stage_epoch,
                global_epoch,
                val_loss,
                stage_monitor,
                patience=monitor_patience,
                min_delta=args.monitor_min_delta,
                output_dir=args.output_dir,
            )
            global_improved = val_loss < best_val
            if global_improved:
                best_val = val_loss
        scheduler.step()

        print(f"train={train_metrics} | val={val_metrics if should_validate else 'skipped'}")
        if monitor_message:
            print(monitor_message)
        if should_validate and args.visualize_freq > 0 and global_epoch % args.visualize_freq == 0:
            save_validation_visuals(
                model,
                val_loader,
                device,
                args.output_dir,
                global_epoch,
                max_images=args.num_visuals,
                amp=amp,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
            )
        save_checkpoint(args.output_dir / "last.pth", model, optimizer, scheduler, global_epoch, best_val, args, monitor_state)
        save_checkpoint(args.output_dir / f"last_{stage_name}.pth", model, optimizer, scheduler, global_epoch, best_val, args, monitor_state)
        if global_improved:
            save_checkpoint(args.output_dir / "best.pth", model, optimizer, scheduler, global_epoch, best_val, args, monitor_state)
            print(f"Saved new best checkpoint: {best_val:.6f}")
        if monitor_action == "advance":
            return global_epoch_offset + epochs, best_val
        if monitor_action == "pause" and args.stop_on_plateau:
            print(f"[monitor] stop-on-plateau enabled; stopping after {stage_name} global epoch {global_epoch}.")
            return global_epoch_offset + epochs, best_val

    return global_epoch_offset + epochs, best_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a standalone I -> T/B prediction network.")
    parser.add_argument("--data-root", type=Path, default=Path("basicsr/data/DATA"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/tb_prediction"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--two-stage", action="store_true", help="Use the 256px pretrain + 512px finetune schedule.")
    parser.add_argument("--stage1-patch-size", type=int, default=256)
    parser.add_argument("--stage1-batch-size", type=int, default=16)
    parser.add_argument("--stage1-epochs", type=int, default=60)
    parser.add_argument("--stage1-lr", type=float, default=2e-4)
    parser.add_argument("--stage2-patch-size", type=int, default=512)
    parser.add_argument("--stage2-batch-size", type=int, default=4)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cache-data", type=str, default="none", choices=("none", "ram"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--grad-weight", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="auto",
        choices=("auto", "bf16", "fp16"),
        help="AMP dtype. auto prefers BF16 on supported CUDA devices because it is more stable than FP16.",
    )
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--limit-train", type=int, default=None, help="Limit train samples for smoke tests.")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit validation samples for smoke tests.")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--no-rot", action="store_true")
    parser.add_argument("--visualize-freq", type=int, default=5, help="Save validation visualizations every N epochs. Set <=0 to disable.")
    parser.add_argument("--val-freq", type=int, default=1, help="Run full validation every N epochs. Set <=0 to disable validation.")
    parser.add_argument("--monitor-min-delta", type=float, default=1e-4, help="Minimum validation loss improvement counted by the stage monitor.")
    parser.add_argument("--stage1-patience", type=int, default=2, help="Validation checks without improvement before advancing from stage1 to stage2.")
    parser.add_argument("--stage2-patience", type=int, default=4, help="Validation checks without improvement before suggesting pause in stage2.")
    parser.add_argument("--stop-on-plateau", action="store_true", help="Stop automatically when stage2 monitor reaches patience.")
    parser.add_argument("--num-visuals", type=int, default=4, help="Number of validation samples to visualize each time.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory format on CUDA for faster convolutions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(device, amp, args.amp_dtype)
    channels_last = bool(args.channels_last and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(describe_dataset(args.data_root, val_ratio=args.val_ratio))
    if amp:
        print(f"AMP enabled: dtype={amp_dtype_name(amp_dtype)}")
    if args.clip_grad > 0:
        print(f"Gradient clipping: max_norm={args.clip_grad:g}")
    if channels_last:
        print("Channels-last memory format enabled.")
    if args.cache_data != "none":
        print(f"Data cache enabled: {args.cache_data}")

    model = TBResUNet(base_channels=args.base_channels).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device.type, enabled=amp and amp_dtype == torch.float16)

    best_val = math.inf
    monitor_state: Dict[str, object] = {}
    print(f"Model parameters: {count_parameters(model):,}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.two_stage:
        completed_epoch = 0
        if args.resume is not None:
            start_epoch, best_val, monitor_state = load_checkpoint(
                args.resume,
                model,
                optimizer,
                scheduler=None,
                device=device,
                return_monitor_state=True,
            )
            completed_epoch = max(0, start_epoch - 1)
            print(f"Resuming two-stage training from {args.resume} after global epoch {completed_epoch}.")

        total_epochs = args.stage1_epochs + args.stage2_epochs
        if completed_epoch >= total_epochs:
            print(f"Checkpoint already reached global epoch {completed_epoch}/{total_epochs}; nothing to train.")
            return

        stage1_start = min(max(completed_epoch + 1, 1), args.stage1_epochs + 1)
        global_epoch, best_val = train_stage(
            "stage1_256",
            model,
            optimizer,
            scaler,
            device,
            args,
            best_val,
            patch_size=args.stage1_patch_size,
            batch_size=args.stage1_batch_size,
            epochs=args.stage1_epochs,
            lr=args.stage1_lr,
            global_epoch_offset=0,
            total_epochs=total_epochs,
            amp=amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            monitor_state=monitor_state,
            start_stage_epoch=stage1_start,
        )

        stage2_completed = max(0, completed_epoch - args.stage1_epochs)
        stage2_start = min(max(stage2_completed + 1, 1), args.stage2_epochs + 1)
        train_stage(
            "stage2_512",
            model,
            optimizer,
            scaler,
            device,
            args,
            best_val,
            patch_size=args.stage2_patch_size,
            batch_size=args.stage2_batch_size,
            epochs=args.stage2_epochs,
            lr=args.stage2_lr,
            global_epoch_offset=args.stage1_epochs,
            total_epochs=total_epochs,
            amp=amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            monitor_state=monitor_state,
            start_stage_epoch=stage2_start,
        )
        return

    train_loader, val_loader = create_dataloaders(args, patch_size=args.patch_size, batch_size=args.batch_size, device=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    start_epoch = 1
    if args.resume is not None:
        start_epoch, best_val, monitor_state = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device=device,
            return_monitor_state=True,
        )
        print(f"Resuming training from {args.resume} at epoch {start_epoch}/{args.epochs}.")
    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}/{args.epochs}; nothing to train.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | lr={lr:.6g}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            grad_weight=args.grad_weight,
            tv_weight=args.tv_weight,
            amp=amp,
            amp_dtype=amp_dtype,
            clip_grad=args.clip_grad,
            channels_last=channels_last,
        )
        should_validate = args.val_freq > 0 and (epoch % args.val_freq == 0 or epoch == args.epochs)
        if should_validate:
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                scaler=scaler,
                grad_weight=args.grad_weight,
                tv_weight=args.tv_weight,
                amp=amp,
                amp_dtype=amp_dtype,
                clip_grad=0,
                channels_last=channels_last,
            )
        else:
            val_metrics = {}
        global_improved = False
        if should_validate and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            global_improved = True
        scheduler.step()

        print(f"train={train_metrics} | val={val_metrics if should_validate else 'skipped'}")
        if should_validate and args.visualize_freq > 0 and epoch % args.visualize_freq == 0:
            save_validation_visuals(
                model,
                val_loader,
                device,
                args.output_dir,
                epoch,
                max_images=args.num_visuals,
                amp=amp,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
            )
        save_checkpoint(args.output_dir / "last.pth", model, optimizer, scheduler, epoch, best_val, args, monitor_state)
        if global_improved:
            save_checkpoint(args.output_dir / "best.pth", model, optimizer, scheduler, epoch, best_val, args, monitor_state)
            print(f"Saved new best checkpoint: {best_val:.6f}")


if __name__ == "__main__":
    main()
