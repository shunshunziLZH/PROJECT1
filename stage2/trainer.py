from __future__ import annotations

import math
import random
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Sampler

from .config import save_config
from .data import build_data_manifest, dataset_from_config
from .losses import (
    BankLossWeights,
    BankPretrainingLoss,
    build_value_invariance_pair,
    psnr,
    ssim_metric,
)
from .models import BankPretrainingModel
from .runtime import build_stage2_model, load_module_state_dicts, module_state_dicts
from .stage1 import load_frozen_stage1, predict_tb
from .utils import (
    atomic_torch_save,
    load_checkpoint,
    resolve_device,
    seed_worker,
    set_seed,
    trainable_parameter_counts,
)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _make_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    worker_generator = torch.Generator().manual_seed(seed + 1009)
    sampler = None
    if shuffle:
        sampler = GroupedPatchSampler(dataset, torch.Generator().manual_seed(seed))
    arguments: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": worker_generator,
    }
    if num_workers > 0:
        # Recreate workers every epoch so their seeded Python/NumPy RNG streams
        # are reproducible after a checkpoint resume.
        arguments["persistent_workers"] = False
        arguments["prefetch_factor"] = 2
    return DataLoader(**arguments)


class GroupedPatchSampler(Sampler[int]):
    """Shuffle images while keeping their patch indices adjacent for decode reuse."""

    def __init__(self, dataset: Any, generator: torch.Generator):
        if not hasattr(dataset, "samples") or not hasattr(dataset, "patches_per_image"):
            raise TypeError("GroupedPatchSampler requires an AlignedBankDataset")
        self.dataset = dataset
        self.generator = generator

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterable[int]:
        image_order = torch.randperm(len(self.dataset.samples), generator=self.generator).tolist()
        patches = int(self.dataset.patches_per_image)
        for image_index in image_order:
            patch_order = torch.randperm(patches, generator=self.generator).tolist()
            for patch_index in patch_order:
                yield image_index * patches + patch_index


def create_dataloaders(config: Mapping[str, Any], device: torch.device) -> Tuple[DataLoader, DataLoader]:
    training = config["training"]
    train_dataset = dataset_from_config(config, "train")
    val_dataset = dataset_from_config(config, "val")
    train_loader = _make_loader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        shuffle=True,
        seed=int(config["seed"]),
        device=device,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        shuffle=False,
        seed=int(config["seed"]) + 1,
        device=device,
    )
    return train_loader, val_loader


def _criterion_from_config(config: Mapping[str, Any]) -> BankPretrainingLoss:
    loss = config["loss"]
    return BankPretrainingLoss(
        weights=BankLossWeights(
            restore=float(loss["lambda_restore"]),
            key=float(loss["lambda_key"]),
            align=float(loss["lambda_align"]),
            query=float(loss["lambda_query"]),
            invariance=float(loss["lambda_inv"]),
            ranking=float(loss["lambda_rank"]),
        ),
        lambda_grad=float(loss["lambda_grad"]),
        lambda_ssim=float(loss["lambda_ssim"]),
        lambda_fft=float(loss["lambda_fft"]),
        tau_physical=float(loss["tau_physical"]),
        tau_embedding=float(loss["tau_embedding"]),
        ranking_margin=float(loss["rank_margin"]),
        physical_relation_size=int(loss["physical_relation_size"]),
    )


def _move_images(batch: Mapping[str, Any], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(batch[field].to(device, non_blocking=True) for field in ("I", "J", "T", "B"))


def _finite_loss(loss: torch.Tensor, components: Mapping[str, torch.Tensor]) -> None:
    if not torch.isfinite(loss):
        detail = {name: float(value.detach()) for name, value in components.items()}
        raise FloatingPointError(f"Non-finite Stage 2 loss: {detail}")


def _accumulate(target: MutableMapping[str, float], values: Mapping[str, float], count: int) -> None:
    target["samples"] = target.get("samples", 0.0) + count
    for name, value in values.items():
        target[name] = target.get(name, 0.0) + float(value) * count


def _averages(sums: Mapping[str, float]) -> Dict[str, float]:
    count = max(1.0, sums.get("samples", 0.0))
    return {name: value / count for name, value in sums.items() if name != "samples"}


def _validation_invariance_pair(
    J: torch.Tensor, T: torch.Tensor, B: torch.Tensor, batch_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    devices = []
    if J.device.type == "cuda":
        devices = [J.device.index if J.device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0x5A17 + batch_index)
        if devices:
            torch.cuda.manual_seed(0x5A17 + batch_index)
        return build_value_invariance_pair(J, T, B)


def _different_sample_permutation(sample_ids: Iterable[str], device: torch.device) -> Optional[torch.Tensor]:
    identifiers = list(sample_ids)
    count = len(identifiers)
    for shift in range(1, count):
        indices = [(index + shift) % count for index in range(count)]
        if all(identifiers[index] != identifiers[indices[index]] for index in range(count)):
            return torch.tensor(indices, device=device, dtype=torch.long)
    return None


def _capture_rng_state(train_loader: DataLoader) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "worker_generator": train_loader.generator.get_state() if train_loader.generator is not None else None,
    }
    sampler_generator = getattr(train_loader.sampler, "generator", None)
    state["sampler_generator"] = sampler_generator.get_state() if sampler_generator is not None else None
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any], train_loader: DataLoader) -> None:
    required = ("python", "numpy", "torch")
    missing = [name for name in required if name not in state]
    if missing:
        raise KeyError(f"Resume checkpoint RNG state is missing: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if train_loader.generator is not None and state.get("worker_generator") is not None:
        train_loader.generator.set_state(state["worker_generator"])
    sampler_generator = getattr(train_loader.sampler, "generator", None)
    if sampler_generator is not None and state.get("sampler_generator") is not None:
        sampler_generator.set_state(state["sampler_generator"])


def _validate_resume_configuration(current: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    comparisons = {
        "seed": (current.get("seed"), previous.get("seed")),
        "model": (current.get("model"), previous.get("model")),
        "data.patch_size": (current["data"].get("patch_size"), previous.get("data", {}).get("patch_size")),
        "data.patches_per_image": (
            current["data"].get("patches_per_image"), previous.get("data", {}).get("patches_per_image")
        ),
        "data.use_hflip": (current["data"].get("use_hflip"), previous.get("data", {}).get("use_hflip")),
        "data.use_vflip": (current["data"].get("use_vflip"), previous.get("data", {}).get("use_vflip")),
        "data.use_rot90": (current["data"].get("use_rot90"), previous.get("data", {}).get("use_rot90")),
        "stage1": (current.get("stage1"), previous.get("stage1")),
        "loss": (current.get("loss"), previous.get("loss")),
    }
    for field in (
        "epochs", "warmup_epochs", "batch_size", "learning_rate", "weight_decay",
        "gradient_clip_norm", "num_workers",
    ):
        comparisons[f"training.{field}"] = (
            current["training"].get(field), previous.get("training", {}).get(field)
        )
    mismatched = [name for name, (now, before) in comparisons.items() if now != before]
    if mismatched:
        raise ValueError(
            "Resume configuration is incompatible with the checkpoint: " + ", ".join(mismatched)
        )


def _validate_data_manifest(current: Mapping[str, Any], previous: Mapping[str, Any]) -> None:
    if previous.get("format_version") != 1 or previous.get("training_split_only") is not True:
        raise ValueError("Checkpoint data_manifest lacks valid training-only provenance")
    fields = ("train_ids", "val_ids", "train_fingerprint", "val_fingerprint")
    missing = [field for field in fields if field not in previous]
    if missing:
        raise KeyError(f"Checkpoint data_manifest is missing fields: {missing}")
    mismatched = [field for field in fields if current.get(field) != previous.get(field)]
    if mismatched:
        raise ValueError(
            "Current dataset/split does not match the Stage 2 checkpoint: " + ", ".join(mismatched)
        )


def run_epoch(
    model: BankPretrainingModel,
    loader: DataLoader,
    criterion: BankPretrainingLoss,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Any,
    stage1_predictor: Optional[nn.Module],
    enable_alignment: bool,
    enable_amp: bool,
    gradient_clip_norm: float,
    global_step: int,
    log_frequency: int,
    writer: Any = None,
) -> Tuple[Dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    if stage1_predictor is not None:
        stage1_predictor.eval()
    totals: Dict[str, float] = {}

    for batch_index, batch in enumerate(loader, start=1):
        I, J, T, B = _move_images(batch, device)
        J_aug, I_aug = (
            build_value_invariance_pair(J, T, B)
            if training
            else _validation_invariance_pair(J, T, B, batch_index)
        )
        T_pred: Optional[torch.Tensor] = None
        B_pred: Optional[torch.Tensor] = None
        if stage1_predictor is not None:
            T_pred, B_pred = predict_tb(stage1_predictor, I)

        if training:
            optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            with _autocast(device, enable_amp):
                outputs = model(I, J, T, B, T_pred, B_pred, J_aug, I_aug)
                wrong_restoration = None
                if criterion.weights.ranking > 0:
                    value = outputs["v"]
                    assert value is not None
                    permutation = (
                        _different_sample_permutation(batch["sample_id"], I.device)
                        if I.shape[0] > 1
                        else None
                    )
                    if permutation is not None:
                        wrong_restoration = model.temporary_restorer(I, value[permutation])
                    else:
                        warnings.warn(
                            "Ranking loss is enabled but this batch cannot form a complete cross-image "
                            "wrong-value assignment; its ranking term is zero. Increase batch size or "
                            "reduce patches_per_image.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                components = criterion(
                    outputs,
                    J,
                    wrong_restoration=wrong_restoration,
                    enable_alignment=enable_alignment,
                )
                loss = components["total"]
            _finite_loss(loss, components)

            if training:
                scaler.scale(loss).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

        with torch.no_grad():
            restored = outputs["J_temp"]
            assert restored is not None
            values = {name: float(value.detach()) for name, value in components.items()}
            values["psnr"] = float(psnr(restored, J))
            values["ssim"] = float(ssim_metric(restored, J))
            q_pred, q = outputs.get("q_pred"), outputs.get("q")
            values["query_cosine"] = (
                float(torch.nn.functional.cosine_similarity(q_pred, q, dim=-1).mean())
                if q_pred is not None and q is not None
                else math.nan
            )
        _accumulate(totals, values, I.shape[0])

        if training and writer is not None:
            for name, value in values.items():
                if math.isfinite(value):
                    writer.add_scalar(f"train_step/{name}", value, global_step)
            writer.add_scalar("train_step/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        if training and log_frequency > 0 and (batch_index % log_frequency == 0 or batch_index == len(loader)):
            print(
                f"  batch {batch_index}/{len(loader)} | total={values['total']:.5f} "
                f"key={values['key_rec']:.5f} restore={values['restore']:.5f}"
            )
    return _averages(totals), global_step


@torch.no_grad()
def value_usage_diagnostic(
    model: BankPretrainingModel, loader: DataLoader, device: torch.device, max_batches: int = 4
) -> Dict[str, float]:
    model.eval()
    totals = {"correct": 0.0, "shuffled": 0.0, "zero": 0.0}
    counts = {"correct": 0, "shuffled": 0, "zero": 0}
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        I, J, _, _ = _move_images(batch, device)
        value = model.value_encoder(torch.cat((J, J - I), dim=1))
        predictions: Dict[str, torch.Tensor] = {
            "correct": model.temporary_restorer(I, value),
            "zero": model.temporary_restorer(I, torch.zeros_like(value)),
        }
        permutation = _different_sample_permutation(batch["sample_id"], I.device)
        if permutation is not None:
            predictions["shuffled"] = model.temporary_restorer(I, value[permutation])
        for name, prediction in predictions.items():
            totals[name] += float(psnr(prediction, J))
            counts[name] += 1
    result = {
        name: totals[name] / counts[name] if counts[name] else math.nan
        for name in totals
    }
    if counts["shuffled"] == 0:
        warnings.warn(
            "Value diagnostic could not form a cross-image shuffled batch; shuffled PSNR is unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif result["correct"] <= max(result["shuffled"], result["zero"]) + 1.0e-3:
        warnings.warn(
            "Temporary restorer value diagnostic failed: correct values do not outperform both shuffled "
            "and zero values. The restorer may be ignoring the value embedding.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def _save_training_checkpoint(
    path: Path,
    model: BankPretrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    *,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    train_loader: DataLoader,
    amp_enabled: bool,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        **module_state_dicts(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": dict(config),
        "data_manifest": dict(data_manifest),
        "metrics": dict(metrics),
        "rng_state": _capture_rng_state(train_loader),
        "amp_enabled": bool(amp_enabled),
    }
    atomic_torch_save(payload, path)


def train(config: Mapping[str, Any]) -> Dict[str, Any]:
    set_seed(int(config["seed"]), deterministic=True)
    device = resolve_device(str(config["device"]))
    training = config["training"]
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_manifest = build_data_manifest(config)
    train_loader, val_loader = create_dataloaders(config, device)
    model = build_stage2_model(config).to(device)
    stage1_predictor = load_frozen_stage1(config, device)
    criterion = _criterion_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"]), eta_min=1.0e-6
    )
    amp_enabled = bool(training["use_amp"] and device.type == "cuda")
    if bool(training["use_amp"]) and not amp_enabled:
        warnings.warn("AMP requested but only enabled for CUDA; continuing in float32.", RuntimeWarning)
    scaler = _make_scaler(amp_enabled)

    start_epoch, global_step = 1, 0
    best_warmup_loss, best_joint_loss = math.inf, math.inf
    previous_metrics: Mapping[str, Any] = {}
    resume = training.get("resume")
    if resume:
        checkpoint = load_checkpoint(resume, map_location=device)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("Stage 2 resume checkpoint must be a mapping")
        previous_config = checkpoint.get("config")
        if not isinstance(previous_config, Mapping):
            raise KeyError("Resume checkpoint is missing its configuration snapshot")
        _validate_resume_configuration(config, previous_config)
        previous_manifest = checkpoint.get("data_manifest")
        if not isinstance(previous_manifest, Mapping):
            raise KeyError("Resume checkpoint is missing data_manifest")
        _validate_data_manifest(data_manifest, previous_manifest)
        load_module_state_dicts(model, checkpoint)
        for name, object_to_load in (("optimizer", optimizer), ("scheduler", scheduler)):
            if name not in checkpoint:
                raise KeyError(f"Resume checkpoint is missing {name}")
            object_to_load.load_state_dict(checkpoint[name])
        if "scaler" not in checkpoint:
            raise KeyError("Resume checkpoint is missing scaler")
        if bool(checkpoint.get("amp_enabled", False)) == amp_enabled:
            scaler.load_state_dict(checkpoint["scaler"])
        else:
            warnings.warn(
                "AMP mode differs from the resume checkpoint; optimizer/model state was restored but "
                "a fresh GradScaler is being used.",
                RuntimeWarning,
            )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        previous_metrics = checkpoint.get("metrics", {})
        if not isinstance(previous_metrics, Mapping):
            raise TypeError("Resume checkpoint metrics must be a mapping")
        best_warmup_loss = float(previous_metrics.get("best_warmup_validation_total", math.inf))
        best_joint_loss = float(previous_metrics.get("best_joint_validation_total", math.inf))
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise KeyError("Resume checkpoint is missing rng_state")
        _restore_rng_state(rng_state, train_loader)
        print(f"Resuming Stage 2 from {resume} at epoch {start_epoch}.")

    save_config(config, output_dir / ("config_resume_snapshot.yaml" if resume else "config_snapshot.yaml"))
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output_dir / "tensorboard"), purge_step=global_step if resume else None)
    except (ImportError, ModuleNotFoundError):
        warnings.warn("TensorBoard is unavailable; scalar logging will use stdout only.", RuntimeWarning)

    counts = trainable_parameter_counts({name: getattr(model, name) for name in (
        "key_encoder", "key_decoder", "value_encoder", "temporary_restorer", "key_projector", "value_projector"
    )})
    print(f"Stage 2 device={device}, AMP={amp_enabled}, train patches={len(train_loader.dataset)}, val patches={len(val_loader.dataset)}")
    print("Trainable parameters: " + ", ".join(f"{name}={count:,}" for name, count in counts.items()))

    epochs = int(training["epochs"])
    warmup_epochs = int(training["warmup_epochs"])
    last_metrics: Dict[str, Any] = dict(previous_metrics)
    try:
        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            alignment = epoch > warmup_epochs
            print(f"Epoch {epoch}/{epochs} | phase={'joint' if alignment else 'warmup'} | lr={optimizer.param_groups[0]['lr']:.6g}")
            train_metrics, global_step = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                stage1_predictor=stage1_predictor,
                enable_alignment=alignment,
                enable_amp=amp_enabled,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                global_step=global_step,
                log_frequency=int(training["log_frequency"]),
                writer=writer,
            )
            validate = int(training["val_frequency"]) > 0 and (
                epoch % int(training["val_frequency"]) == 0 or epoch == epochs
            )
            val_metrics: Dict[str, float] = {}
            if validate:
                val_metrics, _ = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    optimizer=None,
                    scaler=scaler,
                    stage1_predictor=stage1_predictor,
                    enable_alignment=alignment,
                    enable_amp=amp_enabled,
                    gradient_clip_norm=0.0,
                    global_step=global_step,
                    log_frequency=0,
                )
            diagnostic: Dict[str, float] = {}
            diagnostic_frequency = int(training["diagnostic_frequency"])
            if validate and diagnostic_frequency > 0 and epoch % diagnostic_frequency == 0:
                diagnostic = value_usage_diagnostic(model, val_loader, device)

            scheduler.step()
            improved = False
            best_name = ""
            if validate:
                score = val_metrics["total"]
                if alignment and score < best_joint_loss:
                    best_joint_loss = score
                    improved, best_name = True, "best.pt"
                elif not alignment and score < best_warmup_loss:
                    best_warmup_loss = score
                    improved, best_name = True, "best_warmup.pt"
            last_metrics = {
                "train": train_metrics,
                "validation": val_metrics,
                "value_diagnostic_psnr": diagnostic,
                "best_validation_total": best_joint_loss,
                "best_joint_validation_total": best_joint_loss,
                "best_warmup_validation_total": best_warmup_loss,
                "epoch_seconds": time.time() - started,
                "stage1_query_enabled": stage1_predictor is not None,
                "parameter_counts": counts,
            }
            _save_training_checkpoint(
                output_dir / "last.pt", model, optimizer, scheduler, scaler,
                epoch=epoch, global_step=global_step, config=config, data_manifest=data_manifest,
                metrics=last_metrics, train_loader=train_loader, amp_enabled=amp_enabled,
            )
            if improved:
                _save_training_checkpoint(
                    output_dir / best_name, model, optimizer, scheduler, scaler,
                    epoch=epoch, global_step=global_step, config=config, data_manifest=data_manifest,
                    metrics=last_metrics, train_loader=train_loader, amp_enabled=amp_enabled,
                )
            if writer is not None:
                for split, metrics in (("train", train_metrics), ("validation", val_metrics)):
                    for name, value in metrics.items():
                        if math.isfinite(value):
                            writer.add_scalar(f"{split}/{name}", value, epoch)
                writer.add_scalar("epoch/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            print(f"  train={train_metrics}")
            if val_metrics:
                print(f"  validation={val_metrics}")
            if diagnostic:
                print(f"  value diagnostic PSNR={diagnostic}")
            print(
                f"  saved {output_dir / 'last.pt'}"
                + (f" and new {best_name} ({val_metrics['total']:.5f})" if improved else "")
            )
    finally:
        if writer is not None:
            writer.close()
    return last_metrics


__all__ = ["create_dataloaders", "run_epoch", "train", "value_usage_diagnostic"]
