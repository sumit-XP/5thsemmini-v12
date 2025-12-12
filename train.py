from __future__ import annotations
import os
import time
from typing import List

import torch
from torch import optim
from torch import amp

from config import TRAINING_CONFIG as C
from models.gc_res2_yolov3 import GCRes2YOLOv3
from utils.data_loader import create_dataloader
from utils.loss import SimpleYOLOLoss


def train_one_epoch(model: torch.nn.Module, dl, optimizer, scaler: amp.GradScaler, device: str) -> float:
    model.train()
    loss_fn = SimpleYOLOLoss(num_classes=C.num_classes)
    epoch_loss = 0.0
    device_type = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    mixed_enabled = C.mixed_precision and device_type == "cuda"
    accum_steps = C.gradient_accumulation_steps
    print(f"Starting training loop with {len(dl)} batches (gradient accumulation: {accum_steps} steps)...")
    pin_mem = C.pin_memory and device_type == "cuda" and torch.cuda.is_available()
    last_end = time.perf_counter()
    
    for batch_idx, (images, targets) in enumerate(dl):
        t_batch_start = time.perf_counter()
        data_wait = t_batch_start - last_end
        images = images.to(device, non_blocking=pin_mem)
        
        # Only zero gradients at the start of accumulation cycle
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with amp.autocast(device_type=device_type, enabled=mixed_enabled):
            preds = model(images)
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        with amp.autocast(device_type=device_type, enabled=mixed_enabled):
            loss = loss_fn(preds, targets)
            # Scale loss by accumulation steps to maintain gradient magnitude
            loss = loss / accum_steps
        if device_type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        scaler.scale(loss).backward()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        # Only step optimizer every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dl):
            scaler.step(optimizer)
            scaler.update()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        
        epoch_loss += loss.item() * accum_steps  # Undo the scaling for logging
        last_end = t4
        
        if batch_idx < 5 or batch_idx % 10 == 0:
            step_marker = "→STEP" if (batch_idx + 1) % accum_steps == 0 else ""
            print(
                f"  Batch {batch_idx}/{len(dl)} {step_marker} - loss: {loss.item() * accum_steps:.4f} | "
                f"data_wait {data_wait*1000:.1f}ms, H2D {(t0 - t_batch_start)*1000:.1f}ms, "
                f"fwd {(t1 - t0)*1000:.1f}ms, loss_fn {(t2 - t1)*1000:.1f}ms, "
                f"bwd {(t3 - t2)*1000:.1f}ms, step {(t4 - t3)*1000:.1f}ms"
            )
    return epoch_loss / max(1, len(dl))


def main() -> None:
    os.makedirs(C.save_dir, exist_ok=True)
    device = C.device
    
    # Verify GPU is available
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() returned False. Check your GPU drivers and PyTorch installation.")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print(f"Using device: {device}")

    # Data
    print("\n[1/5] Loading dataset...")
    pin_mem = C.pin_memory and device.startswith("cuda") and torch.cuda.is_available()
    print(f"  - Dataset root: {C.dataset_root}")
    print(f"  - Image size: {C.img_size}")
    print(f"  - Batch size: {C.batch_size}")
    print(f"  - Num workers: {C.num_workers}")
    print(f"  - Pin memory: {pin_mem}")
    print(f"  - Mosaic augmentation: {C.use_mosaic}")
    train_ds, train_dl = create_dataloader(C.dataset_root, "train", C.img_size, C.batch_size, C.num_workers, pin_mem, C.use_mosaic)
    print(f"✓ Dataset loaded: {len(train_ds)} images, {len(train_dl)} batches")

    # Model
    print("\n[2/5] Initializing model...")
    print(f"  - Model: GCRes2YOLOv3")
    print(f"  - Num classes: {C.num_classes}")
    model = GCRes2YOLOv3(num_classes=C.num_classes).to(device)
    print(f"✓ Model initialized and moved to {device}")
    if C.use_compile and hasattr(torch, "compile") and device.startswith("cuda") and torch.cuda.is_available():
        print("\n[3/5] Compiling model (this may take several minutes on first run)...")
        model = torch.compile(model)  # type: ignore
        print("✓ Model compiled")
    else:
        print("\n[3/5] Skipping model compilation")

    print("\n[4/5] Setting up optimizer and scaler...")
    optimizer = optim.SGD(model.parameters(), lr=C.learning_rate, momentum=C.momentum, weight_decay=C.weight_decay)
    scaler = amp.GradScaler(enabled=C.mixed_precision and torch.cuda.is_available())
    effective_batch_size = C.batch_size * C.gradient_accumulation_steps
    print(f"✓ Optimizer: SGD (lr={C.learning_rate}, momentum={C.momentum})")
    print(f"✓ Mixed precision: {C.mixed_precision and torch.cuda.is_available()}")
    print(f"✓ Gradient accumulation: {C.gradient_accumulation_steps} steps (effective batch size: {effective_batch_size})")

    print(f"\n[5/5] Starting training for {C.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, C.epochs + 1):
        print(f"\nEpoch {epoch}/{C.epochs}")
        loss = train_one_epoch(model, train_dl, optimizer, scaler, device)
        print(f"Epoch {epoch}/{C.epochs} - Average loss: {loss:.4f}")
        if epoch % C.save_every == 0:
            ckpt = os.path.join(C.save_dir, f"epoch_{epoch}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
            print(f"  ✓ Checkpoint saved: {ckpt}")

    # Save final
    print("\n" + "=" * 60)
    print("Training complete! Saving final model...")
    ckpt = os.path.join(C.save_dir, "final.pth")
    torch.save({"model": model.state_dict(), "epoch": C.epochs}, ckpt)
    print(f"✓ Final model saved: {ckpt}")


if __name__ == "__main__":
    main()
