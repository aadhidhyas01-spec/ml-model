"""
YOLOv8 Maximum Accuracy Training Script
Dataset: yolov8-trash-detections (10 classes)
Strategy: YOLOv8m + augmentation + optimized for Windows
NOTE: All code inside if __name__ == "__main__" for Windows multiprocessing.
"""

import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    # --- Paths ---
    SCRIPT_DIR  = Path(__file__).resolve().parent
    DATA_YAML   = str(SCRIPT_DIR / "data.yaml")
    PROJECT_DIR = str(SCRIPT_DIR / "runs" / "detect")
    EXPERIMENT  = "trash_max_accuracy"

    # --- Detect best device ---
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        device = "0"
        print(f"[OK] CUDA GPU detected: {gpu}")
        batch_size = 16
        workers    = 2   # Low workers on Windows to avoid stalls
    else:
        device     = "cpu"
        batch_size = 8
        workers    = 0
        print("[WARN] No GPU found. Training on CPU.")

    # --- Validate dataset ---
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"data.yaml not found at '{DATA_YAML}'")

    # --- Model selection: yolov8m ---
    BASE_MODEL = str(SCRIPT_DIR / "yolov8m.pt")
    if not Path(BASE_MODEL).exists():
        BASE_MODEL = "yolov8m.pt"

    print("\n" + "=" * 65)
    print("  YOLOv8m - Maximum Accuracy Training (Optimized)")
    print("=" * 65)
    print(f"  Model       : {BASE_MODEL}")
    print(f"  Data YAML   : {DATA_YAML}")
    print(f"  Device      : {device}  |  Batch: {batch_size}  |  Workers: {workers}")
    print(f"  Output      : {PROJECT_DIR}/{EXPERIMENT}")
    print("=" * 65 + "\n")

    # --- Load model ---
    model = YOLO(BASE_MODEL)

    # --- TRAINING ---
    print("[1/3] Starting training...\n")
    results = model.train(
        # Core
        data        = DATA_YAML,
        epochs      = 150,
        batch       = batch_size,
        imgsz       = 640,
        device      = device,
        workers     = workers,
        # Output
        project     = PROJECT_DIR,
        name        = EXPERIMENT,
        exist_ok    = True,
        save_period = 10,
        # LR schedule
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        cos_lr        = True,
        warmup_epochs = 5,
        # Regularisation
        weight_decay  = 0.0005,
        dropout       = 0.1,
        # Early stopping
        patience      = 30,
        # Augmentation
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 10.0,
        translate     = 0.1,
        scale         = 0.5,
        shear         = 2.0,
        perspective   = 0.0005,
        flipud        = 0.05,
        fliplr        = 0.5,
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.3,
        # NO disk cache - avoids Windows I/O stalls
        cache         = False,
        verbose       = True,
        plots         = True,
    )

    # --- Validation ---
    print("\n[2/3] Running final validation on best weights...\n")
    val_results = model.val(
        data   = DATA_YAML,
        split  = "val",
        imgsz  = 640,
        device = device,
        plots  = True,
    )

    # --- Copy best weights to /weights/ ---
    save_dir    = Path(results.save_dir)
    best_pt     = save_dir / "weights" / "best.pt"
    dest_folder = SCRIPT_DIR / "weights"
    dest_folder.mkdir(exist_ok=True)
    dest_best   = dest_folder / "best_trash_yolov8m.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, dest_best)
        print(f"\n[3/3] Best weights copied to: {dest_best}")
    else:
        print(f"\n[3/3] WARNING: Could not find best.pt at {best_pt}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE - RESULTS SUMMARY")
    print("=" * 65)
    try:
        maps = val_results.box
        print(f"  mAP50       : {maps.map50:.4f}  ({maps.map50*100:.2f}%)")
        print(f"  mAP50-95    : {maps.map:.4f}  ({maps.map*100:.2f}%)")
        print(f"  Precision   : {maps.mp:.4f}  ({maps.mp*100:.2f}%)")
        print(f"  Recall      : {maps.mr:.4f}  ({maps.mr*100:.2f}%)")
    except Exception:
        pass
    print(f"\n  Best weights  : {dest_best}")
    print(f"  Run directory : {save_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
