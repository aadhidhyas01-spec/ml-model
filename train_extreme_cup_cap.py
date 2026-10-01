"""
YOLOv8 Extreme Accuracy Training Script: Drinking Cups & Bottle Caps
=====================================================================
Optimized for:
1. High Resolution (800x800) -> Delivers crisp edge features for small bottle caps & cup rims
2. Close Mosaic (last 15 epochs) -> Prevents shrinking bottle caps into tiny multi-image corners
3. High Box Loss (box=8.5) -> Guarantees tight bounding coordinates for robotic arm claw alignment
4. AdamW + Cosine LR Schedule -> Smooth convergence to global minimum on RTX 4060
"""

import os
import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    SCRIPT_DIR  = Path(__file__).resolve().parent
    DATA_YAML   = str(SCRIPT_DIR / "data.yaml")
    PROJECT_DIR = str(SCRIPT_DIR / "runs" / "detect")
    EXPERIMENT  = "trash_extreme_cup_cap"

    # Device & Hardware Optimization
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        device = "0"
        print(f"[OK] NVIDIA GPU Acceleration Enabled: {gpu}")
        batch_size = 8    # 8 images @ 800x800 fits comfortably in 8GB RTX 4060 VRAM
        workers = 2       # Stable for Windows multiprocessing
    else:
        device = "cpu"
        batch_size = 4
        workers = 0
        print("[WARN] No GPU found. Training on CPU (will be slower).")

    # Image Resolution: 800x800 provides the pixel density needed for tiny bottle caps
    img_size = 800

    # Base model: Start from pretrained yolov8m.pt for deep transfer learning
    BASE_MODEL = str(SCRIPT_DIR / "yolov8m.pt")
    if not Path(BASE_MODEL).exists():
        BASE_MODEL = "yolov8m.pt"

    print("\n" + "=" * 70)
    print("  🚀 YOLOv8 EXTREME TRAINING: CUPS & BOTTLE CAPS OPTIMIZATION 🚀")
    print("=" * 70)
    print(f"  Base Architecture : {BASE_MODEL}")
    print(f"  Dataset Config    : {DATA_YAML}")
    print(f"  Resolution        : {img_size}x{img_size} (High-Res for small objects)")
    print(f"  Hardware Device   : {device} (Batch: {batch_size}, Workers: {workers})")
    print(f"  Target Focus      : Plastic/Paper Cups & Plastic Bottle Caps")
    print(f"  Output Directory  : {PROJECT_DIR}/{EXPERIMENT}")
    print("=" * 70 + "\n")

    model = YOLO(BASE_MODEL)

    print("[1/3] Launching extreme precision training phase...\n")
    results = model.train(
        # Core Parameters
        data         = DATA_YAML,
        epochs       = 150,
        batch        = batch_size,
        imgsz        = img_size,
        device       = device,
        workers      = workers,
        project      = PROJECT_DIR,
        name         = EXPERIMENT,
        exist_ok     = True,
        save_period  = 10,

        # Learning Rate Schedule (Cosine Annealing)
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.01,
        cos_lr       = True,
        warmup_epochs= 4,

        # Loss Functions Optimized for Robotic Grasping & Small Objects
        box          = 8.5,       # Higher weight on bounding box tight fit
        cls          = 1.2,       # Boosted classification weight to penalize inter-class confusion
        dfl          = 1.5,       # Distribution focal loss
        label_smoothing = 0.05,   # Smooth noisy boundaries

        # Regularization & Early Stopping
        weight_decay = 0.0005,
        dropout      = 0.15,
        patience     = 40,

        # Augmentation Strategy for Cups & Caps
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        degrees      = 12.0,      # Slight rotation for cups on floor
        translate    = 0.1,
        scale        = 0.7,       # Scale up and down for distance invariance
        shear        = 2.0,
        flipud       = 0.0,       # Trash is right side or on side, rarely upside down in air
        fliplr       = 0.5,
        mosaic       = 1.0,
        mixup        = 0.15,
        copy_paste   = 0.25,

        # CRITICAL: Turn OFF mosaic during last 15 epochs so model learns clean, uncropped caps & cups
        close_mosaic = 15,

        cache        = False,     # Avoids Windows disk lock stalls
        verbose      = True,
        plots        = True,
    )

    print("\n[2/3] Running final rigorous evaluation on validation set...\n")
    val_results = model.val(
        data   = DATA_YAML,
        split  = "val",
        imgsz  = img_size,
        device = device,
        plots  = True,
    )

    # Automatically deploy best weights
    save_dir  = Path(results.save_dir)
    best_pt   = save_dir / "weights" / "best.pt"
    dest_dir  = SCRIPT_DIR / "weights"
    dest_dir.mkdir(exist_ok=True)
    dest_best = dest_dir / "best_trash_yolov8m.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, dest_best)
        print(f"\n[3/3] ✅ SUCCESS: Best model weights deployed to: {dest_best}")
    else:
        print(f"\n[3/3] ⚠️ Warning: best.pt not found at {best_pt}")

    # Summary Display
    print("\n" + "=" * 70)
    print("  EXTREME TRAINING COMPLETE - FINAL METRICS")
    print("=" * 70)
    try:
        maps = val_results.box
        print(f"  Overall Precision (P)    : {maps.mp*100:6.2f}%")
        print(f"  Overall Recall    (R)    : {maps.mr*100:6.2f}%")
        print(f"  mAP @ 0.50               : {maps.map50*100:6.2f}%")
        print(f"  mAP @ 0.50:0.95 (Strict) : {maps.map*100:6.2f}%")
        print("-" * 70)
        print(f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10}")
        print("-" * 70)
        names = val_results.names
        for cid, cname in names.items():
            p_val = float(maps.p[cid]) if cid < len(maps.p) else 0.0
            r_val = float(maps.r[cid]) if cid < len(maps.r) else 0.0
            a50 = float(maps.ap50[cid]) if cid < len(maps.ap50) else 0.0
            marker = " <<< TARGET" if cname in ['cup', 'plastic bottle cap'] else ""
            print(f"{cname:<22} | {p_val*100:6.2f}%    | {r_val*100:6.2f}%    | {a50*100:6.2f}%{marker}")
    except Exception as e:
        print(f"Metrics summary error: {e}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
