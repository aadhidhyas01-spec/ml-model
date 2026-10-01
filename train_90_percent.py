"""
YOLOv8 Extreme Accuracy Training Script (Targeting >90%)
"""

import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    SCRIPT_DIR  = Path(__file__).resolve().parent
    DATA_YAML   = str(SCRIPT_DIR / "data.yaml")
    PROJECT_DIR = str(SCRIPT_DIR / "runs" / "detect")
    EXPERIMENT  = "trash_push_90"
    
    # Resume from our best weights so far (52.7%)
    BEST_WEIGHTS = str(SCRIPT_DIR / "weights" / "best_trash_yolov8m.pt")
    if not os.path.exists(BEST_WEIGHTS):
        print("ERROR: Previous best weights not found. Starting from scratch with yolov8m.pt")
        BEST_WEIGHTS = "yolov8m.pt"
        
    device = "0" if torch.cuda.is_available() else "cpu"
    
    # Key change 1: Increased image size from 640 to 800 (Helps with tiny items like bottle caps)
    # Key change 2: Reduced batch size to 8 so the larger images fit in your 8GB RTX 4060 VRAM
    img_size = 800
    batch_size = 8
    workers = 2

    print("\n" + "=" * 65)
    print("  🚀 PUSHING FOR 90%+ ACCURACY 🚀")
    print("=" * 65)
    print(f"  Resuming Model : {BEST_WEIGHTS}")
    print(f"  Image Size     : {img_size}x{img_size} (Upscaled for tiny objects)")
    print(f"  Device         : {device}  |  Batch: {batch_size}  |  Workers: {workers}")
    print("=" * 65 + "\n")

    model = YOLO(BEST_WEIGHTS)

    print("[1/3] Starting extreme training phase...\n")
    results = model.train(
        data        = DATA_YAML,
        epochs      = 300,      # Train for a massive 300 epochs
        batch       = batch_size,
        imgsz       = img_size, 
        device      = device,
        workers     = workers,
        project     = PROJECT_DIR,
        name        = EXPERIMENT,
        exist_ok    = True,
        
        # Fine-tuning parameters
        optimizer     = "AdamW",
        lr0           = 0.0005,  # Lower learning rate since we are resuming
        lrf           = 0.001,
        cos_lr        = True,
        warmup_epochs = 3,
        
        weight_decay  = 0.0005,
        dropout       = 0.2,     # Higher dropout to prevent overfitting
        
        patience      = 60,      # Wait longer before early stopping
        
        # Heavy Augmentation
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 15.0,
        translate     = 0.1,
        scale         = 0.6,
        mosaic        = 1.0,
        mixup         = 0.2,
        copy_paste    = 0.3,
        
        cache         = False,
        verbose       = True,
    )

    print("\n[2/3] Running final validation on best weights...\n")
    val_results = model.val(
        data   = DATA_YAML,
        split  = "val",
        imgsz  = img_size,
        device = device,
    )

    save_dir    = Path(results.save_dir)
    best_pt     = save_dir / "weights" / "best.pt"
    dest_best   = SCRIPT_DIR / "weights" / "best_trash_yolov8m.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, dest_best)
        print(f"\n[3/3] Best weights copied to: {dest_best}")

    print("\n" + "=" * 65)
    print("  EXTREME TRAINING COMPLETE - FINAL RESULTS")
    print("=" * 65)
    try:
        maps = val_results.box
        print(f"  mAP50       : {maps.map50:.4f}  ({maps.map50*100:.2f}%)")
        print(f"  Precision   : {maps.mp:.4f}  ({maps.mp*100:.2f}%)")
    except Exception:
        pass
    print("=" * 65)

if __name__ == "__main__":
    main()
