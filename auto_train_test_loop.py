"""
YOLOv8 Iterative Automated Train-and-Test Continuous Loop (Target: >90% Accuracy)
================================================================================
This script repeatedly trains and tests YOLOv8 in successive rounds:
1. Evaluates baseline accuracy on the validation/test set.
2. Trains for N epochs using high-accuracy hyperparameters (AdamW, Cosine LR, Augmentation).
3. Evaluates overall accuracy (Precision, Recall, F1, mAP50, mAP50-95) after each round.
4. If a new round achieves higher accuracy, saves it as the new best model.
5. Dynamically adapts learning rate and focus across rounds to break plateaus.
6. Halts when the >90% target is reached or max rounds complete.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
import csv
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = str(ROOT_DIR / "data.yaml")
WEIGHTS_DIR = ROOT_DIR / "weights"
DEST_BEST_WEIGHTS = WEIGHTS_DIR / "best_trash_yolov8m.pt"
HISTORY_CSV = ROOT_DIR / "runs" / "detect" / "train_test_loop_history.csv"

def get_compute_device() -> str:
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"[Device] Using NVIDIA CUDA GPU: {gpu}")
        return "0"
    print("[Device] WARNING: CUDA GPU not available, running on CPU.")
    return "cpu"

def evaluate_model(model_path: str, data_yaml: str = DATA_YAML, imgsz: int = 640, device: str = "0", use_tta: bool = False):
    """Run validation and return metric dictionary."""
    model = YOLO(model_path)
    res = model.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        conf=0.25,
        iou=0.6,
        augment=use_tta,
        device=device,
        plots=False,
        verbose=False
    )
    p = float(res.box.mp)
    r = float(res.box.mr)
    map50 = float(res.box.map50)
    map_all = float(res.box.map)
    f1 = float(2 * (p * r) / (p + r + 1e-8))
    
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": map50,
        "map50_95": map_all,
        "per_class_p": res.box.p,
        "per_class_r": res.box.r,
        "per_class_map50": res.box.ap50,
        "names": res.names
    }

def print_metrics_banner(title: str, metrics: dict):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)
    print(f"  Overall Precision (P)    : {metrics['precision']*100:6.2f}%")
    print(f"  Overall Recall    (R)    : {metrics['recall']*100:6.2f}%")
    print(f"  Overall F1-Score         : {metrics['f1']*100:6.2f}%")
    print(f"  Overall mAP @ 0.50       : {metrics['map50']*100:6.2f}%")
    print(f"  Overall mAP @ 0.50:0.95  : {metrics['map50_95']*100:6.2f}%")
    print("=" * 70 + "\n")

def run_iterative_train_test_loop(
    rounds: int = 5,
    epochs_per_round: int = 25,
    batch_size: int = 12,
    imgsz: int = 640,
    target_metric: str = "map50",
    target_val: float = 0.90,
    base_weights: str = "weights/best_trash_yolov8m.pt"
):
    device = get_compute_device()
    WEIGHTS_DIR.mkdir(exist_ok=True)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Resolve initial weights
    current_best_weights = str(ROOT_DIR / base_weights) if not os.path.isabs(base_weights) else base_weights
    if not os.path.exists(current_best_weights):
        print(f"[Init] {current_best_weights} not found, falling back to yolov8m.pt")
        current_best_weights = "yolov8m.pt"

    print("\n" + "#" * 75)
    print("  🚀 YOLOv8 ITERATIVE TRAIN & TEST ENGINE (PUSH FOR >90% ACCURACY) 🚀")
    print("#" * 75)
    print(f"  Total Rounds          : {rounds}")
    print(f"  Epochs per Round      : {epochs_per_round} (Total potential epochs: {rounds * epochs_per_round})")
    print(f"  Batch Size            : {batch_size}")
    print(f"  Image Resolution      : {imgsz}x{imgsz}")
    print(f"  Initial Model Weights : {current_best_weights}")
    print(f"  Target Goal           : {target_metric.upper()} >= {target_val*100:.1f}%")
    print("#" * 75 + "\n")

    # Step 1: Initial Baseline Evaluation
    print("[Phase 0] Testing baseline accuracy before training...")
    baseline_metrics = evaluate_model(current_best_weights, imgsz=imgsz, device=device)
    print_metrics_banner("Baseline Accuracy (Pre-Training)", baseline_metrics)

    # Record baseline
    history = [{
        "round": 0,
        "type": "baseline",
        "epochs": 0,
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
        "f1": baseline_metrics["f1"],
        "map50": baseline_metrics["map50"],
        "map50_95": baseline_metrics["map50_95"],
        "weights": current_best_weights
    }]
    best_score = baseline_metrics[target_metric]
    print(f"[Baseline] Current best {target_metric.upper()} is {best_score*100:.2f}%. Commencing iterative training rounds...\n")

    # Loop rounds
    for r in range(1, rounds + 1):
        round_name = f"train_test_round_{r}"
        print("\n" + "=" * 75)
        print(f"  >>> STARTING ROUND {r}/{rounds}: TRAINING PHASE <<<")
        print("=" * 75)
        
        # Adaptive learning rate schedule across rounds
        # Round 1: fine-tune at 0.0005, subsequent rounds polish at lower LR
        lr0 = max(0.0001, 0.0006 * (0.85 ** (r - 1)))
        lrf = 0.01

        print(f"  Base weights for Round {r} : {current_best_weights}")
        print(f"  Learning Rate (lr0)        : {lr0:.6f} (Cosine Annealing to {lr0*lrf:.7f})")
        print(f"  Epochs in this round       : {epochs_per_round}")
        print("=" * 75 + "\n")

        model = YOLO(current_best_weights)
        train_results = model.train(
            data=DATA_YAML,
            epochs=epochs_per_round,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=2,
            project=str(ROOT_DIR / "runs" / "detect"),
            name=round_name,
            exist_ok=True,
            
            # Loss tuning to aggressively push accuracy & classification
            box=7.5,
            cls=1.5,
            dfl=1.5,
            
            # Optimizer & Regularization
            optimizer="AdamW",
            lr0=lr0,
            lrf=lrf,
            cos_lr=True,
            warmup_epochs=2,
            weight_decay=0.0005,
            dropout=0.15,
            
            # Heavy Data Augmentation to learn difficult views
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=12.0,
            translate=0.1,
            scale=0.5,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,
            close_mosaic=5,
            
            cache=False,
            verbose=True,
            save=True,
            plots=True
        )

        round_save_dir = Path(train_results.save_dir)
        round_best_pt = round_save_dir / "weights" / "best.pt"
        if not round_best_pt.exists():
            round_best_pt = round_save_dir / "weights" / "last.pt"

        # Step 2: Testing Phase (Evaluate Round)
        print("\n" + "=" * 75)
        print(f"  >>> COMPLETED ROUND {r}/{rounds}: TESTING / VALIDATION PHASE <<<")
        print("=" * 75)
        
        round_metrics = evaluate_model(str(round_best_pt), imgsz=imgsz, device=device)
        print_metrics_banner(f"Round {r} Accuracy Results", round_metrics)

        current_score = round_metrics[target_metric]
        improved = current_score > best_score

        history.append({
            "round": r,
            "type": f"round_{r}",
            "epochs": epochs_per_round,
            "precision": round_metrics["precision"],
            "recall": round_metrics["recall"],
            "f1": round_metrics["f1"],
            "map50": round_metrics["map50"],
            "map50_95": round_metrics["map50_95"],
            "weights": str(round_best_pt)
        })

        # Save history to CSV
        fieldnames = ["round", "type", "epochs", "precision", "recall", "f1", "map50", "map50_95", "weights"]
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        if improved:
            gain = (current_score - best_score) * 100
            print(f"🌟 SUCCESS! Round {r} IMPROVED {target_metric.upper()} by +{gain:.2f}% (Now {current_score*100:.2f}%)!")
            best_score = current_score
            current_best_weights = str(round_best_pt)
            
            # Copy to global best location
            shutil.copy2(round_best_pt, DEST_BEST_WEIGHTS)
            print(f"  Updated global best model: {DEST_BEST_WEIGHTS}")
        else:
            print(f"ℹ Round {r} score ({current_score*100:.2f}%) did not beat previous best ({best_score*100:.2f}%).")
            print(f"  Retaining previous best weights: {current_best_weights}")

        # Check if target achieved
        if best_score >= target_val:
            print("\n" + "🎉" * 35)
            print(f"  TARGET ACCURACY ACHIEVED! {target_metric.upper()} = {best_score*100:.2f}% (Target: {target_val*100:.1f}%)")
            print("🎉" * 35 + "\n")
            break

    # Final Summary Table
    print("\n" + "=" * 80)
    print("  🏆 ITERATIVE TRAIN & TEST COMPLETE - HISTORY SUMMARY TABLE 🏆")
    print("=" * 80)
    print(f"{'Round':<8} | {'Type':<12} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'mAP@50':<11} | {'mAP50-95':<11}")
    print("-" * 80)
    for h in history:
        print(f"{h['round']:<8} | {h['type']:<12} | {h['precision']*100:6.2f}%     | {h['recall']*100:6.2f}%     | {h['f1']*100:6.2f}%     | {h['map50']*100:6.2f}%     | {h['map50_95']*100:6.2f}%")
    print("=" * 80)
    print(f"  Final Best Model Saved At : {DEST_BEST_WEIGHTS}")
    print(f"  Full History Saved At     : {HISTORY_CSV}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Iterative YOLOv8 Train & Test Engine")
    parser.add_argument("--rounds", type=int, default=5, help="Number of train-and-test rounds (default: 5)")
    parser.add_argument("--epochs-per-round", type=int, default=25, help="Epochs per training round (default: 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--target", type=float, default=0.90, help="Target accuracy goal (default: 0.90)")
    parser.add_argument("--metric", type=str, default="precision", choices=["precision", "map50", "f1"], help="Target metric (default: precision)")
    parser.add_argument("--weights", type=str, default="weights/best_trash_yolov8m.pt", help="Starting weights")

    args = parser.parse_args()

    run_iterative_train_test_loop(
        rounds=args.rounds,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch,
        imgsz=args.imgsz,
        target_metric=args.metric,
        target_val=args.target,
        base_weights=args.weights
    )
