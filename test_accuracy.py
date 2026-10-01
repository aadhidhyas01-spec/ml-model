"""
Comprehensive Accuracy & Benchmark Tester for YOLOv8 Trash Detection
====================================================================
Tests overall accuracy, mAP50, mAP50-95, precision, recall, and extracts
the exact confidence thresholds where detection accuracy exceeds 90%, 93%, and 95%.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

def test_overall_accuracy(
    weights="weights/best_trash_yolov8m.pt",
    data_yaml="data.yaml",
    imgsz=640,
    conf=0.25,
    iou=0.6,
    use_tta=False
):
    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "0" else "CPU"

    print("\n" + "=" * 78)
    print("  🚀 YOLOv8 OVERALL ACCURACY & BENCHMARK EVALUATOR 🚀")
    print("=" * 78)
    print(f"  Model Weights : {weights}")
    print(f"  Dataset Config: {data_yaml}")
    print(f"  Resolution    : {imgsz}x{imgsz}")
    print(f"  Confidence    : {conf}")
    print(f"  IoU Threshold : {iou}")
    print(f"  TTA (Augment) : {use_tta}")
    print(f"  Hardware      : {device_name} (CUDA: {device == '0'})")
    print("=" * 78 + "\n")

    model = YOLO(weights)
    results = model.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=use_tta,
        device=device,
        plots=False,
        verbose=False
    )

    mp = float(results.box.mp)
    mr = float(results.box.mr)
    map50 = float(results.box.map50)
    map_all = float(results.box.map)
    f1 = float(2 * (mp * mr) / (mp + mr + 1e-8))

    print("=" * 78)
    print("  OVERALL ACCURACY METRICS (CONF = 0.25)")
    print("=" * 78)
    print(f"  Detection Precision (P)  : {mp*100:6.2f}%   (When trash is detected, how often is it right?)")
    print(f"  Detection Recall (R)     : {mr*100:6.2f}%   (Percentage of total trash items detected)")
    print(f"  Harmonic F1-Score        : {f1*100:6.2f}%   (Balanced accuracy combining Precision & Recall)")
    print(f"  mAP @ 0.50 (Standard)    : {map50*100:6.2f}%   (Mean Average Precision across all categories)")
    print(f"  mAP @ 0.50:0.95 (Strict) : {map_all*100:6.2f}%   (Strict multi-IoU challenge benchmark)")
    print("=" * 78)

    print("\n" + "=" * 82)
    print(f"{'Class Name':<22} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'Status':<16}")
    print("-" * 82)

    names = results.names
    ap50s = results.box.ap50
    ps = results.box.p
    rs = results.box.r

    for c_id, c_name in names.items():
        p_val = float(ps[c_id]) if c_id < len(ps) else 0.0
        r_val = float(rs[c_id]) if c_id < len(rs) else 0.0
        a50_val = float(ap50s[c_id]) if c_id < len(ap50s) else 0.0
        
        if p_val >= 0.85:
            status = "EXCELLENT (>85%)"
        elif p_val >= 0.70:
            status = "GOOD (>70%)"
        else:
            status = "MODERATE (<70%)"

        print(f"{c_name:<22} | {p_val*100:6.2f}%    | {r_val*100:6.2f}%    | {a50_val*100:6.2f}%    | {status:<16}")

    print("=" * 82)

    # Fast Precision Curve Analysis across all confidence thresholds
    px = results.box.px
    p_curve = results.box.p_curve
    r_curve = results.box.r_curve
    mean_p = np.mean(p_curve, axis=0)
    mean_r = np.mean(r_curve, axis=0)

    print("\n" + "=" * 78)
    print("  CONFIDENCE THRESHOLD VS DETECTION ACCURACY (PRECISION)")
    print("=" * 78)
    print(f"{'Confidence Threshold':<22} | {'Precision (Accuracy)':<22} | {'Recall':<15} | {'Target Status':<15}")
    print("-" * 78)

    for conf_val in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        idx = min(int(conf_val * len(px)), len(px) - 1)
        p_at_c = mean_p[idx] * 100
        r_at_c = mean_r[idx] * 100
        if p_at_c >= 95.0:
            status_tag = ">>> EXTREME (>95%)!"
        elif p_at_c >= 90.0:
            status_tag = ">>> ACHIEVED (>90%)!"
        else:
            status_tag = "Normal"
        print(f"Conf >= {conf_val:4.2f}            | {p_at_c:6.2f}%                | {r_at_c:6.2f}%        | {status_tag}")

    idx_95 = np.where(mean_p >= 0.95)[0]
    idx_90 = np.where(mean_p >= 0.90)[0]
    if len(idx_95) > 0:
        first_idx = idx_95[0]
        c_star = px[first_idx]
        print("=" * 78)
        print(f"🎯 95%+ ACCURACY DEPLOYMENT THRESHOLD: conf >= {c_star:.2f}")
        print(f"   At confidence >= {c_star:.2f}, overall model detection precision is {mean_p[first_idx]*100:.2f}% (>95%)!")
    elif len(idx_90) > 0:
        first_idx = idx_90[0]
        c_star = px[first_idx]
        print("=" * 78)
        print(f"🎯 90%+ ACCURACY DEPLOYMENT THRESHOLD: conf >= {c_star:.2f}")
        print(f"   At confidence >= {c_star:.2f}, overall model detection precision is {mean_p[first_idx]*100:.2f}% (>90%)!")
    print("=" * 78 + "\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test overall accuracy of YOLOv8 trash model")
    parser.add_argument("--weights", type=str, default="weights/best_trash_yolov8m.pt", help="Path to weights file")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")

    args = parser.parse_args()
    test_overall_accuracy(
        weights=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        use_tta=args.tta
    )
