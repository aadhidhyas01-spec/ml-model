"""
Interactive Real-World Cup & Bottle Cap Sample Collector
========================================================
Capture real photos of YOUR exact cup and bottle cap using ESP32-CAM or Webcam.
- Automatically detects bounding boxes with YOLOv8 or lets you click-and-drag.
- Saves directly to train/images and train/labels with correct class IDs:
    Class 10: cup
    Class  8: plastic bottle cap
Controls:
  [SPACE] or [C] : Capture CUP (Auto-detected & saved to training set)
  [B]             : Capture BOTTLE CAP (Auto-detected & saved to training set)
  [D]             : Draw custom bounding box manually (if auto-detect misses)
  [Q] or [ESC]   : Quit & show total captured count
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_IMG = ROOT_DIR / "train" / "images"
TRAIN_LBL = ROOT_DIR / "train" / "labels"
VALID_IMG = ROOT_DIR / "valid" / "images"
VALID_LBL = ROOT_DIR / "valid" / "labels"

TRAIN_IMG.mkdir(parents=True, exist_ok=True)
TRAIN_LBL.mkdir(parents=True, exist_ok=True)
VALID_IMG.mkdir(parents=True, exist_ok=True)
VALID_LBL.mkdir(parents=True, exist_ok=True)

# Drag-to-box state
drawing = False
ix, iy = -1, -1
rx, ry = -1, -1
manual_box = None

def mouse_callback(event, x, y, flags, param):
    global ix, iy, rx, ry, drawing, manual_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rx, ry = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rx, ry = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry = x, y
        x1, y1 = min(ix, rx), min(iy, ry)
        x2, y2 = max(ix, rx), max(iy, ry)
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            manual_box = (x1, y1, x2, y2)

def save_sample(frame, class_id: int, class_name: str, box_xywhn: tuple, sample_idx: int):
    """Save image and YOLO annotation line."""
    # 85% train, 15% valid
    is_val = (sample_idx % 6 == 0)
    dest_img_dir = VALID_IMG if is_val else TRAIN_IMG
    dest_lbl_dir = VALID_LBL if is_val else TRAIN_LBL

    timestamp = int(time.time() * 1000)
    stem = f"real_{class_name.replace(' ', '_')}_{timestamp}"
    img_path = dest_img_dir / f"{stem}.jpg"
    lbl_path = dest_lbl_dir / f"{stem}.txt"

    cv2.imwrite(str(img_path), frame)
    bx, by, bw, bh = box_xywhn
    with open(lbl_path, "w") as f:
        f.write(f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

    split_str = "VALID" if is_val else "TRAIN"
    print(f"✅ Saved [{class_name.upper()}] to {split_str}: {img_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Collect real cup and bottle cap training samples")
    parser.add_argument("--source", type=str, default="0", help="Camera source: '0' for laptop webcam, or http://<IP>:81/stream for ESP32-CAM")
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  📸 REAL-WORLD CUP & BOTTLE CAP SAMPLE COLLECTOR 📸")
    print("=" * 68)
    print(f"  Source: {args.source}")
    print("  Controls:")
    print("    [SPACE] or [C] -> Save CUP (Class 10)")
    print("    [B]            -> Save BOTTLE CAP (Class 8)")
    print("    [Click & Drag] -> Manually draw box if camera lighting is dim")
    print("    [Q] / [ESC]    -> Finish & view dataset")
    print("=" * 68 + "\n")

    # Load YOLOv8 for real-time auto detection
    base_model = ROOT_DIR / "yolov8m.pt"
    if not base_model.exists():
        base_model = "yolov8m.pt"
    print("Loading YOLOv8 AI assistant for instant bounding-box preview...")
    model = YOLO(str(base_model))

    # Open camera stream using zero-latency SmartStreamCapture
    from test_manual_ml_cam import SmartStreamCapture
    cap = SmartStreamCapture(args.source)

    cv2.namedWindow("Real Cup & Cap Collector")
    cv2.setMouseCallback("Real Cup & Cap Collector", mouse_callback)

    saved_cups = 0
    saved_caps = 0
    global manual_box

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        display = frame.copy()

        # Run assistant YOLO inference to detect cup (41) or bottle/cap
        results = model(frame, verbose=False, conf=0.35)[0]
        detected_cup_box = None
        for b in results.boxes:
            cid = int(b.cls[0])
            name = model.names[cid]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if name == 'cup':
                detected_cup_box = b.xywhn[0].tolist()
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"AUTO CUP: {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif name in ['bottle']:
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 1)
                cv2.putText(display, f"Bottle: {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Draw manual box if user dragged mouse
        if drawing and ix != -1 and rx != -1:
            cv2.rectangle(display, (ix, iy), (rx, ry), (0, 0, 255), 2)
        elif manual_box is not None:
            mx1, my1, mx2, my2 = manual_box
            cv2.rectangle(display, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
            cv2.putText(display, "MANUAL BOX READY", (mx1, my1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # HUD Overlay
        cv2.rectangle(display, (0, 0), (w, 45), (20, 20, 20), -1)
        hud_text = f"[SPACE/C] Save CUP: {saved_cups}  |  [B] Save BOTTLE CAP: {saved_caps}  |  [Q] Exit"
        cv2.putText(display, hud_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Real Cup & Cap Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:  # ESC or Q
            break
        elif key in [ord('c'), 32]:  # C or SPACE -> CUP
            # Determine box
            box = None
            if manual_box is not None:
                mx1, my1, mx2, my2 = manual_box
                bw = (mx2 - mx1) / w
                bh = (my2 - my1) / h
                bx = (mx1 + mx2) / (2 * w)
                by = (my1 + my2) / (2 * h)
                box = (bx, by, bw, bh)
                manual_box = None
            elif detected_cup_box is not None:
                box = detected_cup_box
            else:
                # Default centered region if no detection
                box = (0.5, 0.5, 0.4, 0.5)
                print("ℹ️ Auto-detector did not lock cup; saved center frame box (drag mouse next time for custom box)")

            save_sample(frame, class_id=10, class_name="cup", box_xywhn=box, sample_idx=saved_cups)
            saved_cups += 1
            cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 10)
            cv2.imshow("Real Cup & Cap Collector", display)
            cv2.waitKey(100)

        elif key == ord('b'):  # B -> BOTTLE CAP
            box = None
            if manual_box is not None:
                mx1, my1, mx2, my2 = manual_box
                bw = (mx2 - mx1) / w
                bh = (my2 - my1) / h
                bx = (mx1 + mx2) / (2 * w)
                by = (my1 + my2) / (2 * h)
                box = (bx, by, bw, bh)
                manual_box = None
            else:
                # Bottle caps are small: centered 15% box
                box = (0.5, 0.5, 0.15, 0.15)
                print("ℹ️ Saved center 15% box for bottle cap (drag mouse to outline cap precisely)")

            save_sample(frame, class_id=8, class_name="plastic bottle cap", box_xywhn=box, sample_idx=saved_caps)
            saved_caps += 1
            cv2.rectangle(display, (0, 0), (w, h), (0, 165, 255), 10)
            cv2.imshow("Real Cup & Cap Collector", display)
            cv2.waitKey(100)

    cv2.destroyAllWindows()
    if not is_stream and 'cap' in locals():
        cap.release()

    print("\n" + "=" * 65)
    print("  COLLECTION COMPLETE")
    print(f"  Cups Captured       : {saved_cups}")
    print(f"  Bottle Caps Captured: {saved_caps}")
    print("  All samples have been directly injected into train & valid sets!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
