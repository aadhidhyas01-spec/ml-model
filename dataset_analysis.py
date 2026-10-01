"""
Analyze dataset distribution, class balance, box sizes, and missing labels.
"""
from pathlib import Path
from collections import Counter
import yaml

def analyze_dataset():
    data_yaml = Path("data.yaml")
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    classes = data.get('names', [])
    print(f"Total classes ({len(classes)}): {classes}")
    
    for split in ['train', 'valid']:
        img_dir = Path(split) / "images"
        lbl_dir = Path(split) / "labels"
        
        imgs = list(img_dir.glob("*.*"))
        lbls = list(lbl_dir.glob("*.txt"))
        
        class_counts = Counter()
        total_boxes = 0
        empty_labels = 0
        small_boxes = 0 # area < 0.005 of image
        
        for lbl_file in lbls:
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            if not lines:
                empty_labels += 1
                continue
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])
                area = w * h
                class_counts[cls_id] += 1
                total_boxes += 1
                if area < 0.005:
                    small_boxes += 1
                    
        print(f"\n--- {split.upper()} SET ---")
        print(f"Total Images: {len(imgs)}")
        print(f"Total Labels: {len(lbls)}")
        print(f"Empty Label Files: {empty_labels}")
        print(f"Total Bounding Boxes: {total_boxes}")
        print(f"Small Bounding Boxes (<0.5% image area): {small_boxes} ({small_boxes/max(total_boxes,1)*100:.1f}%)")
        print("\nClass Breakdown:")
        for idx, name in enumerate(classes):
            cnt = class_counts[idx]
            pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0
            print(f"  [{idx}] {name:<22}: {cnt:5d} ({pct:5.1f}%)")

if __name__ == "__main__":
    analyze_dataset()
