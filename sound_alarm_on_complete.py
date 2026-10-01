"""
Training Completion Sound Alarm & Monitor
=========================================
Watches the active training session (trash_extreme_cup_cap).
As soon as the 150-epoch training finishes and weights are saved,
plays a celebratory multi-tone beep alarm to alert you immediately.
"""

import os
import sys
import time
from pathlib import Path
import winsound

ROOT_DIR = Path(__file__).resolve().parent
RUN_DIR = ROOT_DIR / "runs" / "detect" / "trash_extreme_cup_cap"
BEST_PT = RUN_DIR / "weights" / "best.pt"
DEST_BEST = ROOT_DIR / "weights" / "best_trash_yolov8m.pt"
RESULTS_PNG = RUN_DIR / "results.png"

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def play_completion_alarm():
    """Play a cheerful, multi-frequency celebratory beep sequence."""
    print("\n" + "=" * 65)
    print("  [***] BEEP! BEEP! BEEP! TRAINING HAS COMPLETED! [***]")
    print("=" * 65 + "\n")
    # Melodic completion chime: (frequency_hz, duration_ms)
    melody = [
        (880, 200),   # A5
        (1046, 200),  # C6
        (1318, 250),  # E6
        (1568, 300),  # G6
        (2093, 600),  # C7
        (0, 200),     # Pause
        (1568, 250),
        (2093, 800)
    ]
    for freq, duration in melody:
        if freq == 0:
            time.sleep(duration / 1000.0)
        else:
            try:
                winsound.Beep(freq, duration)
            except Exception:
                pass
        time.sleep(0.05)

def main():
    print("\n" + "=" * 65)
    print("  [*] TRAINING COMPLETION AUDIO ALARM ACTIVE")
    print("=" * 65)
    print(f"  Watching: {RUN_DIR}")
    print("  Will sound melodic beep chimes the moment training finishes.")
    print("=" * 65 + "\n")

    # Give initial grace period for training process
    time.sleep(10)

    # Monitor loop
    while True:
        # Check if training has finalized:
        # YOLO writes results.png and copies best.pt at the very end of model.train()
        has_results_png = RESULTS_PNG.exists()
        has_best_weights = BEST_PT.exists()

        if has_results_png and has_best_weights:
            # Let the final evaluation and copy complete
            time.sleep(8)
            play_completion_alarm()
            break

        time.sleep(5)

if __name__ == "__main__":
    main()
