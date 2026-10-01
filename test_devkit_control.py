"""
Interactive Wireless Robot & Robotic Arm Remote Control Tester
==============================================================
Tests wireless UDP control with the ESP32 DevKit at 192.168.137.69:8888

Controls:
  [W] Drive Forward
  [S] Drive Backward
  [A] Steer Left
  [D] Steer Right
  [Space] Stop Motors

Arm Testing:
  [1] Pick & Sort Plastic (Bin 1 - Left)
  [2] Pick & Sort Metal / Can (Bin 2 - Right)
  [3] Pick & Sort Glass / General (Bin 3 - Center)
  [H] Arm Home Position

  [Q] Exit
"""

import socket
import time
import sys

DEVKIT_IP = "192.168.137.227"
DEVKIT_PORT = 8888

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = (DEVKIT_IP, DEVKIT_PORT)

def send_cmd(cmd: str):
    try:
        sock.sendto(cmd.encode('utf-8'), target)
        print(f"📡 [Sent to ESP32 DevKit] --> {cmd}")
    except Exception as e:
        print(f"❌ [Error] {e}")

def main():
    print("=" * 65)
    print(" Wireless Smart Waste Robot & Arm Interactive Controller")
    print(f" Target: {DEVKIT_IP}:{DEVKIT_PORT}")
    print("=" * 65)
    print(" MOVEMENT KEYS:")
    print("   [W] Forward    [S] Backward    [A] Left    [D] Right    [X] Stop")
    print("\n ROBOTIC ARM TEST SEQUENCES:")
    print("   [1] Pick Plastic Bottle  (Sort to Bin 1 - 45 deg)")
    print("   [2] Pick Metal Can       (Sort to Bin 2 - 135 deg)")
    print("   [3] Pick Glass Bottle    (Sort to Bin 3 - 0 deg)")
    print("\n   [Q] Quit")
    print("=" * 65)

    # Initial safety stop
    send_cmd("CMD:STOP")

    while True:
        try:
            key = input("\nEnter command key (W/A/S/D/X/1/2/3/Q): ").strip().upper()
            if not key:
                continue

            if key == 'W':
                send_cmd("CMD:FORWARD")
            elif key == 'S':
                send_cmd("CMD:BACKWARD")
            elif key == 'A':
                send_cmd("CMD:LEFT")
            elif key == 'D':
                send_cmd("CMD:RIGHT")
            elif key == 'X' or key == ' ':
                send_cmd("CMD:STOP")
            elif key == '1':
                print("Executing Arm Pick for PLASTIC...")
                send_cmd("ARM:PICK:PLASTIC BOTTLE")
            elif key == '2':
                print("Executing Arm Pick for METAL...")
                send_cmd("ARM:PICK:CAN")
            elif key == '3':
                print("Executing Arm Pick for GLASS...")
                send_cmd("ARM:PICK:GLASS BOTTLE")
            elif key == 'Q':
                send_cmd("CMD:STOP")
                print("Exiting...")
                break
            else:
                print(f"Unknown key: {key}")
        except KeyboardInterrupt:
            send_cmd("CMD:STOP")
            break

    sock.close()

if __name__ == "__main__":
    main()
