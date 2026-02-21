import sys
import time
import numpy as np
import mss
import cv2
from ultralytics import YOLO
import socket
import keyboard
from pynput.mouse import Listener, Button
import winsound
import time

# ---------------- CONFIG ----------------
# https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification/data
MODEL_PATH = r"C:\YOLO\Delta\runs\detect\train\weights\best.pt"
IMGSZ = 640
CONF = 0.2

SCREEN_W = 2560
SCREEN_H = 1440
CENTER = (SCREEN_W // 2, SCREEN_H // 2)
FPS = 500
sleep_time = 1 / FPS

# Crop region: center 800x800
CROP_W = 500
CROP_H = 500
CROP_CENTER = (CENTER[0], CENTER[1])
CROP_TOP = int(CENTER[1] - CROP_H // 2)
CROP_LEFT = int(CENTER[0] - CROP_W // 2)

CAPTURE = {
    "top": CROP_TOP,
    "left": CROP_LEFT,
    "width": CROP_W,
    "height": CROP_H
}

CT_HEAD = 2
T_HEAD = 4

DETECTION_DISTANTCE_THRESHOLD = 50

# ----------------------------------------

def main():
    print("Initialize Start")
    model = YOLO(MODEL_PATH)
    sct = mss.mss()

    print(f"Finished Initizlizing models")
    last_toggle = 0

    aimmode = 0

    def loop():
        nonlocal last_toggle, aimmode

        # Toggle aimmode with XBUTTON_1 (x1). Debounced.
        if mouse_button_pressed_x1:
            if time.time() - last_toggle > 0.2:
                aimmode = (aimmode + 1) % 2
                print(f"AimMode: {aimmode}")
                last_toggle = time.time()
                freq = 600
                if aimmode == 0:
                    freq = 1200
                winsound.Beep(freq, 100)

        # target class is directly the aimmode (labels are 0 or 1)
        target_class = aimmode

        img_full = np.array(sct.grab(CAPTURE))
        frame_crop = cv2.cvtColor(img_full, cv2.COLOR_BGRA2BGR)

        results = model.predict(
            source=frame_crop,
            imgsz=IMGSZ,
            conf=CONF,
            device="cuda",
            verbose=False
        )

        closest = None
        min_dist = float("inf")

        for box in results[0].boxes:
            cls = int(box.cls)
            print(cls)

            # Only consider boxes matching the current aimmode label
            if cls != target_class:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Map cropped coordinates back to full-screen space
            cx_crop = (x1 + x2) // 2
            cy_crop = (y1 + y2) // 2

            cx_full = CROP_LEFT + cx_crop
            cy_full = CROP_TOP + cy_crop

            dx = cx_full - CENTER[0]
            dy = cy_full - CENTER[1]
            dist = dx * dx + dy * dy

            """ if dist > DETECTION_DISTANTCE_THRESHOLD ** 2:
                continue """

            if dist < min_dist:
                min_dist = dist
                closest = (x1, y1, x2, y2)

        # Log detection info
        if closest:
            cx_crop, cy_crop = (closest[0] + closest[2]) // 2, (closest[1] + closest[3]) // 2
            cx_full = CROP_LEFT + cx_crop
            cy_full = CROP_TOP + cy_crop

            distance = min_dist ** 0.5
            print(f"[DETECTED] AimMode {aimmode}: Head at ({cx_full}, {cy_full}) - Distance: {distance:.1f}")

            dx = cx_full - CENTER[0]
            dy = cy_full - CENTER[1]

            send_mouse_move(dx, dy)
        else:
            print(f"[NO HEAD DETECTED] AimMode {aimmode}")

    # Create socket client
    def send_mouse_move(x, y):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", 8080))
            message = f"{x} {y}\n"
            sock.send(message.encode())
            sock.close()
        except Exception as e:
            print(f"Error sending mouse move: {e}")

    # Global flag to track if middle button is pressed

    mouse_button_pressed_x1 = False
    mouse_button_pressed_x2 = False

    def on_click(x, y, button, pressed):
        nonlocal mouse_button_pressed_x1, mouse_button_pressed_x2
        if pressed and button == Button.x2:
            mouse_button_pressed_x2 = True  # Set flag for current frame
        else:
            mouse_button_pressed_x2 = False  # Reset after release

        if pressed and button == Button.x1:
            mouse_button_pressed_x1 = True  # Set flag for current frame
        else:
            mouse_button_pressed_x1 = False  # Reset after release

        

    # Start listening for middle mouse button (XBUTTON_1)
    with Listener(on_click=on_click) as listener:
        while True:
            try:
                if keyboard.is_pressed("insert"):  # Keep insert key to exit
                    break
                loop()
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()