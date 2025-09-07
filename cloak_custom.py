import cv2
import numpy as np
import time

# ------------------- Predefined HSV Ranges -------------------
COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "blue": [
        (np.array([90, 80, 2]), np.array([130, 255, 255]))
    ],
    "green": [
        (np.array([40, 40, 40]), np.array([80, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255]))
    ],
    "orange": [
        (np.array([10, 100, 20]), np.array([25, 255, 255]))
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([180, 255, 30]))  # Low brightness & saturation
    ],
}

# Default cloak color
cloak_color = "red"

# ------------------- Capture Background -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not access the camera. Try using index 1 or 2.")
    exit()

print("📸 Capturing background... please stay still")
time.sleep(2)
for _ in range(60):
    ret, bg = cap.read()
    if ret:
        bg = cv2.flip(bg, 1)
bg = bg.copy()
print("✅ Background captured")

# ------------------- Main Loop -------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror effect
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Build mask based on selected color
    mask = None
    for lower, upper in COLOR_RANGES[cloak_color]:
        this_mask = cv2.inRange(hsv, lower, upper)
        if mask is None:
            mask = this_mask
        else:
            mask = mask | this_mask  # combine if multiple ranges (like red)

    # Refine mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask_inv = cv2.bitwise_not(mask)

    # Apply cloak effect
    cloak_area = cv2.bitwise_and(bg, bg, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final = cv2.add(cloak_area, non_cloak_area)

    # Show result
    cv2.putText(final, f"Cloak Color: {cloak_color.upper()}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("🪄 Invisibility Cloak", final)
    cv2.imshow("🎭 Cloak Mask (debug)", mask)

    # Key controls
    key = cv2.waitKey(30) & 0xFF
    if key != 255:
        print("Pressed:", chr(key))  # Debugging print

    if key == 27:  # ESC = quit
        break
    elif key == ord('c'):  # recapture background
        print("♻ Re-capturing background...")
        for _ in range(60):
            ret, bg = cap.read()
            if ret:
                bg = cv2.flip(bg, 1)
        bg = bg.copy()
        print("✅ Background updated")

    # Cloak color hotkeys
    elif key == ord('r'):
        cloak_color = "red"
        print("🔴 Cloak color changed to RED")
    elif key == ord('g'):
        cloak_color = "green"
        print("🟢 Cloak color changed to GREEN")
    elif key == ord('b'):
        cloak_color = "blue"
        print("🔵 Cloak color changed to BLUE")
    elif key == ord('y'):
        cloak_color = "yellow"
        print("🟡 Cloak color changed to YELLOW")
    elif key == ord('o'):
        cloak_color = "orange"
        print("🟠 Cloak color changed to ORANGE")
    elif key == ord('k'):
        cloak_color = "black"
        print("⚫ Cloak color changed to BLACK")

cap.release()
cv2.destroyAllWindows()
