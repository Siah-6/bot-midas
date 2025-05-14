import cv2
import pyautogui
import numpy as np
import time
from PIL import ImageGrab

# Emulator screen region (top-left x, y, bottom-right x, y)
SCAN_REGION = (558, 328, 1423, 816)
BUTTON_X, BUTTON_Y = 1305, 356
TARGET_IMAGE_PATH = "midas_snippet.png"  # Path to the cropped image of "Midas"

# Load the template image (the "Midas" snippet)
template = cv2.imread(TARGET_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Function to perform template matching using an image pyramid
def detect_name_in_region():
    # Get the width and height of the template image
    w, h = template.shape[::-1]

    screenshot = ImageGrab.grab(bbox=SCAN_REGION)
    screenshot = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Generate an image pyramid for the template (different scales)
    pyramid = [template]
    for scale in np.linspace(0.5, 1.5, 5):  # scale the image between 50% and 150%
        resized = cv2.resize(template, (int(w * scale), int(h * scale)))
        pyramid.append(resized)

    # Perform template matching at each scale
    threshold = 0.8  # Adjust this threshold as needed
    for temp in pyramid:
        w, h = temp.shape[::-1]  # Update width and height for each resized template
        res = cv2.matchTemplate(screenshot_gray, temp, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        # If a match is found, return True
        if len(loc[0]) > 0 and len(loc[1]) > 0:
            return True
    return False

# Main loop
def main():
    print("🔍 Running... waiting for 'Midas' using template matching with scale variations")
    while True:
        if detect_name_in_region():
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Found 'Midas'! Clicking button.")
            pyautogui.click(BUTTON_X, BUTTON_Y)
            time.sleep(2)  # avoid multiple clicks too fast
        time.sleep(0.3)

if __name__ == "__main__":
    main()
