import cv2
import pyautogui
import numpy as np
import time
from PIL import ImageGrab

SCAN_REGION = (558, 328, 1423, 816)
BUTTON_X, BUTTON_Y = 1305, 356
TARGET_IMAGE_PATH = "midas_snippet.png"  

template = cv2.imread(TARGET_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

def detect_name_in_region():
    screenshot = ImageGrab.grab(bbox=SCAN_REGION)
    screenshot = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  
    loc = np.where(res >= threshold)

    if len(loc[0]) > 0 and len(loc[1]) > 0:
        return True
    return False

def main():
    print("🔍 Running... waiting for 'Midas' using template matching")
    while True:
        if detect_name_in_region():
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Found 'Midas'! Clicking button.")
            pyautogui.click(BUTTON_X, BUTTON_Y)
            time.sleep(2)  
        time.sleep(0.3)

if __name__ == "__main__":
    main()
