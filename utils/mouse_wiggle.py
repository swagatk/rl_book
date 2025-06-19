import pyautogui
import threading
import time
import random

pyautogui.PAUSE = 0.1

screenSize = pyautogui.size()

def moveMouse():
    safe_margin = 10
    x = random.randint(safe_margin + int(0.3 * screenSize[0]), int(0.8 * screenSize[0]))
    y = random.randint(safe_margin + int(0.3 * screenSize[1]), int(0.8 * screenSize[1]))
    pyautogui.moveTo(x, y, duration = 1)

def clickMouse():
    pyautogui.click()
    main()

def main():
    try:
        while True:
            moveMouse()
            time.sleep(1.0)
            if time.time() % 10 < 10:
                clickMouse()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Script terminated by user")
        exit(-1)


if __name__ == '__main__':
    main()
