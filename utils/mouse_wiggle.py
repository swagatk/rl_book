import pyautogui
import threading
import datetime
import random

screenSize = pyautogui.size()

def moveMouse():
    x = random.randint(int(0.3 * screenSize[0]), int(0.8 * screenSize[0]))
    y = random.randint(int(0.3 * screenSize[1]), int(0.8 * screenSize[1]))
    pyautogui.moveTo(x, y, duration = 1)

def clickMouse():
    pyautogui.click()
    main()

def main():
    hour = datetime.datetime.now().hour
    threading.Timer(5.0, moveMouse).start()
    threading.Timer(10.0, clickMouse).start()
main()
