import pyautogui
import threading
import datetime

screenSize = pyautogui.size()

def moveMouse():
    pyautogui.moveTo(700, 500, duration = 1)


def clickMouse():
    pyautogui.click(300, 300)
    main()

def main():
    hour = datetime.datetime.now().hour
    if hour == 17 or hour == 12:
        print("end of day reached")
        quit()
    else:
        threading.Timer(5.0, moveMouse).start()
        threading.Timer(10.0, clickMouse).start()
main()
