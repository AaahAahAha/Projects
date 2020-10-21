import cv2
import numpy as np
import pyautogui

'''Note: find the hsv min and max values from ColorDetector and test it at the same place with same brightness level'''

pyautogui.FAILSAFE = False
#reading and displaying webcam
cap = cv2.VideoCapture(0)

mySkinColor = [0, 38, 112, 32, 255, 255] #from ColorDetector.py
myPointer = [255, 255, 255]

def findMyHand(text,img, mySkinColor, W, H):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array(mySkinColor[0:3])
    upper = np.array(mySkinColor[3:6])

    mask = cv2.inRange(imgHSV, lower, upper)
    x, y = getcontours(img, mask)
    action(text, x, y, W, H)

def getcontours(roi, img):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cont in contours:
        #this will detect some noise as well
        area = cv2.contourArea(cont)

        if area>500:#this will avoid the noisiness
            cv2.drawContours(roi, cont, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.02*peri, True)#coordinates of corners
            x, y, w, h = cv2.boundingRect(approx)
            cv2.circle(roi, (x+w//2, y+h//2), 10, myPointer, cv2.FILLED)

    return x+w//2, y+h//2

def movemouse(x, y, w, h):

    mx, my = pyautogui.position()
    del_x = x-(w//2)
    del_y = y-(h//2)
    pyautogui.moveTo(mx+del_x, my+del_y)

def movemouseslow(x, y, w, h):

    mx, my = pyautogui.position()
    del_x = x-(w//2)
    del_y = y-(h//2)
    pyautogui.moveTo(mx+del_x//5, my+del_y//5)


def lclick():
    pyautogui.click(button="left")


def double_lclick():
    pyautogui.click(button="left", clicks=2)


def rclick():
    pyautogui.click(button="right")


def change_window():
    pyautogui.hotkey("alt", "tab")


# Function to assign action to different gestures
def action(text, x, y, w, h):
    if text == "fist" or text == "palm":
        movemouse(x, y, w, h)
    elif text == "swing":
        movemouseslow(x, y, w, h)
    elif text == "one_f" or text == 'thumbs_up':
        lclick()
    elif text == "two_f":
        double_lclick()
    elif text == "three_f":
        rclick()
    elif text == "prev" or text == "next":
        change_window()
    else:
        return
