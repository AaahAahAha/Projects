import cv2
import numpy as np
import os

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/train/thumbs_up")
    os.makedirs("data/train/palm")
    os.makedirs("data/train/fist")
    os.makedirs("data/train/prev")
    os.makedirs("data/train/next")
    os.makedirs("data/train/swing")
    os.makedirs("data/train/one_f")
    os.makedirs("data/train/two_f")
    os.makedirs("data/train/three_f")

#function to stack an array of images
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

#reading and displaying webcam
cap = cv2.VideoCapture(0)

#propId for width is 3, height is 4, brightness is 10
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

count = 1
directory = 'data/train/'

while count<=1000:

    success, img_not = cap.read()  # cap.read() returns a boolean value and an image mat
    img = cv2.flip(img_not, 1)

    x1 = int(0.5 * img.shape[1])
    y1 = 10
    x2 = img.shape[1] - 10
    y2 = int(0.5 * img.shape[1])
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = img[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (256, 256))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #get the hsv values for my skin color from the ColorDetector.py
    lower = np.array([0, 38, 112])
    upper = np.array([32, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)
    imgResult = cv2.bitwise_and(roi, roi, mask = mask)

    imgStack = stackImages(0.6, [img, mask, imgResult])
    cv2.imshow("Result", imgStack)
    cv2.imwrite(directory+'none/'+str(count)+'.jpg', mask)
    count += 1
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()