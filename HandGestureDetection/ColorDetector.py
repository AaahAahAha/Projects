import cv2
import numpy as np

def empty(a):
    pass

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

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 19, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 110, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 240, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 153, 255, empty)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

#reading and displaying webcam
cap = cv2.VideoCapture(0)

#propId for width is 3, height is 4, brightness is 10
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)

while True:

    success, img_not = cap.read()  # cap.read() returns a boolean value and an image mat
    img = cv2.flip(img_not, 1)

    x1 = int(0.5 * img.shape[1])
    y1 = 10
    x2 = img.shape[1] - 10
    y2 = int(0.5 * img.shape[1])

    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = img[y1:y2, x1:x2]
    roi = cv2.resize(roi, (256, 256))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask  = cv2.inRange(hsv, lower, upper)
    imgResult = cv2.bitwise_and(roi, roi, mask = mask)

    imgStack = stackImages(0.6, [img, mask, imgResult])
    cv2.imshow("Result", imgStack)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()