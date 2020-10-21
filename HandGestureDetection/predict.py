import numpy as np
import tensorflow as tf
import operator
import cv2
from Project1.Functions import *

# Loading the model
handG = tf.keras.models.load_model('C:/Users/ALAKH VERMA/PycharmProjects/OpenCVPython/Project1/Models/HandGestureDetection_cnn.h5')

#reading and displaying webcam
cap = cv2.VideoCapture(0)

#propId for width is 3, height is 4, brightness is 10
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)


# list of classes
#classes=['fist', 'next', 'none', 'one_f', 'palm', 'prev', 'swing', 'three_f', 'thumbs_up', 'two_f']

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 3)
    roi = frame[y1:y2, x1:x2]

    W = (x2 - x1)
    H = (y2 - y1)

    # Resizing the ROI so it can be fed to the model for prediction
    input = cv2.resize(roi.copy(), (256, 256))
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    #hsv values for skin colour
    lower = np.array([0, 38, 112]) # lower -> [h_min, s_min, v_min]
    upper = np.array([32, 255, 255])# upper -> [h_max, s_max, v_max]
    test_image = cv2.inRange(hsv, lower, upper)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = handG.predict(test_image.reshape(1, 256, 256, 1))
    prediction = {'fist': result[0][0],
                  'next': result[0][1],
                  'none': result[0][2],
                  'one_f': result[0][3],
                  'palm': result[0][4],
                  'prev': result[0][5],
                  'swing': result[0][6],
                  'three_f': result[0][7],
                  'thumbs_up': result[0][8],
                  'two_f': result[0][9],
                  }

    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    # calling the funtion imported from Functions.py
    findMyHand(prediction[0][0], roi, [0, 38, 112, 32, 255, 255], W, H)
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (x1, y2+20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(1)
    if interrupt & 0xFF == 27:  # esc key
        break

cap.release()
cv2.destroyAllWindows()