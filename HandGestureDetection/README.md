# Hand Gesture Detection using Convolutional Neural Networks and Computer Vision

## About the Project
This project implements a hand recognition and hand gesture recognition system using **OpenCV**. 
The hand gestures are detected and recognized using **Convolutional Neural Networks (CNN)** classification approach. 
The process flow consists of **hand region of interest segmentation using mask** image.
The detected hand (within the region of interest) is then processed and modelled by finding contours. 
Finally, a gesture object is created from the recognized pattern which is compared to a defined gesture dictionary to detect different hand gestures.
Using the detected hand gesture, a function can be performed from all the functions built using **pyautogui library** some of which include moving your mouse, 
various click events, etc.

## Requirements

  * Python3
  * OpenCV headless (cv2) for Python3
  * Tensorflow version == 2.3.0
  * pyautogui
  * Keras

## Description

### Dataset
The dataset has been created using the DataGenerator.py file. The dataset consists of 10 different classes:
* fist
* next
* none
* one_f
* palm
* prev
* swing
* three_f
* thumbs_up
* two_f

Each one of them consists of **1000 images** which have been further split into train and test in a **ratio of 9:1** using **FileTransfer.py**.

### File Description  
* **ColorDetector.py**
  
This file is used to get the hsv values of the hand set using the trackbars. The hsv values were chosen based on the result obtained from the masked images taking in these 
hsv values as input. You can use this file to detect the hsv values of your hand by tweaking the trackbars position.

* **DataGenerator.py**

This file is used to generate the training data and validation data comprised of the masked images of different hand gestures using the hsv values of the hand calculated 
using **ColorDetector.py**.

* **FileTransfer.py**

This file is used to split the data between the **train** and **test** directories by randomly sampling 100 images from each of the gesture directory under the data directory to 
each of the gesture directory under the test directory.

### Train 

* **train.py**

This file consists of the entire model.
The network contains **3 hidden convolution layers** with **Relu** as the activation function, **1 hidden Fully connected layer** and a **softmax layer**.

The network is trained across **6 iterations (epochs)** with a **batch size** of **32**.

![Model Summary](https://i.ibb.co/Syz7LFJ/Screenshot-2020-10-20-175741.png)

The training process:

![Training Process](https://i.ibb.co/BCMHzmh/Screenshot-2020-10-20-175619.png)

The graphs for the accuracy and loss were also created -

**Accuracy**            |  **Loss**
:-------------------------:|:-------------------------:
![Image for accuracy curve](https://i.ibb.co/vL9YnZc/1.png)  | ![Image for loss curve](https://i.ibb.co/ZGTMF1F/2.png)

The trained model is then saved as HandGestureDetection_cnn.h5 file under Models directory.

* **Functions.py**

This file contains functions namely: **getcontours(), movemouse(), movemouseslow(), lclick(), double_lclick(), rclick(), change_window(), action() and findMyHand()**. 

* **getcontours()**

This function finds and draws the contours. This function returns the center of the contour using boundingRect()

### pyautogui-based functions 

     movemouse() - triggered when 'palm' or 'fist' is predicted
     movemouseslow() - triggered when 'swing' is predicted
     lclick() - triggered when 'one_f' or 'thumbs_up' is predicted
     double_lclick() - triggered when 'two_f' is predicted
     rclick() - triggered when 'three_f' is predicted
     change_window() - triggered when 'prev' or 'next' is predicted

### Some important functions

* **action()**

This function calls one of the above stated pyautogui functions for the detected gesture.

* **findMyHand()**

This function calls **getcontours()** and **action()** functions and is finally called in **predict.py** file to detect the gesture.

### Test

* **predict.py**

This file loads our trained model and predicts one of the gestures from the defined dictionary of gestures detected in the region of interest on the frame.
Each detected gesture is associated with one of the defined functions as stated above. For example, if the detected gesture is **'palm'** according to the prediction then moving 
your palm will **move the mouse** on your screen by calling the **movemouse()** function. 




