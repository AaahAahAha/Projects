# Music Genre Classification

## About the Project

This project implements **Music Genre Classification** using **Librosa** library on [GTZAN Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification). The Classification has been done using different Deep-Learning architectures namely **Convolutional Neural Network**
, **RNN-LSTM model** and **Multi-Layered Perceptron model** and their comparison is given. 

## Requirements

* Python3
* Tensorflow version == 2.3.0
* Keras
* Numpy, Pandas, Matplotlib
* Librosa - 0.8.0

## Description

### Dataset

The Dataset consists of 10 Music Genres namely:

* blues
* classical
* country
* disco
* hiphop
* jazz
* metal
* pop
* reggae
* rock

The Original Dataset contains **100 tracks** under each genre. However, the dataset used in this project used only **50 tracks** per genre further segmented into **10 segments** and stored 
in a JSON file **MusicData.json** which contains the **dictionary 'data'** holding the **genres, mfcc and labels**.

This json file has been created using **proccessData.py**. 

### Models

* Multi-Layered Perceptron 

This model has been implemented in **MLPModel.py** and is saved in **Multi_layer_Perceptron_Model.h5**.

#### Model Summary 

![Model](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/MLP_Images/Model_Summary.png)

#### Accuracy and Loss Curves

![Curve](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/MLP_Images/Curve.png)

#### Training Proccess

This image contains the last ten epochs of the training proccess.

![Training](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/MLP_Images/Training.png)

* Convolutional Neural Network

This model has been implemented in **CNNModel.py** and is saved in **CNN_Model.h5**.

#### Model Summary 

![Model](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/CNN_Images/Model_Summary.png)

#### Accuracy and Loss Curves

![Curve](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/CNN_Images/Curve.png)

#### Training Proccess

This image contains the last ten epochs of the training proccess alongwith the Test Accuracy.

![Training](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/CNN_Images/Training.png)

* Recurrent Neural Network using LSTM

This model has been implemented in **RNN-LSTM_Model.py** and is saved in **RNN_LSTM_Model.h5**.

#### Model Summary 

![Model](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/RNN_LSTM_Images/Model_Summary.png)

#### Accuracy and Loss Curves

![Curve](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/RNN_LSTM_Images/Curve.png)

#### Training Proccess

This image contains the last ten epochs of the training proccess alongwith the Test Accuracy.

![Training](https://github.com/AaahAahAha/Projects/blob/master/Music_Genre_Classification/RNN_LSTM_Images/Training.png)
