import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os.path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

train_path = "C:/Users/ALAKH VERMA/PycharmProjects/OpenCVPython/Project1/data/train"
valid_path = "C:/Users/ALAKH VERMA/PycharmProjects/OpenCVPython/Project1/data/test"

train_batches = ImageDataGenerator(rescale=1./255) \
    .flow_from_directory(directory=train_path, color_mode='grayscale', target_size=(256,256), classes=['fist', 'next', 'none', 'one_f', 'palm', 'prev', 'swing', 'three_f', 'thumbs_up', 'two_f'], batch_size=32)
valid_batches = ImageDataGenerator(rescale=1./255) \
    .flow_from_directory(directory=valid_path, color_mode='grayscale', target_size=(256,256), classes=['fist', 'next', 'none', 'one_f', 'palm', 'prev', 'swing', 'three_f', 'thumbs_up', 'two_f'], batch_size=32)

"""
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(256,256,1)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dense(units=256, activation='relu'),
    Flatten(),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=6,
    verbose=2,
    callbacks=early_stopping_cb
)

if not os.path.exists("Models"):
    os.makedirs("Models")
model.save("C:/Users/ALAKH VERMA/PycharmProjects/OpenCVPython/Project1/Models/HandGestureDetection_cnn.h5")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, "b", label = 'Training Accuracy')
plt.plot(epochs, val_acc, "b--", label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, "r", label = 'Training Loss')
plt.plot(epochs, val_loss, "r--", label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
"""

#loading the model and checking the summary
handG = tf.keras.models.load_model('C:/Users/ALAKH VERMA/PycharmProjects/OpenCVPython/Project1/Models/HandGestureDetection_cnn.h5')
handG.summary()
