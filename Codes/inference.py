"""
Use a convolutional neural network model
to predict the probability of fire/no fire
in a single image.
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time
model=load_model(r'take2.h5')

# image size used when training model

IMG_SIZE = 64

#Time taken =  0.14790678024291992
#FPS:  6.761015271629973
#Fire Probability:  99.99998807907104
#Predictions:  [[9.99999881e-01 1.09400766e-07]] # FIRE
image = cv2.imread('datasets/tmp/fire/20200724_175212_001.jpg')
#Time taken =  0.14594578742980957
#FPS:  6.851859293855501
#Fire Probability:  0.00011978886504948605
#Predictions:  [[1.1978887e-06 9.9999881e-01]] # NO FIRE
#image = cv2.imread('datasets/tmp/nofire/20200724_175219_001.jpg')

orig = image.copy()

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

tic = time.time()
fire_prob = model.predict(image)[0][0] * 100
toc = time.time()
print("Time taken = ", toc - tic)
print("FPS: ", 1 / np.float64(toc - tic))
print("Fire Probability: ", fire_prob)
print("Predictions: ", model.predict(image))
print(image.shape)

