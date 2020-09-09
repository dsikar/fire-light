import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time

# loading the stored model from file
# model=load_model(r'Fire-64x64-color-v7-soft.h5')
# model=load_model(r'take2.h5')

# works but suspicious with no fire - more data needed ?
model=load_model(r'take2.h5')
# cap = cv2.VideoCapture(r'fire-lighter.mp4') # about 90%
# cap = cv2.VideoCapture(r'fire-no-lighter.mp4') # 90% when off 99% when on
# cap = cv2.VideoCapture(r'no-fire-lighter.mp4') # max probability about 82%
cap = cv2.VideoCapture(r'empty-plus-fire.mp4') # <# 91% empty, <# 96% lighter, <# 99% lit lighter
# Mixing an matching, testing lighter model with real fire:
# model=load_model(r'take2.h5')
# cap = cv2.VideoCapture(r'FireVid20.mp4') # 100%
# cap = cv2.VideoCapture(r'NoFireVid10') # 92%
# cap = cv2.VideoCapture(r'NoFireVid11.mp4') # 99%
# model=load_model(r'take2.h5')
# cap = cv2.VideoCapture(r'FireVid20.mp4') # 100%
# cap = cv2.VideoCapture(r'NoFireVid10') # 92%
# cap = cv2.VideoCapture(r'NoFireVid11.mp4') # 99%
# model=load_model(r'Fire-64x64-color-v7-soft.h5')
# cap = cv2.VideoCapture(r'fire-lighter.mp4') # < 5%
# cap = cv2.VideoCapture(r'fire-no-lighter.mp4') # < 5%
# cap = cv2.VideoCapture(r'no-fire-lighter.mp4') # < 5%

# next, single image inference...


time.sleep(2)

if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False

IMG_SIZE = 64
# IMG_SIZE = 224

#for i in range(2500):
#    cap.read()



while(1):

    rval, image = cap.read()
    if rval==True:
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
        
        label = "Fire Probability: " + str(fire_prob)
        cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

        cv2.imshow("Output", orig)
        
        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
    elif rval==False:
            break
end = time.time()


cap.release()
cv2.destroyAllWindows()
