

import cv2
import imutils

import os
import numpy as np
from PIL import ImageTk, Image
from keras.models import Sequential, load_model
from keras.models import load_model






file_path='C:\\Users\\lenovo\\Desktop\\code\\aaj.jpeg'
model = load_model('trainc.h5')
clas = {1:'Raw',2:'Ripened'}
image = Image.open(file_path).convert("RGB")
image = image.resize((100,100))
image = np.expand_dims(image, axis=0)
image = np.array(image)
pred = model.predict([image])[0]
classes = np.argmax(pred, axis=-1)
sign = clas[classes+1]
print(sign)

