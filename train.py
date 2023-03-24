import cv2
from PIL import Image
from PIL import ImageTk
import imutils

import numpy as np
import os

from sklearn.model_selection import train_test_split
from  tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,Activation





data = []
labels = []
classes = 2
cur_path = os.getcwd()
for i in range(classes):
 path = os.path.join(cur_path,'train',str(i))
 print(path)
 images = os.listdir(path)
 for a in images:
     try:
         image = Image.open(path + '\\'+ a)
         image = image.resize((100,100))
         image = np.array(image)
         data.append(image)
         labels.append(i)                
     except:
         print("Error loading image")

data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 65
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)    
accuracy=accuracy*100    
model.save("trainc.h5")
print("done")
