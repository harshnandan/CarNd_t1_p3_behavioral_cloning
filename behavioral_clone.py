import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re

cwd = os.getcwd()
#trainingDataFolder = "../windows-sim/recorded_training_data/"
trainingDataFolder = "../sample_train_data/data/"
csvfile = trainingDataFolder + "/driving_log.csv"

centerImg_lst = []
steeringAngle_lst = []

with open(csvfile) as csvfile:
    drive_log = csv.reader(csvfile)
    next(drive_log, None)  # skip the headers    
    counter = 0
    for r in drive_log:
        centerImgLocation = r[0]
        centerImgLocation = re.sub(r'\\', r'//', centerImgLocation)
        ImgFileName = re.split(r'\/', centerImgLocation)
        srcBGR = cv2.imread(trainingDataFolder + "/IMG/" + ImgFileName[-1] )
        srcRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
        centerImg_lst.append( srcBGR )
        steeringAngle_lst.append(r[3])
        counter += 1


X_train = np.array(centerImg_lst)
y_train = np.array(steeringAngle_lst, dtype=np.float32)
y_train = np.expand_dims(y_train, axis=1)

# plt.imshow(X_train[1])
# plt.show()
#  
# plt.plot(y_train)
# plt.show()


print("Shape of training images " + str(X_train.shape))


from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(24, (5, 5), activation=None))
# model.add(Conv2D(36, 5, 5, activation=None))
# model.add(Conv2D(48, 3, 3, activation=None))
# model.add(Conv2D(64, 3, 3, activation=None))
model.add(Flatten( ))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
train_log = model.fit(x=X_train, y=y_train, validation_split=0.2, 
                            shuffle=True, epochs=2, batch_size=64)

model.save('model.h5')

# summarize history for loss
plt.plot(train_log.history['loss'])
plt.plot(train_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'cross-validation'], loc='upper left')
plt.show()
