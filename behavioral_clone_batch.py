import os
import csv

samples = []
trainingDataFolder = "../sample_train_data/data/"
with open(trainingDataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, 1)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = trainingDataFolder + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, Lambda

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3)))
#model.add(Conv2D(36, (5, 5), activation=None))
#model.add(Conv2D(48, (3, 3), activation=None))
#model.add(Conv2D(64, (3, 3), activation=None))
model.add(Flatten( ))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

