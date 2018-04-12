import os
import csv
import gc
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from random import shuffle

samples = []
trainingDataFolder = "../recorded_training_data-2/"
with open(trainingDataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, 1)
    for line in reader:
        samples.append(line)
        if abs(float(line[3])) > 0.15:
            samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.1)

def adjust_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = 0.25 + np.random.uniform()
    hsv[:, :, 2] = hsv[:, :, 2] * bright
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = batch_sample[0]
                center_image = cv2.imread(name_center)
                if np.random.randint(2) == 0:
                    center_image = adjust_image_brightness(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                name_left = batch_sample[1]
                left_image = cv2.imread(name_left)
                if np.random.randint(2) == 0:
                    left_image = adjust_image_brightness(left_image)
                left_angle = center_angle + 0.2
                images.append(left_image)
                angles.append(left_angle)

                name_right = batch_sample[2]
                right_image = cv2.imread(name_right)
                if np.random.randint(2) == 0:
                    right_image = adjust_image_brightness(right_image)
                right_angle = center_angle - 0.2
                images.append(right_image)
                angles.append(right_angle)

                images.append(cv2.flip(center_image, 1))
                angles.append(-1.*center_angle)

                images.append(cv2.flip(left_image, 1))
                angles.append(-1.*left_angle)

                images.append(cv2.flip(right_image, 1))
                angles.append(-1.*right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=8)
validation_generator = generator(validation_samples, batch_size=8)

a = next(train_generator)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, Lambda, Cropping2D, Dropout, Activation

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((65,25),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.0))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2) ))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2) ))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2) ))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3 ))
model.add(Activation('relu'))
model.add(Convolution2D(72, 3, 3 ))
model.add(Activation('relu'))
model.add(Flatten( ))

model.add(Dense(1164))
#model.add(Dense(512))
model.add(Dropout(0.4))

model.add(Dense(100))
model.add(Dropout(0.4))

model.add(Dense(50))
model.add(Dropout(0.4))

model.add(Dense(20))
model.add(Dropout(0.4))

model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
#
# 1.2.1 call
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*6, validation_data=validation_generator,
                        nb_val_samples=len(validation_samples)*6, nb_epoch=10)

# 2.1.5 call
#nb_train_sample_per_epoch = len(train_samples)
#nb_val_sample_per_epoch = len(validation_samples)
#model.fit_generator(train_generator, steps_per_epoch = nb_train_sample_per_epoch/(8), validation_data=validation_generator,
#                        validation_steps = nb_val_sample_per_epoch/(8), nb_epoch=4)
gc.collect()
model.save('model.h5')
