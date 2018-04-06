import os
import csv
import gc
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from random import shuffle

samples = []
trainingDataFolder = "../recorded_training_data/"
with open(trainingDataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, 1)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = trainingDataFolder + '/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)


                name_left = trainingDataFolder + '/IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.imread(name_left)
                left_angle = center_angle + 1.0
                images.append(left_image)
                angles.append(left_angle)

                name_right = trainingDataFolder + '/IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.imread(name_right)
                right_angle = center_angle - 1.0
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

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, Lambda, Cropping2D, Dropout

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((80,20),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.0))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
model.add(Flatten( ))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.add(Dropout(0.4))

model.compile(loss='mse', optimizer='adam')
#
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*6, validation_data=validation_generator,
                        nb_val_samples=len(validation_samples)*6, nb_epoch=4)

model.save('model.h5')
gc.collect()
