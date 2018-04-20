import cv2, sys, gc, csv, pandas
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import imageio
from random import shuffle

# plot histogram of steering angles
def showDrivingAngles(samples, aug='original', title="recorded training samples"):
    # each image is mirror imaged by default to create the augmented dataset
    if aug=='reflect':
        lst = [float(sample[3]) for sample in samples ] + [-1*float(sample[3]) for sample in samples ]
    elif aug == 'left_right':
        lst = [float(sample[3]) for sample in samples ] + [float(sample[3])+0.2 for sample in samples ] + [float(sample[3])-0.2 for sample in samples ] 
    elif aug == 'original':
        lst = [float(sample[3]) for sample in samples ] 
    elif aug == 'left_right_reflect':
        lst_1 = [float(sample[3]) for sample in samples ] + [float(sample[3])+0.2 for sample in samples ] + [float(sample[3])-0.2 for sample in samples ]
        lst = [float(ls) for ls in lst_1 ] + [-1*float(ls) for ls in lst_1]
    # use 64 bins
    plt.hist(lst, 64)
    plt.title("Steering angle distribution in " + title)

def visualizeSteeringAngle(image, Ang):
    # draw a line of arbitrary length to visualize the steering angle
    line_len = -80
    line_y, line_x = int(line_len * np.cos(Ang*25/180*np.pi)), int(line_len * np.sin(-Ang*25/180*np.pi))
    x1, y1 = int(320/2), int(160)
    x2, y2 = x1+line_x, y1+line_y
    center_image_st = cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)    

def adjust_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright = 0.25 + np.random.uniform()
    hsv[:, :, 2] = hsv[:, :, 2] * bright
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
samples = []
samples_original = []
trainingDataFolder = "../recorded_training_data-2/"
with open(trainingDataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, 1)
    for line in reader:
        samples.append(line)
        samples_original.append(line)
        # adding 
        if abs(float(line[3])) > 0.1:
            samples.append(line)

# Avoid running the gif creation each time
if 0:
    with imageio.get_writer('./training_movie.gif', mode='I') as writer:
        for filenameIdx in range(0, len(samples), 10):
            filename = samples_original[filenameIdx][0]
            image = imageio.imread(filename)
            writer.append_data(image)


# Plots stats of training steering angle           
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
showDrivingAngles(samples_original, aug='original', title="\ndata set with center images only")
plt.subplot(1, 3, 2)
showDrivingAngles(samples_original, aug='left_right', title="\ndata set with left-right images")
plt.subplot(1, 3, 3)
showDrivingAngles(samples, aug='left_right_reflect', title="\ndata set with left-right images and flipped images")
plt.show()

# Exploration of image and steering angle data using a sample image    
line_nb = 800
name_center =  samples[line_nb][0]
name_left =  samples[line_nb][1]
name_right =  samples[line_nb][2]
stAng =  float(samples[line_nb][3])

# read center image
center_image = cv2.imread(name_center)
center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

# read left image
left_image = cv2.imread(name_left)
left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

# read right image
right_image = cv2.imread(name_right)
right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

# convert normalized steering angle to absolute steering angle and plot
left_angle = (stAng + 0.2) 
visualizeSteeringAngle(left_image, left_angle)

center_angle = stAng
visualizeSteeringAngle(center_image, center_angle)

right_angle = (stAng - 0.2)
visualizeSteeringAngle(right_image, right_angle)

# plot all 3 images together 
plt.figure(figsize=(15, 3))
plt.subplot(1, 3, 1)
plt.imshow(left_image)
plt.title('Left Camera Image' )
plt.subplot(1, 3, 2)
plt.imshow(center_image)
plt.title('Center Camera Image - Angle: {:.2f} deg'.format(stAng*25) )
plt.subplot(1, 3, 3)
plt.imshow(right_image)
plt.title('Right Camera Image')
plt.show()
 
# plot horizontally flipped image
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.imshow(center_image)
plt.title('Recorded Center Camera Image')
plt.subplot(1, 2, 2)
plt.imshow(cv2.flip(center_image, 1))
plt.title('Horizontally Flipped Center Camera Image')
plt.show()

# plot image brightness
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.imshow(center_image)
plt.title('Recorded Center Camera Image')
plt.subplot(1, 2, 2)
plt.imshow(adjust_image_brightness(center_image))
plt.title('Adjustments made to brightness')
plt.show()

# plot the cropped image
from keras.models import Sequential
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((75,23),(0,0)), input_shape=(160,320,3)))
model.summary()
croppedimage_left = model.predict(np.expand_dims( left_image, axis=0 ) )[0]
croppedimage_center = model.predict(np.expand_dims( center_image, axis=0 ) )[0]
croppedimage_right = model.predict(np.expand_dims( right_image, axis=0 ) )[0]

plt.figure(figsize=(15, 3))
plt.subplot(1, 3, 1)
plt.imshow(np.uint8(croppedimage_left))
plt.title('Left Camera Image' )
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(croppedimage_center))
plt.title('Center Camera Image')
plt.subplot(1, 3, 3)
plt.imshow(np.uint8(croppedimage_right))
plt.title('Right Camera Image')
plt.show()

gc.collect()

