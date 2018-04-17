import cv2, sys, gc, csv, pandas
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def showDrivingAngles(samples, title="recorded training samples"):
    lst = [float(sample[3]) for sample in samples ] + [-1*float(sample[3]) for sample in samples ]
    plt.hist(lst, 64)
    plt.title("Steering angle distribution in " + title)
    plt.show()

samples = []
trainingDataFolder = "../recorded_training_data-2/"
with open(trainingDataFolder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, 1)
    for line in reader:
        samples.append(line)
        if abs(float(line[3])) > 0.1:
            samples.append(line)

line_nb = 790
name_center =  samples[line_nb][0]
name_left =  samples[line_nb][1]
name_right =  samples[line_nb][2]
stAng =  float(samples[line_nb][3])

showDrivingAngles(samples)


center_image = cv2.imread(name_center)
center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

left_image = cv2.imread(name_left)
left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

right_image = cv2.imread(name_right)
right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

center_angle = stAng * 25
line_len = 80

line_y, line_x = int(line_len * np.cos(center_angle)), int(line_len * np.sin(center_angle))
x1, y1 = int(320/2), int(160)
x2, y2 = x1+line_x, y1+line_y
center_image_st = cv2.line(center_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# plt.figure(figsize=(15, 3))
# plt.subplot(1, 3, 1)
# plt.imshow(left_image)
# plt.title('Left Camera Image' )
# plt.subplot(1, 3, 2)
# plt.imshow(center_image_st)
# plt.title('Center Camera Image - Angle: {:.2f}'.format(stAng*25) )
# plt.subplot(1, 3, 3)
# plt.imshow(right_image)
# plt.title('Right Camera Image')
# plt.show()
# 
# sys.exit()
#  
# plt.figure(figsize=(10, 3))
# plt.subplot(1, 2, 1)
# plt.imshow(center_image)
# plt.title('Recorded Center Camera Image')
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.flip(center_image, 1))
# plt.title('Horizontally Flipped Center Camera Image')
# plt.show()

# from keras.models import Sequential
# from keras.layers import Cropping2D
# 
# model = Sequential()
# model.add(Cropping2D(cropping=((75,23),(0,0)), input_shape=(160,320,3)))
# # model.summary()
# croppedimage_left = model.predict(np.expand_dims( left_image, axis=0 ) )[0]
# croppedimage_center = model.predict(np.expand_dims( center_image, axis=0 ) )[0]
# croppedimage_right = model.predict(np.expand_dims( right_image, axis=0 ) )[0]
# 
# plt.figure(figsize=(15, 3))
# plt.subplot(1, 3, 1)
# plt.imshow(croppedimage_left)
# plt.title('Left Camera Image' )
# plt.subplot(1, 3, 2)
# plt.imshow(croppedimage_center)
# plt.title('Center Camera Image')
# plt.subplot(1, 3, 3)
# plt.imshow(croppedimage_right)
# plt.title('Right Camera Image')
# plt.show()
# 
# gc.collect()
<<<<<<< HEAD
=======

>>>>>>> ae7949c7c716fa955b20382c8b59649fe678e5ce
