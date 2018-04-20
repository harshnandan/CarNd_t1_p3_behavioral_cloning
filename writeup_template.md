# **Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project** 

---

**Introduction**

The objective of this project is to use deep learning techniques to effectively teach a car autonomously in a simulated environment. For the purpose of this project, Udacity provided a simulator capable of operating in both autonomous and training mode. Although the simulator had two tracks, this project has been tested on the test track. The "challenge-track" will be tested in a follow-up effort.

The first step of teaching a deep neural network to drive is to collect a lot of rich data. The data is collected by running the simulator in training mode and using mouse and arrow keys to drive the car. The simulator captures steering angle, throttle, brake, speed and dashboard camera images and stores it in an organized fashion for further use. This data should cover key aspects of driving like making left and right turns, recovering from the side of the road etc. 

After appropriate inspection of data and required augmentation, the data is used to train a CNN network built using Keras. The trained model is saved as a .h5 file. This file is then used by drive.py file drive the car autonomously in the simulation environment.

The goals/steps of this project are the following:
* Use the simulator to collect data on good driving behavior
* Analyze the recorded data and decide on how to augment the data
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./img/training_movie.gif "Training data"
[image2]: ./img/left_center_right_recorded_data.png "Perspective from 3 vantage points"
[image3]: ./img/steeringAngle_Distribution.png "Spread of Steering angles"
[image4]: ./img/horizontally_flipped_image.png "Flipping the image"
[image5]: ./img/brightness_adjusted.png "Random adjustments to brightness"
[image6]: ./img/keras_cropped_image.png "Image cropping using Keras"
[image7]: ./img/autonomousDriving.gif "Autonomous Driving"

## Approach
### Data Collection
The data was collected from 2 laps around the test track. In the first attempt the car was driven smoothly around the track. Due to reasons discussed below, it was evident that more varied driving behavior was required for good performance. Hence at places, the car was deliberately taken to the edge of the track and then was steered back to the center of the lane. This way a rich data set of appropriate driving behavior was collected.

![alt text][image1]
![alt text][image2]
---

### Architecture of Deployed CNN and Training

#### 1. CNN architecture
First few attempts were made with a very simple architecture of one convolution layer and one dense layer to confirm the flow of data and get a preliminary sense of performance. It was seen that this basic architecture had large training and validation error and hence it was immediately clear that the architecture will require many more trainable parameters. Nvidia's architecture was a good option to start with. It was observed that with the full 160x320x3 image the memory requirement was pretty high. On AWS instance this architecture resulted in an Out-of-Memory error, the introduction of cropping layer reduced the memory requirement drastically. The cropping layer took off 25 pixels from the bottom and 75 pixels from top to retain only the relevant portion of the image. The model has another pre-processing Lambda layer which means normalizes each color channel. A detailed model architecture is shown below. Each Convolution layer is followed by a RELU activation function.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 70, 320, 3)    0           cropping2d_input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 70, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   1824        lambda_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 33, 158, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 15, 77, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 6, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 4, 35, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 72)     41544       activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 2, 33, 72)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4752)          0           activation_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          5532492     flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1164)          0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 20)            1020        dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 20)            0           dense_4[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             21          dropout_4[0][0]
====================================================================================================

Following is the output of cropping image from Keras.
![alt text][image6]

#### 2. Appropriate training data
With the first round of data collection, the car was driven very smoothly around the track once and the model was trained without any augmentation using just the center camera image. It was observed that the car primarily wanted to go straight and was not able to recover when it went close to the edge. The car went out of bounds near the first turn. Increasing the number of epochs led to a biased behavior where the car just wanted to turn left. This indicated that it was necessary to train the car on both left turn and right turn data and also provide data showing the car how to come back to center.

It is seen in the leftmost subfigure below that the recorded dataset has an inherent bias towards negative steering angle. This is the artifact of the track because the majority of turns in the loop are left turns. To augment the training data left and right camera images are also added to the training dataset. When a left image is added steering angle is adjusted by adding 0.2 to recorded steering angle, on the other hand when right camera image is used steering angle is adjusted by subtracting 0.2. The distribution of such a augmentation is shown in the center subfigure below. It still has a bias towards the negative steering angles. To remove this bias all (left-center-right camera) images were flipped horizontally and there steering angle were multiplied by -1. Training the model with this data could easily drive the car around the first major turn but as soon as it reached the turn following the bridge, it was unable to follow the turn and went into the muddy patch. Following this, I increased the significant steering data by repeating all the images whose steering angle is greater than 0.15 or less than -0.15. A random brightness adjustment was added to the repeated figure to avoid redundancy in data. The histogram plotted on the rightmost subfigure shows the distribution of steering angle. It can be seen that the user data set has no bias and has rich data of steering angles.

![alt text][image3]

Following is an example of horizontal flipping of image.
![alt text][image4]

Following is an example of adjusting the brightness of repeated image.
![alt text][image5]


### Autonomous Driving
The model trained using above approach was tested on the track and it was seen that the car successfully completed one lap around the track.
![alt text][image7]