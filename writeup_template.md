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


[//]: # (Image References)

[image1]: ./img/training_movie.gif "Model Visualization"
[image2]: ./img/left_center_right_recorded_data.png "Perspective from 3 vantage points"
[image3]: ./img/steeringAngle_Distribution.png "Spread of Steering angles"
[image4]: ./img/horizontally_flipped_image.png "Flipping the image"
[image5]: ./img/brightness_adjusted.png "Minor adjustments to brightness"
[image6]: ./img/keras_cropped_image.png "Image cropping using Keras"
[image6]: ./img/autonomousDriving.gif "Autonomous Driving"

## Approach
### Data Collection

### Data Analysis and Augmentation

### Architecture of Deployed CNN and Training

### Autonomous Driving
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
