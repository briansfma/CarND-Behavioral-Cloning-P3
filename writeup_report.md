# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./home/workspace/sim-training-data/IMG/center_2020_02_25_07_41_15_765.jpg "Center Driving"
[image2]: ./home/workspace/sim-training-data/IMG/center_2020_02_25_07_41_08_443.jpg "Recovery Image 1"
[image3]: ./home/workspace/sim-training-data/IMG/center_2020_02_25_07_41_09_011.jpg "Recovery Image 2"
[image4]: ./home/workspace/sim-training-data/IMG/center_2020_02_25_07_41_09_745.jpg "Recovery Image 3"
[image5]: ./home/workspace/sim-training-data/IMG/center_2020_02_25_07_41_08_443 - flipped.jpg "Flipped Image"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (README.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 53-67) 

The model includes RELU layers everywhere except the last hidden layer and the output layer for nonlinearity. The data is normalized in the model using a Keras lambda layer up top (line 54). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in the first two fully connected layers in order to reduce overfitting (model.py lines 62, 64). The drop rate was set to only 20% as it appeared the model had trouble fitting at all with 50% drop rate.

The model used a 20% validation split on the multi-lap data set to reduce the chance of the model overfitting (line 72). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 71).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from different portions of the lane at various angles in order to give the network into an appropriate "if this, then that" to learn from.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a proven, published network, then ensure that the input data quality was high enough to produce a good result.

My first step was to use a convolution neural network model similar to the NVidia network highlighted in class materials. I thought this model might be appropriate because it is 1) already working and 2) has sufficient convolutional depth that it could probably pick out high-level features like road edges/track curbs, then calculate the rest in the 3 hidden, fully connected layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (20% at random, using the `validation_split` parameter in `model.fit()`. I found that my first model had a low mean squared error on the training set but fluctuated wildly and inconsistently on the validation set. This implied that the model was overfitting, and possibly not nonlinear enough (in the video lecture, the fully connected layers did not have any special activation function) to handle the course.

To combat the overfitting, I added RELU activation and 20% dropout to the first two fully connected layers. This seems to have done the trick, with MSE on the validation set dropping to ~0.0340.

Not seeing much gain from other experiments, I ran the simulator to see how well the car was driving around track one. To my surprise, on the first lap the vehicle had no issues - never fell off the track, never stepped over the marked track boundary, just got close (about 1/4 car width) to the line on some turns. So, good enough for a first pass!

#### 2. Final Model Architecture

The final model architecture (model.py lines 53-67) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

| Layer     			|     Description       						| 
|:---------------------:|:---------------------------------------------:| 
| 0) Input 				| 320x160x3 RGB image   							|
|						|												|
| 1) Lambda				| Normalization (-0.5 to 0.5)					|
|						|												|
| 2) Input 				| Crop image (-65px from top, -25px from bottom)|
|						|												|
| 3) Convolution 5x5 	| 1x1 stride, valid padding, output depth 24	|
| RELU 					|												|
| Max pooling 			| 2x2 stride					 				|
|						|												|
| 4) Convolution 5x5 	| 1x1 stride, valid padding, output depth 36 	|
| RELU 					|												|
| Max pooling 			| 2x2 stride					 				|
|						|												|
| 5) Convolution 5x5 	| 1x1 stride, valid padding, output depth 48 	|
| RELU 					|												|
| Max pooling 			| 2x2 stride					 				|
|						|												|
| 6) Convolution 3x3 	| 1x1 stride, valid padding, output depth 64 	|
| RELU 					|												|
|						|												|
| 7) Convolution 3x3 	| 1x1 stride, valid padding, output depth 64 	|
| RELU 					|												|
|						|												|
| 8) Fully Connected 	| Outputs 100 wide 								|
| RELU 					|												|
| Dropout 				| Keep rate 80% 								|
|						|												|
| 9) Fully Connected 	| Outputs 50 wide 								|
| RELU 					|												|
| Dropout 				| Keep rate 80% 								|
|						|												|
| 10) Fully Connected 	| Outputs 10 wide 								|
| RELU 					|												|
|						|												|
| 11) Output 			| Outputs 1 wide 								|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

I repeated this process, in two different manners, across all the different curbs/boundaries on Track 1:

A) Approaching (heading into) a track boundary
B) Leaving (heading back from) a track boundary

1) Double yellow lines
2) Candy-cane lines
3) Bridge walls
4) Dirt boundaries

To augment the data sat, I also flipped images and angles, giving the network twice the dataset to train with and hopefully, removing the inherent left-hand bias that occurs when driving Track 1 counterclockwise. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image5]

After the collection process, I had 6267 data points. Flipping images doubles this to 12534 points.

I randomly shuffle the data set and put 20% of the data into a validation set for each training operation. I used an adam optimizer so that manually training the learning rate wasn't necessary. The optimal number of epochs depended on the dropout rate, it seems, but with 20% dropout it seemed like 5 epochs would bring the validation error to a minimum, with few gains beyond that.
