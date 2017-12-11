
# Behavioral Cloning Project

The goal of this project is to use a deep neural network to clone driving behavior in a simulation program. 
Specifically, the network predicts the steering angle of the car based on front mounted camera images 
taken while the car is driven around a short track. Images and steering angle are recorded when the simulator 
is in training mode. Throttle is controlled by the user. In autonomous mode, the network is used to predict the
steering angle and throttle is controlled to drive the car at a constant speed, which for this project is 
9 mph.

  


[//]: # (Image References)
[bridge1_img]:      ./examples/bridge1.jpg
[bridge2_img]:      ./examples/bridge2.jpg
[first_turn1_img]:  ./examples/first_turn1.jpg
[first_turn2_img]:  ./examples/first_turn2.jpg
[second_turn1_img]: ./examples/second_turn1.jpg
[second_turn2_img]: ./examples/second_turn2.jpg
[video]: ./final_run.mp4

## Rubric Points  

The organization of this project write-up is motivated by the rubric points
which may be found [here](https://review.udacity.com/#!/rubrics/432/view).
  


- [Discussion](#discussion)
- [Training Data](#training-data)
- [Model Architecture](#model-architecture)
- [Autonomous Driving Video](autonomous-driving-video)


## Discussion

In my initial attempts to use simulated driving data to autonomously drive 
the car, I used a LeNet like network architecture and data from a few passes around the track,
going in the normal direction. I noticed that it was important to use a mouse to turn the car, as that
provided smooth turning angles. Its important for the network to see continuous constant or smoothly 
changing steering angles on curved sections of the road. Using the keyboard to steer generates mostly
zero degree angles with relatively rare and brief steep angles.  The network essentialy predicts 
zero degree turning angles throughout and hence is unable to drive the car correctly.

Even with image data taken going in the opposite direction, and attempting to use the left and right camera images I was unable to get the car past the first turn. After further failed attempts, I decided to adopt the Nvidia architecture and only use the center camera images. Proper inclusion of left and right images I think is perhaps more complex than what is described in the instructional material. I also felt it was necessary to try to drive the car in training mode at the same speed that drive.py uses (9.0 mph), since the proper steering angle of course depends on the speed.  

With the further addition of dropout after two of the fully connected layers in the network (see below),
I found the car drove quite well before hitting the guard rail halfway across the bridge. At this point, 
I added further training data at this point on the bridge and on the subsequent two turns. I then found 
the car drove around the complete track several times without issue. 


## Training Data

The breakdown of the final training data is as follows


|Driving Location          |  Number of Images |
|:------------------------:|:-----------------:|      
| Initial Passes           |  26232            |
| Middle of bridge         |  1194             |  
| First turn after bridge  |  3412             |
| Second turn after bridge |  1898             |

"Initial Passes" refers to driving continuously around the track in the normal direction.

Some example images from these directories.


### Middle of Bridge
Middle of Bridge 1         |  Middle of Bridge 2
:-------------------------:|:-------------------------:
![][bridge1_img]        |  ![][bridge2_img]

### First Turn After Bridge
First Turn 1               |  First Turn  2
:-------------------------:|:-------------------------:
![][first_turn1_img]       |  ![][first_turn2_img]

### Second Turn After Bridge
Second Turn 1               |  Second Turn  2
:--------------------------:|:-------------------------:
![][second_turn1_img]       |  ![][second_turn2_img]



## Model Architecture

The final architecture used is as follows.

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 160x320x3 RGB image                           |
| Normalization         | All pixels normalized to (-0.5, 0.5)          |
| Crop                  | Image cropped at 70th pixel from bottom,      |
|                       | 25th pixel from top (70,25),(0,0)             |
| 2D Convolution Layer  | 24 channel output, 5x5 filter at 2x2 stride, VALID padding, RELU Activation|
|                       | Output 24x30x30                               |
| 2D Convolution Layer  | 36 channel output, 5x5 filter at 2x2 stride, VALID padding, RELU Activation|
|                       | Output 36x13x13                               |
| 2D Convolution Layer  | 48 channel output, 5x5 filter at 2x2 stride, VALID padding, RELU Activation|
|                       | Output 48x5x5                                 |
| 2D Convolution Layer  | 64 channel output, 3x3 filter at 1x1 stride, VALID padding, RELU Activation|
|                       | Output 64x3x3                                 |
| 2D Convolution Layer  | 64 channel output, 3x3 filter at 1x1 stride, VALID padding, RELU Activation|
|                       | Output 64x1x1                                 |
| Flatten               | Flatten layer to 64 channels                  |
| Dense Layer           | 100 Channel Fully connected layer             |
| Dropout               | 20% of inputs randomly zeroed                 |
| Dense Layer           | 50  Channel Fully connected layer             |
| Dropout               | 20% of inputs randomly zeroed                 |
| Dense Layer           | 10  Channel Fully connected layer             |
| Dense Layer           | 1   Channel Fully connected layer, Steering Angle Output |
 


The code model.py that implements the architecture, also makes use of generator to avoid 
out of memory issues. The file "model.h5" was generated by running on an AWS instance with six epochs.

```
python model.py
```


## Autonomous Driving Video

The simulator in autonomous mode gets the predicted steering angle data from the drive.py program which uses the 
model in model.h5 and the current camera image for the prediction.

The final_run directory of data and final_run.mp4 video were produced with the following commands.

```
python drive.py model.h5 final_run
python video.py final_run -fps 48

```

![][video]

