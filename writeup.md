# Behavioral Cloning

## Data Collection
Clearly this project's main focus should be on data collection and implementation in Keras. I collected data normally at first, then I drove the 
car in reverse direction of the track to generalize it. As much of the data is localized around the zero, I used data from udacity with 
non-zero steering angle. To make the model more robust, I intentionally drove the car to edges and corrected the path, however, in this 
scenario I selected the useful data into training set. For example, if I drove the car to right side of the lane and corrected its path by 
turning to left then only positive steering angles are considered. We only need angles that are correct and we don't want the car to steer
towards the lanes. 

This image and its related data is not selected as it is near lane:
![NotSelected](https://github.com/suji0131/Behavioral_Cloning/blob/master/Images/center_2017_06_02_17_21_47_354.jpg)


However data related to correction maneuver are considered:
![Selected](https://github.com/suji0131/Behavioral_Cloning/blob/master/Images/center_2017_06_02_17_21_47_820.jpg)

Apart from data collection other techniques like changing the brightness, reversing the images etc. are done to make 
the model more robust. All in all, the final data set has roughly twenty seven thousand samples.
![DataSummary](https://github.com/suji0131/Behavioral_Cloning/blob/master/Images/Data_Summary.png)

## Architecture
Cropping of images and Normalization of pixel values are done inside the model itself.
```
model = Sequential()

'''50 rows pixels from the top of the image, 20 rows pixels from the bottom of the image,0 columns of pixels
from the left of the image 0 columns of pixels from the right of the image are cropped'''
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

#Adding lambda layer that normalizes and mean centers the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
```

Model uses five convolutional layers and three layers of vanilla neural networks. First three convolutional layers have five by five 
convolutions and padding is set at valid. First two convolutions have a stride of two by two. Rest of the convolutions are three by three 
and has a stride of one by one and valid padding. Output from final convolution layer is flattened and connected to a neural network layers. Dropout layers are added to avoid overfitting. ELU is used for activation in all the layers. Final layer has only a single neuron which gives us the steering angles. Adam optimizer is used to and the function we are minimizing is the mean squared error (mse). 
Network is trained in the training set for two epochs with batch sizes of three hundred.

