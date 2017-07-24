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

Throughout the data collection process I tried to move the car at a constant speed. Apart from data collection other techniques like reversing the images etc. are done to make 
the model more robust. All in all, the final data set has roughly twenty seven thousand samples. Data distribution can be seen below:

![DataSummary](https://github.com/suji0131/Behavioral_Cloning/blob/master/Images/Data_Summary.png)

## Generator
Below generator is used to generate samples of data for each run in a epoch.
```
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = batch_sample[0]
                center_angle = batch_sample[1]
                images.append(center_image)
                angles.append(center_angle)
                
            X_train = np.array(images) 
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=300)
validation_generator = generator(validation_samples, batch_size=300)
```

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

As it can be seen from the video generated using my model.h5 the car stay on the track. In most of the runs the car more or less stayed in the middle. However, I would like to see how the car would perform when it close by the lane. Furthermore, as a future work I would like to include the gas and brake data into the model for pratical purposes.

