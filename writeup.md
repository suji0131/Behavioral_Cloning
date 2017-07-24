# Behavioral Cloning

## Data Collection
Clearly this project's main focus should be on data collection and implementation in Keras. I collected data normally at first, then I drove the 
car in reverse direction of the track to generalize it. As much of the data is localized around the zero, I used data from udacity with 
non-zero steering angle. To make the model more robust, I intentionally drove the car to edges and corrected the path, however, in this 
scenario I selected the useful data into training set. For example, if I drove the car to right side of the lane and corrected its path by 
turning to left then only positive steering angles are considered. We only need angles that are correct and we don't want the car to steer
towards the lanes. Apart from data collection other techniques like changing the brightness, reversing the images etc. are done to make 
the model more robust.

## Architecture
