
# coding: utf-8

# In[6]:

import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import sklearn
import pickle

from sklearn.model_selection import train_test_split
from random import shuffle


xy_tup = []
graph_dat = []

'''
Each run creates data in different folder. This function extracts the information from 
these different folders and it makes it into a unified single data set.
'''
def data_sum(name):
    #reading the data
    col_names = ['Ctr','Lt','Rt','s_angle','throttle','break','speed']
    df = pd.read_csv('Data/'+name+'_Data/driving_log.csv', header=None, names=col_names)
    #print(len(df))
    #print(df.s_angle.value_counts(bins=20, sort=False))
    
    for i in range(len(df.index)):
        angle = df['s_angle'][i]
        
        if name == 'Sam' or angle != 0:
            #update to new address
            source_path = df['Ctr'][i]
            filename = source_path.split('/')[-1]
            current_path = 'Data/'+name+'_Data/IMG/'+filename #New_path = '../data/IMG/'

            image = cv2.imread(current_path)
            tuple_temp = (image,angle)
            xy_tup.append(tuple_temp)
            graph_dat.append(angle)
            if angle !=0: #reversing the image
                tuple_temp_inv = (np.fliplr(image),-angle)
                xy_tup.append(tuple_temp_inv)
                graph_dat.append(-angle)


names = ['Sam','Nor','Rev','Tra']
for nam in names:
    print(nam)
    data_sum(nam)


#plotting histogram of steering angles
sns.distplot(graph_dat, bins=20,kde=False, color='green')
#print(df.s_angle.value_counts(bins=20, sort=False))



# In[7]:

train_samples, val_test_samples = train_test_split(xy_tup, test_size=0.3)

validation_samples, text_samples = train_test_split(xy_tup, test_size=0.7)

# In[8]:

#python generator
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
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=300)
validation_generator = generator(validation_samples, batch_size=300)


# In[9]:

import keras

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[11]:

model = Sequential()

'''50 rows pixels from the top of the image, 20 rows pixels from the bottom of the image,0 columns of pixels
from the left of the image 0 columns of pixels from the right of the image are cropped'''
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

#Adding lambda layer that normalizes and mean centers the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#convolutional layers
"Layer-1"
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid')) #subsample means stride
model.add(Activation('elu'))
#Dropout layer
#model.add(Dropout(0.75))

"Layer-2"
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid'))
model.add(Activation('elu'))
#Dropout layer
#model.add(Dropout(0.75))

"Layer-3"
model.add(Convolution2D(48,5,5, border_mode='valid')) #stride by default is (1,1)
model.add(Activation('elu'))
#Dropout layer
#model.add(Dropout(0.75))

"Layer-4"
model.add(Convolution2D(60,3,3, border_mode='valid')) #stride by default is (1,1)
#2x2 max-pooling layer
model.add(MaxPooling2D((2, 2)))
model.add(Activation('elu'))
#Dropout layer
#model.add(Dropout(0.5))

"Layer-5"
model.add(Convolution2D(72,3,3, border_mode='valid')) #stride by default is (1,1)
#2x2 max-pooling layer
model.add(MaxPooling2D((2, 2)))
model.add(Activation('elu'))
#Dropout layer
#model.add(Dropout(0.5))

#flattening the layer
model.add(Flatten())

"Vanilla neural network"
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('elu'))

model.add(Dense(1))

#Adam Optimizer is used.
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#model.fit(X,Y, batch_size=100, epochs=1, validation_split=0.2, verbose=1)

hist_obj = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),validation_data=validation_generator,nb_val_samples=len(validation_samples),nb_epoch=2, verbose=1)

model.save('model.h5')

print(model.summary())

