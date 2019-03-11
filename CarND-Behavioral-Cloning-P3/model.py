# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 22:48:47 2018

@author: Kang Rong
"""

import os
import cv2
import csv
import numpy as np
import sklearn
import matplotlib.image as mpimg
from keras.models import Sequential,Model
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout,Input
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

##define image augmentation functions
def random_flip(image,steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand()<0.5:
        image=cv2.flip(image,1)
        steering_angle=-steering_angle
    return image,steering_angle

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    HSV=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    ratio=1.0+0.4*(np.random.rand()-0.5)
    HSV[:,:,2]=HSV[:,:,2]*ratio
    return cv2.cvtColor(HSV,cv2.COLOR_HSV2RGB)

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2c) forms a line
    # xm, ym gives all the locations of the image
    x1,y1=IMAGE_WIDTH*np.random.rand(),0
    x2,y2=IMAGE_WIDTH*np.random.rand(),IMAGE_HEIGHT
    xm,ym=np.mgrid[0:IMAGE_HEIGHT,0:IMAGE_WIDTH]
    
    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask=np.zeros_like(image[:,:,1])
    mask[(ym-y1)*(x2-x1)-(y2-y1)*(xm-x1)>0]=1
    
    # choose which side should have shadow and adjust saturation
    cond=mask==np.random.randint(2)
    s_ratio=np.random.uniform(low=0.2,high=0.5)
    
    # adjust Saturation in HLS(Hue, Light, Saturation)
    HLS=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    HLS[:,:,2][cond]=HLS[:,:,2][cond]*s_ratio
    
    return cv2.cvtColor(HLS,cv2.COLOR_HLS2RGB)
 
##define the genrators
def generator(samples,batch_size=32):
    num_samples=len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            
            images=[]
            angles=[]
            for batch_sample in batch_samples:
                correction=0.4
                #three camera images readed
                name='./data/data/IMG/'+batch_sample[0].split('/')[-1]
                #read BGR format image
                center_image=cv2.imread(name)
                center_image=cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                center_angle=float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                #add random_shadow data into training samples
                center_image1=random_shadow(center_image)
                center_angle1=float(batch_sample[3])
                images.append(center_image1)
                angles.append(center_angle1)
                
                # add random_brightness data into training samples
                center_image2=random_brightness(center_image)
                center_angle2=float(batch_sample[3])
                images.append(center_image2)
                angles.append(center_angle2)
                
                # add random_flip data into training samples
                center_image3,center_angle3=random_flip(center_image,center_angle)
                images.append(center_image3)
                angles.append(center_angle3)
                
                
                ###############################left###########################
                name='./data/data/IMG/'+batch_sample[1].split('/')[-1]
                #read BGR format image
                left_image=cv2.imread(name)
                left_image=cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
                left_angle=float(batch_sample[3])+correction
                images.append(left_image)
                angles.append(left_angle)
                
                #add random_shadow data into training samples
                left_image1=random_shadow(left_image)
                left_angle1=float(batch_sample[3])+correction
                images.append(left_image1)
                angles.append(left_angle1)
                
                # add random_brightness data into training samples
                left_image2=random_brightness(left_image)
                left_angle2=float(batch_sample[3])+correction
                images.append(left_image2)
                angles.append(left_angle2)
                
                # add random_flip data into training samples
                left_image3,left_angle3=random_flip(left_image,left_angle)
                images.append(left_image3)
                angles.append(left_angle3)
                
                ###############################right###########################
                name='./data/data/IMG/'+batch_sample[2].split('/')[-1]
                #read BGR format image
                right_image=cv2.imread(name)
                right_image=cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                right_angle=float(batch_sample[3])-correction
                images.append(right_image)
                angles.append(right_angle)
                
                #add random_shadow data into training samples
                right_image1=random_shadow(right_image)
                right_angle1=float(batch_sample[3])-correction
                images.append(right_image1)
                angles.append(right_angle1)
                
                # add random_brightness data into training samples
                right_image2=random_brightness(right_image)
                right_angle2=float(batch_sample[3])-correction
                images.append(right_image2)
                angles.append(right_angle2)
                
                # add random_flip data into training samples
                right_image3,right_angle3=random_flip(right_image,right_angle)
                images.append(right_image3)
                angles.append(right_angle)
                X_train=np.array(images)
                y_train=np.array(angles)
            yield shuffle(X_train,y_train)

#define resize the image function, the resized shape was decided by experiment.        
def resize_img(image):
    from keras.backend import tf
    return tf.image.resize_images(image,(32,100))

'''
Main program 
'''
#import data from .csv
samples=[]
with open('./data/data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#split the data set into trainning data and validation data
train_samples,validation_samples=train_test_split(samples,test_size=0.2)   

#the information of the simulator data (160,320,3)
IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS=160,320,3
INPUT_SHAPE=(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS)   

# compile and train the model using the generator function
train_generator=generator(train_samples,batch_size=75)
validation_generator=generator(validation_samples,batch_size=75)

##Keras model refered to NVIDIA model. But I added Cropping2D and made some of changes.
model=Sequential()
#cropping the images to remove the noise from image, Keras only need to feed input shape in fisrt layer.
model.add(Cropping2D(cropping=((50,20),(2,2)),input_shape=(160,320,3)))
#resize the image data to accelerate the traning step.
model.add(Lambda(resize_img))
#normalize the data
model.add(Lambda(lambda x:x/127.5-1))
#convolution layer with 12 filters(size (5,5)), stride = (2,2)
model.add(Convolution2D(12,5,5,activation='relu',subsample=(2,2)))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#convolution layer with 32 filters(size (5,5)), stride = (2,2)
model.add(Convolution2D(32,5,5,activation='relu',subsample=(2,2)))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#convolution layer with 48 filters(size (5,5)), stride = (2,2)
model.add(Convolution2D(48,5,5,activation='relu',subsample=(2,2)))
#flatten the layer into fully connected layer
model.add(Flatten())
#add dense layer , and the number of node become 100
model.add(Dense(100,activation='relu'))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#add dense layer , and the number of node become 50
model.add(Dense(50,activation='relu'))
#add a dropout layer for preventing overfitting
model.add(Dropout(0.2))
#add dense layer , and the number of node become 10
model.add(Dense(10,activation='relu'))
#add dense layer , and the number of node become 1
model.add(Dense(1))
model.summary()
print(len(train_samples))
#train the model with generator
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)-129, validation_data=validation_generator,\
            nb_val_samples=len(validation_samples), nb_epoch=5)
#save model as model.h5
model.save('model.h5')