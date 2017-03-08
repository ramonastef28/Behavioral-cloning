#importing all the libraries
from PIL import Image
import csv
import numpy as np
import os
import pandas as pd
import cv2
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import matplotlib
import matplotlib.pyplot as plt


####Reading the training, validation and test data
names = ['center', 'steering', 'throttle', 'brake', 'speed']
valid_data = pd.read_csv( './data/valid/driving_log.csv', names=names)
test_data = pd.read_csv('./data/test/driving_log.csv', names=names)
train_file = pd.read_csv('./data/train/driving_log.csv')
train_nonzero = train_file[train_file.steering != 0]

#retainig only a sample of the  zero valued images
train_zero = (train_file[train_file.steering == 0]).sample(frac=0.2)
train_data = pd.concat([train_nonzero, train_zero], ignore_index=True)

#Data augmentation - flipping the image
def flipped_image(img, y):
    img = cv2.flip(img, 1)
    return img, -y


#adding different britness intensity
def brightned_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    img[:,:,2] = img[:,:,2] * random_bright
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


#translate image with a random coeficient 
def translated_image(img, y, trans_range):
    rows, cols, _ = img.shape
    tr_x = trans_range * np.random.uniform() - trans_range/2
    y = y + tr_x/trans_range * 2 *.4
    tr_y = 10 * np.random.uniform() - 10/2
    Trans_M = np.float32([[1,0, tr_x], [0,1, tr_y]])
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    return img, y


#apply augumentation to only 50% of the data
def image_augmentation(img, y):
    if np.random.uniform() < 0.5:
        img, y = flipped_image(img, y)
    img = brightned_image (img)
    img, y = translated_image(img, y, 100)
    return img, y


#use only some part of the image 
def image_transformation(img):
    img = img[60:-20,:,:]
    img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
    return img

input_shape = (64, 64, 3)


#generating the dataset 

#the training set
def get_image_steering_train(row, folder):
    imgpath = row.center.values[0]
    imgpath = imgpath[imgpath.find('IMG'):]
    #print(folder + imgpath)
    img = cv2.imread(folder + imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    steering = row.steering.values[0]
    return img, steering

#validation and testing set
def get_img_and_steering(row, folder):
    imgpath = row.center
    imgpath = imgpath[imgpath.find('IMG'):]
    img = cv2.imread(folder + imgpath)
    #print(folder+imgpatg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    steering = row.steering
    return img, steering


def train_data_generator(batch_size):
    while True:
        X = np.zeros((batch_size, *input_shape), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.float32)
        for idx in range(batch_size):
            row = train_data.sample()
            img, steering = get_image_steering_train(row, './data/train/')
            img, steering = image_augmentation(img, steering)
            img = image_transformation(img)
            X[idx], y[idx] = img, steering
        yield X, y

def valid_data_generator(batch_size):
    seq_idx = 0
    while True:
        X = np.zeros((batch_size, *input_shape), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.float32)
        for idx in range(batch_size):
            row = valid_data.iloc[seq_idx]
            img, steering = get_img_and_steering(row, './data/valid/')
            img = image_transformation(img)
            X[idx], y[idx] = img, steering
            
            seq_idx += 1
            if seq_idx == len(valid_data):
                seq_idx = 0
        yield X, y


def test_data_generator(batch_size):
    seq_idx = 0
    while True:
        X = np.zeros((batch_size, *input_shape), dtype=np.float32)
        y = np.zeros((batch_size,), dtype=np.float32)
        for idx in range(batch_size):
            row = test_data.iloc[seq_idx]
            img, steering = get_img_and_steering(row, './data/test/')
            img = image_transformation(img)
            X[idx], y[idx] = img, steering
            
            seq_idx += 1
            if seq_idx == len(test_data):
                seq_idx = 0
        yield X, y


#3 1X1 filters, followed by 3 convolutional blocks each comprised of 32, 64 and 128 filters of size 3X3. These convolution layers were followed by 3 fully connected layers. All the convolution blocks and the 2 following fully connected layers had exponential relu (ELU) as activation function

def CNN_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))
    model.add(Convolution2D(3,1,1,  border_mode='valid', name='conv0', init='he_normal'))
    model.add(Convolution2D(32,3,3, border_mode='valid', name='conv1', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(32,3,3, border_mode='valid', name='conv2', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(64,3,3, border_mode='valid', name='conv3', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, border_mode='valid', name='conv4', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(128,3,3, border_mode='valid', name='conv5', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(128,3,3, border_mode='valid', name='conv6', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512,name='hidden1', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(64,name='hidden2', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(16,name='hidden3',init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1, name='output', init='he_normal'))
    model.compile(optimizer="adam", loss="mse")
    return model

model = CNN_model()
#baches of size 50 trained the model for 10 epoch with 20k examples in each
model.fit_generator(
    train_data_generator(50),
    samples_per_epoch=20000,
    nb_epoch=10,
    validation_data=valid_data_generator(250),
    nb_val_samples=750,
    callbacks=[ModelCheckpoint(filepath="best_valid_score.h5", verbose=1, save_best_only=True)]
)


#saving the model
import json
with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('model.h5')



