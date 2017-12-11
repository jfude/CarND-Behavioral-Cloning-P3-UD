import os
import csv
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers


## All images must be in the ./IMG directory


## Read CSV file lines
def samples_read(lines_in,dir):    
    with open(dir + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines_in.append(line)


## Set samples here, samples is a list of the lines in the csv file
samples = []
samples_read(samples,"./")


## Set Train/Test split
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)



import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle



def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
               for i in range(1): 
                  name = './IMG/'+batch_sample[i].split('/')[-1]
                  img     =  cv2.imread(name)
                  ## Convert to RGB to be consistent with drive.py & simulator
                  c_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                  ## Steering angle
                  c_angle = float(batch_sample[3])
                  images.append(c_image)
                  angles.append(c_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # image format


model = Sequential()
# Normalize data
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(row, col,ch)))
# Crop image to ignore car and sky/background
model.add(Cropping2D(cropping=((70,25), (0,0))))
# NN Architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10)) 
model.add(Dense(1)) # Output is steering angle only


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            samples_per_epoch=len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=6)

model.save('model.h5')

