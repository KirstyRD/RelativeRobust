#keras 1, theano backend

from __future__ import absolute_import, division, print_function

import os
from keras import *
from keras.models import load_model
import h5py
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.models import Sequential, model_from_json
import numpy as np
from keras.utils import np_utils
from keras.optimizers import RMSprop

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from gtsrb_gray import *
import pandas as pd

#network/data constants
batch_size = 128
nb_classes = 43
epochs = 200
img_channels = 1
img_rows, img_cols = 48, 48
nb_filters = 32
nb_pool = 2
nb_conv = 3
lr = 0.01

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))


#input shape
inputShape = (img_channels, img_rows, img_cols)

##option 1: specify model architecture again
model = Sequential()

model.add(Flatten(input_shape=inputShape))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.summary()

X, Y = read_dataset()

model.fit(X, Y,
          batch_size=batch_size,
          nb_epoch=epochs,
          validation_split=0.2,
          callbacks=[LearningRateScheduler(lr_schedule),
                     ModelCheckpoint('model2.h5', save_best_only=True)]
          )

X_test, Y_test = read_test_dataset()


# load weights form file
model.save_weights('networks/gtsrb/weights_gtsrb2_keras1_gray.h5')
print("Model saved!")

##evaluate
#test = pd.read_csv('networks/gtsrb/GT-final_test.csv', sep=';')
## Load test dataset
#X_test = []
#y_test = []
#i = 0
#for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
#    img_path = os.path.join('networks/gtsrb/Final_Test/Images/', file_name)
#    X_test.append(preprocess_img(io.imread(img_path)))
#    y_test.append(class_id)
#
#X_test = np.array(X_test)
#y_test = np.array(y_test)

# predict and evaluate
print(str(X_test.shape))
print(str(Y_test.shape))
##transform.resize(x, (1, IMG_SIZE,IMG_SIZE))
y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred == Y_test) / np.size(y_pred)
print("Test accuracy = {}".format(acc))




# load weights form file
model.save_weights('networks/gtsrb/weights_gtsrb2_keras1_gray.h5')
print("Model saved!")
