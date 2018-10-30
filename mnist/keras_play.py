import tensorflow as tf

import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "mnist")
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")),"mnist", "tf_specs.json")
LOGS_PATH = os.path.join(ROOT_PATH, "logs")
SUMMARY_PATH = os.path.join(ROOT_PATH, "summary")
METRICS_PATH = os.path.join(ROOT_PATH, "metrics.json")


def check_dir_create(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

check_dir_create(LOGS_PATH)
hyper_params = json.load(open(JSON_PATH))
working_params = hyper_params["keras_play"]

LOGS_PATH = os.path.join(LOGS_PATH, working_params["logs"])
FINAL_LOGS = os.path.join(LOGS_PATH, "weights.best.hdf5")
EPOCHS = working_params["num_steps"]
RATE = working_params["dropout_keep_prob"]
BATCH_SIZE = working_params["minibatch_size"]

def check_dir_create(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

check_dir_create(LOGS_PATH)


def get_batch(X, y, size):
    a = np.random.choice(X.index, size, replace = False)
    return X.loc[X.index.isin(a)], y.loc[y.index.isin(a)]


full_data = pd.read_csv(
                os.path.join(
                        os.path.dirname(
                                os.path.abspath("__file__")
                        ),'mnist/data/train.csv'
                    ))

target_x = pd.read_csv(
                os.path.join(
                        os.path.dirname(
                                os.path.abspath("__file__")
                        ),'mnist/data/test.csv'
                    ))

target_x.index+1
target_x = target_x.set_index(target_x.index + 1)
target_x.values.shape


x = full_data[full_data.columns[full_data.columns!="label"]]
y = pd.DataFrame(full_data["label"])
# no_classes = y.label.unique().shape[0]
# y = y.values
# y
encoder = OneHotEncoder()
encoder.fit(y)

y = pd.DataFrame(encoder.transform(y).toarray())
labels = encoder.active_features_

y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)

y_train = y_train.values
y_test = y_test.values

input_shape = (28, 28, 1)


input_size = x.shape[1]
# input_size
no_classes = len(labels)


model = Sequential()
# model.add(Conv2D(
#     filters = 64,
#     kernel_size = (3, 3),
#     kernel_initializer = keras.initializers.TruncatedNormal(),
#     bias_initializer = keras.initializers.TruncatedNormal(),
#     activation = "relu",
#     input_shape = input_shape))
#
# model.add(Conv2D(
#     filters = 128,
#     kernel_size = (3, 3),
#     kernel_initializer = keras.initializers.TruncatedNormal(),
#     bias_initializer = keras.initializers.TruncatedNormal(),
#     activation = "relu"))
#
# model.add(MaxPool2D(pool_size = (2, 2)))
# model.add(Dropout(rate = RATE))
# model.add(Flatten())
# model.add(Dense(units = 1024, activation = "relu"))
# model.add(Dropout(rate = RATE))
# model.add(Dense(units = no_classes, activation = "softmax"))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1),
                     kernel_initializer = keras.initializers.TruncatedNormal(),
                     bias_initializer = keras.initializers.TruncatedNormal()))
# model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
#                  activation ='relu',
#                      kernel_initializer = keras.initializers.TruncatedNormal(),
#                      bias_initializer = keras.initializers.TruncatedNormal()))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.2))


model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',
                 activation ='relu',
                     kernel_initializer = keras.initializers.TruncatedNormal(),
                     bias_initializer = keras.initializers.TruncatedNormal()))
model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(rate = 0.2))
# model.add(Conv2D(filters = 128, kernel_size = (7,7),padding = 'Same',
#                  activation ='relu',
#                      kernel_initializer = keras.initializers.TruncatedNormal(),
#                      bias_initializer = keras.initializers.TruncatedNormal()))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(rate = 0.2))


model.add(Flatten())
model.add(Dropout(rate = RATE))
model.add(Dense(1024, activation = "relu",
    kernel_initializer = keras.initializers.TruncatedNormal(),
    bias_initializer = keras.initializers.TruncatedNormal()))

model.add(Dense(units = no_classes, activation = "softmax"))

try:
    model.load_weights(FINAL_LOGS)
except OSError:
    pass
model.compile(loss = keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.Adam(),
                metrics = ['accuracy'])

checkpoint = ModelCheckpoint(FINAL_LOGS, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# model.fit(x = x_train, y = y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (x_test, y_test), verbose = 2)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
#                                             patience=3,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.0001)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(x_train)
# h = model.fit_generator(datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
#                               epochs = EPOCHS, validation_data = (x_test,y_test),
#                               verbose = 2, steps_per_epoch=x_train.shape[0] // BATCH_SIZE
#                               , callbacks=[learning_rate_reduction],)
#
h = model.fit_generator(datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
                              epochs = EPOCHS, validation_data = (x_test,y_test),
                              verbose = 2, steps_per_epoch=1,
                              callbacks=callbacks_list)

train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose = 0)
print("Train data loss: ", train_loss)
print("Train data accuracy: ", train_accuracy)
