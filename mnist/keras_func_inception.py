import tensorflow as tf

import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Input, Concatenate

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.callbacks import TensorBoard

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "mnist")
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")),"mnist", "tf_specs.json")
LOGS_PATH = os.path.join(ROOT_PATH, "logs")
SUMMARY_PATH = os.path.join(ROOT_PATH, "summary")
METRICS_PATH = os.path.join(ROOT_PATH, "metrics.json")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def check_dir_create(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

check_dir_create(LOGS_PATH)
hyper_params = json.load(open(JSON_PATH))
working_params = hyper_params["keras_func_inception"]

LOGS_PATH = os.path.join(LOGS_PATH, working_params["logs"])
FINAL_LOGS = os.path.join(LOGS_PATH, "weights.best.hdf5")
EPOCHS = working_params["num_steps"]
RATE = working_params["dropout_keep_prob"]
BATCH_SIZE = working_params["minibatch_size"]
FLAT_SIZE = working_params["flat_size"]
LEARNING_RATE = working_params["learning_rate"]

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
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# x_train.values.reshape(x_train.shape[0], 28, 28, 1).shape

#
# x_train = x_train.values.reshape(-1, 28, 28, 1)
# x_test = x_test.values.reshape(-1, 28, 28, 1)
#
# y_train = y_train.values
# y_test = y_test.values
#
# input_shape = (28, 28, 1)

if K.image_data_format() == 'channels_first': # Theano backend
    x_train = x_train.values.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.values.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:                                         # Tensorflow backend
    x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

input_size = x.shape[1]
# input_size
no_classes = len(labels)

K.clear_session()
reshaped_input = Input(shape = input_shape)

tower_0 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(reshaped_input)

tower_1 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(reshaped_input)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu', input_shape = input_shape)(tower_1)

tower_2 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(reshaped_input)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu', input_shape = input_shape)(tower_2)

tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same', input_shape = input_shape)(reshaped_input)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(tower_3)


output_inception = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
drop_1 = Dropout(rate = RATE)(output_inception)

tower_0_1 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_1)

tower_1_1 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_1)
tower_1_1 = Conv2D(64, (3,3), padding='same', activation='relu', input_shape = input_shape)(tower_1_1)

tower_2_1 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_1)
tower_2_1 = Conv2D(64, (5,5), padding='same', activation='relu', input_shape = input_shape)(tower_2_1)

tower_3_1 = MaxPooling2D((3,3), strides=(1,1), padding='same', input_shape = input_shape)(drop_1)
tower_3_1 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(tower_3_1)

output_inception_1 = keras.layers.concatenate([tower_0_1, tower_1_1, tower_2_1, tower_3_1], axis = 3)
drop_2 = Dropout(rate = RATE)(output_inception_1)

tower_0_2 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_2)

tower_1_2 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_2)
tower_1_2 = Conv2D(64, (3,3), padding='same', activation='relu', input_shape = input_shape)(tower_1_2)

tower_2_2 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_2)
tower_2_2 = Conv2D(64, (5,5), padding='same', activation='relu', input_shape = input_shape)(tower_2_2)

tower_3_2 = MaxPooling2D((3,3), strides=(1,1), padding='same', input_shape = input_shape)(drop_2)
tower_3_2 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(tower_3_2)

output_inception_2 = keras.layers.concatenate([tower_0_2, tower_1_2, tower_2_2, tower_3_2], axis = 3)
drop_3 = Dropout(rate = RATE)(output_inception_2)

tower_0_3 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_3)

tower_1_3 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_3)
tower_1_3 = Conv2D(64, (3,3), padding='same', activation='relu', input_shape = input_shape)(tower_1_3)

tower_2_3 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(drop_3)
tower_2_3 = Conv2D(64, (5,5), padding='same', activation='relu', input_shape = input_shape)(tower_2_3)

tower_3_3 = MaxPooling2D((3,3), strides=(1,1), padding='same', input_shape = input_shape)(drop_3)
tower_3_3 = Conv2D(64, (1,1), padding='same', activation='relu', input_shape = input_shape)(tower_3_3)

output_inception_3 = keras.layers.concatenate([tower_0_3, tower_1_3, tower_2_3, tower_3_3], axis = 3)

drop_4 = Dropout(rate = RATE)(output_inception_3)

flatten_inception = Flatten()(drop_4)
output_inception_final = Dense(FLAT_SIZE, activation = "relu",
                                kernel_initializer = "truncated_normal",
                                bias_initializer = "zeros")(flatten_inception)

# concat_layer = Concatenate(axis = -1)([output_1, output_2, output_inception_final])
drop = Dropout(rate = RATE)(output_inception_final)
norm_4 = BatchNormalization()(drop)
output = Dense(units = no_classes, activation = "softmax",
    kernel_initializer = "truncated_normal",
    bias_initializer = "truncated_normal")(norm_4)

### OUTPUT

model = Model(inputs = reshaped_input, outputs = output)

try:
    model.load_weights(FINAL_LOGS)
except OSError:
    pass
model.compile(loss = keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.Adam(0.001),
                metrics = ['accuracy'])

checkpoint = ModelCheckpoint(FINAL_LOGS, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

tensorbard = TensorBoard(log_dir=os.path.join(LOGS_PATH, "board"), histogram_freq=1,
          write_graph=True, write_images=False)
# tbCallback.set_model(model)

callbacks_list = [checkpoint, tensorbard]
# model.fit(x = x_train, y = y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (x_test, y_test), verbose = 2)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
#                                             patience=3,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.0001)

datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=True,  # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        # zoom_range = 0.2, # Randomly zoom image
        # shear_range = 0.2,
        # width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
# h = model.fit_generator(datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
#                               epochs = EPOCHS, validation_data = (x_test,y_test),
#                               verbose = 2, steps_per_epoch=x_train.shape[0] // BATCH_SIZE
#                               , callbacks=[learning_rate_reduction],)
#
h = model.fit_generator(datagen.flow(x_train,y_train,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    # save_to_dir = LOGS_PATH,
                                    save_to_dir = None),
                              epochs = EPOCHS, validation_data = (x_test,y_test),
                              verbose = 1,
                              callbacks=callbacks_list)




y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(y_pred, axis = 1)
Y_true = np.argmax(y_test.values, axis = 1)
# Y_true
# Y_true
# Y_pred_classes
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = range(10))









train_loss, train_accuracy = model.evaluate(x_test, y_test, verbose = 0)
print("Train data loss: ", train_loss)
print("Train data accuracy: ", train_accuracy)
