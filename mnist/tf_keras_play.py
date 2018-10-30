import tensorflow as tf

import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd

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
working_params = hyper_params["tf_keras_play"]

EPOCHS = working_params["num_steps"]
RATE = working_params["learning_rate"]
BATCH_SIZE = working_params["minibatch_size"]



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

encoder = OneHotEncoder()
encoder.fit(y)

y = pd.DataFrame(encoder.transform(y).toarray())
labels = encoder.active_features_

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)

input_shape = (28, 28, 1)


input_size = x.shape[1]
input_size
no_classes = len(labels)
no_classes

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    activation = "relu",
    input_shape = input_shape))

model.add(tf.keras.layers.Conv2D(
    filters = 128,
    kernel_size = (3, 3),
    activation = "relu"))

model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(tf.keras.layers.Dropout(rate = RATE))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units = 1024, activation = "relu"))
model.add(tf.keras.layers.Dropout(rate = RATE))
model.add(tf.keras.layers.Dense(units = no_classes, activation = "softmax"))
model.compile(loss = tf.keras.losses.categorical_crossentropy,
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ['accuracy'])
model.fit(x_train, y_train, BATCH_SIZE, EPOCHS, (x_test, y_test))

train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose = 0)
print("Train data loss: ", train_loss)
print("Train data accuracy: ", train_accuracy)
