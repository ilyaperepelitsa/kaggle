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
working_params = hyper_params["external_2"]





def get_batch(X, y, size):
    a = np.random.choice(X.index, size, replace = False)
    return X.loc[X.index.isin(a)], y.loc[y.index.isin(a)]

def add_variable_summary(tf_variable, summary_name):
  with tf.name_scope(summary_name + '_summary'):
    mean = tf.reduce_mean(tf_variable)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('standard_deviation'):
        standard_deviation = tf.sqrt(tf.reduce_mean(
            tf.square(tf_variable - mean)))
    tf.summary.scalar('StandardDeviation', standard_deviation)
    tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
    tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
    tf.summary.histogram('Histogram', tf_variable)


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
# y
labels = encoder.active_features_

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#





input_size = x.shape[1]
input_size
no_classes = len(labels)
no_classes

def one_hot_to_indices(data):
    indices = []
    for el in data:
        indices.append(list(el).index(1))
    return indices


def convolution_layer(input_layer, filters,
                        kernel_size=[3,3],
                        activation=tf.nn.relu):

    layer = tf.layers.conv2d(inputs = input_layer,
                                filters = filters,
                                kernel_size = kernel_size,
                                activation = activation
                                )

    add_variable_summary(layer, 'convolution')
    return layer


def pooling_layer(input_layer, pool_size=[2, 2], strides=2):
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides
    )
    add_variable_summary(layer, 'pooling')
    return layer


def dense_layer(input_layer, units, activation=tf.nn.relu):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation
    )
    add_variable_summary(layer, 'dense')
    return layer

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape)
    variable = tf.Variable(initial)
    # add_variable_summary(variable, "weight")
    return variable

def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape)
    variable = tf.Variable(initial)
    # add_variable_summary(variable, "bias")
    return variable

def conv2d(x, w, strides = [1, 1, 1, 1], padding = "SAME"):
    variable = tf.nn.conv2d(x, w, strides = strides, padding = padding)
    # add_variable_summary(variable, "convolution")
    return variable

def max_pool_2x2(x, padding = "SAME"):
    variable = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding=padding)
    # add_variable_summary(variable, "pooling")
    return variable

def avg_pool_2x2(x, padding = "SAME"):
    variable = tf.nn.avg_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding=padding)
    # add_variable_summary(variable, "pooling")
    return variable

def conv_layer(input, shape):
    w = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, w) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    w = weight_variable([in_size, size])
    b = bias_variable([size])
    variable = tf.matmul(input, w) + b
    add_variable_summary(variable, "dense_layer")
    return variable

g = tf.Graph()

with g.as_default():
    with tf.name_scope("x_input") as scope:
        x_input = tf.placeholder(dtype = tf.float32, name = "x_input", shape = [None, input_size])

    with tf.name_scope("y_input") as scope:
        y_input = tf.placeholder(dtype = tf.float32, name = "y_input", shape = [None, no_classes])

    with tf.name_scope("reshape") as scope:
        x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1],
                                     name='input_reshape')



    with tf.name_scope("stack_1") as scope:

        conv_1 = conv_layer(x_input_reshape, shape = [5, 5, 1, 128])
        conv_1_pool = max_pool_2x2(conv_1)

        conv_2 = conv_layer(conv_1_pool, shape = [5, 5, 128, 64])
        conv_2_pool = max_pool_2x2(conv_2)

        # conv_2_flat = tf.reshape(conv_2_pool, [-1,])
        pooling_shape = conv_2_pool.get_shape().as_list()
        conv_2_flat = tf.reshape(conv_2_pool, [-1, pooling_shape[1] * pooling_shape[2] * pooling_shape[3]])


    with tf.name_scope("stack_2") as scope:

        conv_1_1 = conv_layer(x_input_reshape, shape = [5, 5, 1, 128])
        conv_1_1_pool = avg_pool_2x2(conv_1_1)

        conv_2_1 = conv_layer(conv_1_1_pool, shape = [5, 5, 128, 64])
        conv_2_1_pool = avg_pool_2x2(conv_2_1)

        # conv_2_flat = tf.reshape(conv_2_pool, [-1,])
        pooling_shape_1 = conv_2_1_pool.get_shape().as_list()
        conv_2_1_flat = tf.reshape(conv_2_1_pool, [-1, pooling_shape_1[1] * pooling_shape_1[2] * pooling_shape_1[3]])

    with tf.name_scope("stack_3") as scope:

        conv_1_2 = conv_layer(x_input_reshape, shape = [3, 3, 1, 128])
        conv_1_2_pool = avg_pool_2x2(conv_1_2)

        conv_2_2 = conv_layer(conv_1_2_pool, shape = [3, 3, 128, 64])
        conv_2_2_pool = avg_pool_2x2(conv_2_2)

        # conv_2_flat = tf.reshape(conv_2_pool, [-1,])
        pooling_shape_2 = conv_2_2_pool.get_shape().as_list()
        conv_2_2_flat = tf.reshape(conv_2_2_pool, [-1, pooling_shape_2[1] * pooling_shape_2[2] * pooling_shape_2[3]])

    with tf.name_scope("stack_4") as scope:

        conv_1_3 = conv_layer(x_input_reshape, shape = [3, 3, 1, 128])
        conv_1_3_pool = max_pool_2x2(conv_1_3)

        conv_2_3 = conv_layer(conv_1_3_pool, shape = [3, 3, 128, 64])
        conv_2_3_pool = max_pool_2x2(conv_2_3)

        pooling_shape_3 = conv_2_3_pool.get_shape().as_list()
        conv_2_3_flat = tf.reshape(conv_2_3_pool, [-1, pooling_shape_3[1] * pooling_shape_3[2] * pooling_shape_3[3]])

    with tf.name_scope("flatten") as scope:
        full_1 = tf.nn.relu(full_layer(conv_2_flat, 1024))
        full_2 = tf.nn.relu(full_layer(conv_2_1_flat, 1024))
        full_3 = tf.nn.relu(full_layer(conv_2_2_flat, 1024))
        full_4 = tf.nn.relu(full_layer(conv_2_3_flat, 1024))


    with tf.name_scope("concat") as scope:
        concat_layer = tf.concat([full_1, full_2, full_3, full_4], 1)

    with tf.name_scope("dropout") as scope:
        keep_prob = tf.placeholder(dtype = tf.float32)
        full_1_drop = tf.nn.dropout(concat_layer, keep_prob=keep_prob)

    with tf.name_scope("logits") as scope:
        logits = full_layer(full_1_drop, no_classes)



    with tf.name_scope('loss'):
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_input, logits=logits)
        loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
        tf.summary.scalar('loss', loss_operation)

    with tf.name_scope('optimiser'):
        optimiser = tf.train.AdamOptimizer().minimize(loss_operation)


    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            predictions = tf.argmax(logits, 1)
            correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
        with tf.name_scope('accuracy'):
            accuracy_operation = tf.reduce_mean(
                tf.cast(correct_predictions, tf.float32))
    tf.summary.scalar('accuracy', accuracy_operation)

    with tf.Session() as session:
        saver = tf.train.Saver()
        # session = tf.Session()

        try:
            saver.restore(session, os.path.join(LOGS_PATH, working_params["logs"], "model.ckpt"))
        except tf.errors.InvalidArgumentError:
            session.run(tf.global_variables_initializer())


        merged_summary_operation = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, working_params["summary"], "train"), session.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, working_params["summary"], "test"))


        test_images, test_labels = x_test.values, y_test.values

        for batch_no in range(working_params["num_steps"]):

            batch_x, batch_y = get_batch(x_train, y_train, working_params["minibatch_size"])
            train_images = batch_x.values
            train_labels = batch_y.values

            _, merged_summary = session.run([optimiser, merged_summary_operation],
                                            feed_dict={
                x_input: train_images,
                y_input: train_labels,
                keep_prob : working_params["dropout_keep_prob"]
                # dropout_bool: True

            })
            train_summary_writer.add_summary(merged_summary, batch_no)

            # print(session.run(concat, feed_dict={
            #         x_input: train_images,
            #         y_input: train_labels,
            #         keep_prob : 0.5
            #         }))

            if batch_no % 10 == 0:
                saver.save(session, os.path.join(LOGS_PATH, working_params["logs"], "model.ckpt"))

                merged_summary, _ = session.run([merged_summary_operation,
                                                 accuracy_operation], feed_dict={
                    x_input: test_images,
                    y_input: test_labels,
                    keep_prob : 1.0
                    # dropout_bool: False

                })
                test_summary_writer.add_summary(merged_summary, batch_no)

            if batch_no % 100 == 0:

                chunk_size = 5000
                list_df = [target_x[i:i+chunk_size] for i in range(0,target_x.shape[0],chunk_size)]

                for data_chunk in list_df:

                    submission = session.run(predictions, feed_dict = {x_input: data_chunk.values, keep_prob : 1})
                    submission = pd.DataFrame({"ImageId": data_chunk.index.values, "Label": submission })
                    # print(submission.shape)

                    submission.to_csv(os.path.join(
                            os.path.dirname(
                                    os.path.abspath("__file__")
                            ),
                            'mnist/data/' + working_params["name"] + "_" + str(batch_no) +
                            '_submission.csv'
                        ),
                        index=False,
                        mode = "a")
