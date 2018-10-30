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
working_params = hyper_params["basic_conv_20"]
# working_params

def get_batch(X, y, size):
    a = np.random.choice(X.index, size, replace = False)
    return X.loc[X.index.isin(a)], y.loc[y.index.isin(a)]



def add_variable_summary(tf_variable, summary_name):
    with tf.name_scope(summary_name + "_summary"):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar("Mean", mean)

        with tf.name_scope("standard_deviation"):
            standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable - mean)))
        tf.summary.scalar("StandardDeviation", standard_deviation)
        tf.summary.scalar("Maximum", tf.reduce_max(tf_variable))
        tf.summary.scalar("Minimum", tf.reduce_min(tf_variable))
        tf.summary.histogram("Histogram", tf_variable)

# def convolution_layer(input_layer, filters, kernel_size=[3,3], activation = tf.nn.relu):
#     layer = tf.layers.conv2d(inputs = input_layer,
#                                 filters = filters,
#                                 kernel_size = kernel_size,
#                                 activation = activation,
#                                 )
#     add_variable_summary(layer, "convolution")
#     return layer
#
# def pooling_layer(input_layer, pool_size = [2,2], strides = 2):
#     layer = tf.layers.max_pooling2d(
#                 inputs = input_layer,
#                 pool_size = pool_size,
#                 strides = strides
#         )
#     add_variable_summary(layer, "pooling")
#     return layer

def convolution_layer(input_layer, weight_shape, bias_shape, activation = tf.nn.relu):
    input_dim = weight_shape[0] * weight_shape[1] * weight_shape[2]
    # weight_init = tf.random_normal_initializer(stddev = (2.0/input_dim)**0.5)
    # bias_init = tf.constant_initializer(value = 0)

    # with tf.name_scope("weights") as scope:
    w = tf.Variable(tf.truncated_normal(weight_shape), dtype=tf.float32)
    # with tf.name_scope("biases") as scope:
    b = tf.Variable(tf.truncated_normal(bias_shape), dtype=tf.float32)

    # w = tf.get_variable("W", weight_shape, initializer = weight_init)
    # b = tf.get_variable("b", bias_shape, initializer = bias_init)

    conv_out = tf.nn.conv2d(input_layer, w, strides = [1, 1, 1, 1], padding="SAME")
    activated = activation(tf.nn.bias_add(conv_out, b))


    add_variable_summary(activated, "convolution")
    return activated

def pooling_layer(input_layer, k = 2, pooler = tf.nn.max_pool):
    pooled = pooler(input_layer, ksize = [1, k, k, 1],
                            strides = [1, k, k, 1], padding="SAME")
    add_variable_summary(pooled, "pooling")
    return pooled

def dense_layer(input_layer, units, activation = tf.nn.relu):
    layer = tf.layers.dense(
                inputs = input_layer,
                units = units,
                activation = activation
        )
    add_variable_summary(layer, "dense")
    return layer

################################################################################
################################################################################
################################################################################

full_data = pd.read_csv(
                os.path.join(
                        os.path.dirname(
                                os.path.abspath("__file__")
                        ),'mnist/data/train.csv'
                    ))
#
# full_data.head()

x = full_data[full_data.columns[full_data.columns!="label"]]
y = pd.DataFrame(full_data["label"])





encoder = OneHotEncoder()
encoder.fit(y)

y = pd.DataFrame(encoder.transform(y).toarray())
y
labels = encoder.active_features_


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape
x_test.shape
y_train.shape
y_test.shape


input_size = x.shape[1]
no_classes = len(labels)



def one_hot_to_indices(data):
    indices = []
    for el in data:
        indices.append(list(el).index(1))
    return indices



#
g = tf.Graph()
# true_i = 0

with g.as_default():
    x_input = tf.placeholder(tf.float32, shape = [None, input_size])
    y_input = tf.placeholder(tf.float32, shape = [None, no_classes])
    x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1], name = "input_reshape")


    with tf.name_scope("convolution_1") as scope:
        # convolution_layer_1 = convolution_layer(x_input_reshape, 64)
        convolution_layer_1 = convolution_layer(x_input_reshape, [3, 3, 1, 20], [20],
                                                    activation = tf.nn.relu)

    with tf.name_scope("pooling_1") as scope:
        pooling_layer_1 = pooling_layer(convolution_layer_1)

    with tf.name_scope("convolution_2") as scope:
        # convolution_layer_2 = convolution_layer(pooling_layer_1, 128)
        convolution_layer_2 = convolution_layer(pooling_layer_1, [5, 5, 20, 50], [50],
                                                    activation = tf.nn.relu)

    with tf.name_scope("pooling_2") as scope:
        pooling_layer_2 = pooling_layer(convolution_layer_2)

    # with tf.name_scope("convolution_3") as scope:
    #     # convolution_layer_3 = convolution_layer(pooling_layer_2, 256)
    #     convolution_layer_3 = convolution_layer(pooling_layer_2, [10, 10, 64, 64], [64],
    #                                                 activation = tf.nn.relu)
    #
    # with tf.name_scope("pooling_3") as scope:
    #     pooling_layer_3 = pooling_layer(convolution_layer_3)

    with tf.name_scope("flatten_pool") as scope:
        pooling_shape = pooling_layer_2.get_shape().as_list()
        flattened_pool = tf.reshape(pooling_layer_2, [-1, pooling_shape[1] * pooling_shape[2] * pooling_shape[3]])

    with tf.name_scope("bottleneck") as scope:
        dense_layer_bottleneck = dense_layer(flattened_pool, 1024)

    with tf.name_scope("dropout") as scope:
        # dropout_bool = tf.placeholder(tf.bool, name="dropout_bool")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        dropout_layer = tf.layers.dropout(
                inputs = dense_layer_bottleneck,
                rate = keep_prob
            )

    with tf.name_scope("logits") as scope:
        logits = dense_layer(dropout_layer, no_classes, activation=tf.nn.sigmoid)

    with tf.name_scope("loss") as scope:
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                    labels = y_input,
                                    logits = logits
                )
        loss_operation = tf.reduce_mean(softmax_cross_entropy, name="loss")
        tf.summary.scalar("loss", loss_operation)



    with tf.name_scope("train") as scope:
        learning_rate = working_params["learning_rate"]
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss_operation)

    with tf.name_scope("accuracy") as scope:
        with tf.name_scope("correct_prediction") as scope:
            predictions = tf.argmax(logits, 1)
            correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
        with tf.name_scope("accuracy") as scope:
            accuracy_operation = tf.reduce_mean(
                    tf.cast(correct_predictions, tf.float32)
                )
            tf.summary.scalar("accuracy", accuracy_operation)



        #
        # loss = tf.reduce_mean(tf.square(y_true - output_6))
        # true_loss = tf.sqrt(tf.reduce_mean( tf.square( tf.log(y_true + 1) - tf.log(output_6 + 1) )))

    # init = tf.global_variables_initializer()
    with tf.Session() as session:

    # session = tf.Session()


        saver = tf.train.Saver()
        learning_counter = 0
        prev_loss_test = 0
        current_loss_test = 0

        merged_summary_operation = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, working_params["summary"], "train"), session.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(SUMMARY_PATH, working_params["summary"], "test"))

        # img_d_summary_dir = os.path.join(SUMMARY_PATH, working_params["summary"], "summaries", "img")
        # img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, session.graph)



        try:
            saver.restore(session, os.path.join(LOGS_PATH, working_params["logs"], "model.ckpt"))
        except (tf.errors.InvalidArgumentError):
        # session.run(init)
            session.run(tf.global_variables_initializer())


        for step in range(working_params["num_steps"]):
            # print(step)
            batch_x, batch_y = get_batch(x_train, y_train, working_params["minibatch_size"])
            batch_x = batch_x.values
            # print(batch_x)
            batch_y = batch_y.values
            # print(batch_y)
            _, merged_summary = session.run([train, merged_summary_operation], feed_dict={
                x_input : batch_x,
                y_input : batch_y,
                keep_prob : working_params["dropout_keep_prob"]
            })

            train_summary_writer.add_summary(merged_summary, step)

            # print(session.run(dense_layer_bottleneck, feed_dict={x_input : x_test.values,
            #                                         y_input : y_test.values,}))

            if step % working_params["print_and_save"] == 0:
                saver.save(session, os.path.join(LOGS_PATH, working_params["logs"], "model.ckpt"))
                # if prev_loss_test == 0:
                #     prev_loss_test = test_loss
                #     current_loss_test = test_loss - 1
                # else :
                #     prev_loss_test = current_loss_test
                #     current_loss_test = test_loss

                # if current_loss_test > prev_loss_test:
                #     learning_counter += 1
                # else:
                #     learning_counter = 0


                # predict_labels = session.run(predictions, feed_dict={
                #         x_input : x_test.values,
                #         keep_prob : 0
                #     })
                #
                # real_values = one_hot_to_indices(y_test.values)
                #
                # con = tf.confusion_matrix(labels=real_values, predictions=predict_labels)
                #

                # print(real_values[0:25])
                # print(predict_labels[0:25])


                # img_d_summary = plot_confusion_matrix(y_test.values, predict_labels, labels, tensor_name='dev/cm')
                # img_d_summary_writer.add_summary(img_d_summary, step)

                merged_summary, _ = session.run([merged_summary_operation,
                                                accuracy_operation],
                                                feed_dict = {
                                                x_input : x_test.values,
                                                y_input : y_test.values,
                                                keep_prob : 0
                                                })

                test_summary_writer.add_summary(merged_summary, step)




                    # submission = sess.run(output_6, feed_dict = {x: target_x, keep_prob : 1})
                    # print(submission.shape)
                    # #
                    # submission = pd.DataFrame(submission).set_index(index_copy)
                    # #
                    # submission = submission.rename(columns = {0: "target "})
                    # print(submission.head())
                    #
                    # #
                    # submission.to_csv(os.path.join(
                    #                 os.path.dirname(
                    #                         os.path.abspath("__file__")
                    #                 ),
                    #                 'santander/data/' + working_params["name"] +
                    #                 '_submission.csv'
                    #             ))
                    # print(submission.shape)




                # try:
                #     with open(METRICS_PATH) as jsr:
                #         data = json.load(jsr)
                #     test_df = pd.read_json(METRICS_PATH, orient = "index")
                #     if test_df.loc[test_df.name == working_params["name"],:].empty:
                #         true_i = step
                #     else:
                #         true_i = int(test_df.loc[test_df.name == working_params["name"], "step"].sort_values()[-1]) + working_params["print_and_save"]
                #
                #     entry = dict(working_params)
                #     entry["step"] = true_i
                #     # entry["train_loss"] = int(train_loss)
                #     # entry["test_loss"] = int(test_loss)
                #     # entry["test_loss_rmsle"] = int(test_loss_rmsle)
                #
                #     data.update({working_params["name"] + "_" + str(true_i) : entry})
                #     # Write the file
                #
                # except FileNotFoundError:
                #     true_i = step
                #     entry = dict(working_params)
                #     entry["step"] = true_i
                #     # entry["train_loss"] = int(train_loss)
                #     # entry["test_loss"] = int(test_loss)
                #     # entry["test_loss_rmsle"] = test_loss_rmsle
                #     data = {working_params["name"] + "_" + str(true_i) : entry}
                #
                # with open(METRICS_PATH, 'w') as jsw:
                #     json.dump(data, jsw)
test_summary_writer.close()
train_summary_writer.close()
