"""
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer
This is the FFN
compare output to intended output > cost function (cross entropy because of classification)
optimization function (optimizer) > mimimize cost (adamoptimizer, SGD, AdaGrad)
backpropagation
feed forward + backprop = epoch
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

n_classes = 3
n_nodes_hl1 = 90
n_nodes_hl2 = 90
batch_size = 100
test_size = 0.1

# [height, width] no height because we flatten it out
x = tf.placeholder('float', [None, 17])  # input data
y = tf.placeholder('float')  # label of the data


def one_hot_encode(values):
    classes = ["GALAXY", "STAR", "QSO"]
    encodings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    print("Encoding the classes")
    return [list(encodings[classes.index(v)]) for v in values]


def parse_data(filename):
    data = pd.read_csv(filename)
    print("Data read")
    slice_index = int(len(data) * test_size)
    training_data = data[:-slice_index]
    test_data = data[-slice_index:]
    test_features = list(test_data.loc[:, test_data.columns != "class"].values)
    train_features = list(training_data.loc[:, training_data.columns != "class"].values)
    sc = StandardScaler()
    train_features = sc.fit_transform(train_features)
    test_features = sc.transform(test_features)
    return {
        "test_labels": list(one_hot_encode(test_data["class"].values)),
        "test_features": test_features,
        "train_labels": list(one_hot_encode(training_data["class"].values)),
        "train_features": train_features,
    }


def neural_network_model(data):
    """
    models a neural network. basically done with the 'computation graph'
    """
    print("Creating model...")
    hidden_1_layer = {
        "weights": tf.Variable(tf.random_normal([17, n_nodes_hl1])),
        "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))
    }

    # hidden_2_layer = {
    #     "weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #     "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))
    # }

    output_layer = {
        "weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
        "biases": tf.Variable(tf.random_normal([n_classes]))
    }

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)  # relu if that is your activation function

    # l2 = tf.add(
    #     tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    # l2 = tf.nn.relu(l2)

    output = tf.matmul(l1, output_layer["weights"]) + output_layer["biases"]
    print("Finished creating model...")
    return output


def train_neural_network(x):
    """
    x = input data
    """
    print("Parsing the data")
    data = parse_data('sloan-digital-sky-survey/dataset.csv')
    print("Training the neural network")
    prediction = neural_network_model(x)
    print("Setting cost")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # cost function used to update weights in backprop
    # adam optimizer takes a learning_rate parameter, default is 0.001
    print("Setting optimizer")
    optimizer = tf.train.AdamOptimizer().minimize(cost)  # AdamOptimizer is synonymous with stochastic gradient descent
    # how many epochs = cycles of feed forward + backprop
    hm_epochs = 100
    with tf.Session() as sess:
        print("Starting the session")
        sess.run(tf.global_variables_initializer())
        # This section trains the network
        for epoch in range(hm_epochs):
            print("Starting epoch {}".format(epoch))
            epoch_loss = 0
            i = 0
            while i < len(data["train_features"]):
                start = i
                end = i + batch_size
                batch_x = np.array(data["train_features"][start:end])
                batch_y = np.array(data["train_labels"][start:end])
                i += batch_size
                # c is the cost
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
            print("Epoch {} completed out of {}, loss: {}".format(epoch, hm_epochs, epoch_loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # Check the test data against the trained network
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy: {}".format(accuracy.eval({x: data["test_features"], y: data["test_labels"]})))


train_neural_network(x)