import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

n_classes = 3
n_nodes_hl1 = 90
batch_size = 100
test_size = 0.1
NUM_RUNS = 1

# [height, width] no height because we flatten it out
x = tf.placeholder('float', [None, 17])  # input data
y = tf.placeholder('float')  # label of the data


def one_hot_encode(values):
    classes = ["GALAXY", "STAR", "QSO"]
    encodings = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    print("Encoding the classes")
    return [list(encodings[classes.index(v)]) for v in values]


def parse_data(filename):
    data = pd.read_csv(filename).sample(frac=1)
    print("Data read")
    slice_index = int(len(data) * test_size)
    training_data = data[:-slice_index]
    test_data = data[-slice_index:]
    test_features = list(test_data.loc[:, test_data.columns != "class"].values)
    train_features = list(training_data.loc[:, training_data.columns != "class"].values)
    sc = RobustScaler()
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

    output_layer = {
        "weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
        "biases": tf.Variable(tf.random_normal([n_classes]))
    }

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)  # relu if that is your activation function

    output = tf.matmul(l1, output_layer["weights"]) + output_layer["biases"]
    print("Finished creating model...")
    return output


def train_neural_network(x):
    """
    x = input data
    """
    print("Training the neural network")
    prediction = neural_network_model(x)
    print("Setting cost")
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # cost function used to update weights in backprop
    # adam optimizer takes a learning_rate parameter, default is 0.001
    print("Setting optimizer")
    optimizer = tf.train.AdamOptimizer().minimize(cost)  # AdamOptimizer is synonymous with stochastic gradient descent
    results = []
    for i in range(NUM_RUNS):
        print("RUN #{}".format(i))
        print("Parsing the data")
        data = parse_data('sloan-digital-sky-survey/dataset.csv')
        # how many epochs = cycles of feed forward + backprop
        hm_epochs = 100
        with tf.Session() as sess:
            print("Starting the session")
            sess.run(tf.global_variables_initializer())
            # This section trains the network
            run_results = {
                "test_accuracy": 0,
                "train_accuracy": 0,
                "epoch_loss": [],
            }

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            train_features = data["train_features"][:]
            train_labels = data["train_labels"][:]
            for epoch in range(hm_epochs):
                epoch_loss = 0
                i = 0
                while i < len(train_features):
                    start = i
                    end = i + batch_size
                    batch_x = np.array(train_features[start:end])
                    batch_y = np.array(train_labels[start:end])
                    i += batch_size
                    # c is the cost
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                run_results["epoch_loss"].append(epoch_loss)
                print("Epoch {} completed out of {}, loss: {}".format(epoch, hm_epochs, epoch_loss))

            # Check the test data against the trained network
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            run_results["test_accuracy"] = accuracy.eval({x: data["test_features"], y: data["test_labels"]})
            print("Test Accuracy: {}".format(run_results["test_accuracy"]))
            run_results["train_accuracy"] = accuracy.eval({x: data["train_features"], y: data["train_labels"]})
            print("Train Accuracy: {}".format(run_results["train_accuracy"]))
            run_results["correct"] = correct.eval({x: data["train_features"], y: data["train_labels"]})
            results.append(run_results)
        print("=" * 50)
    return results


results = train_neural_network(x)