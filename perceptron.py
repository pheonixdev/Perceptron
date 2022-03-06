"""This is a program that implements binary perceptron and extends the binary perceptron for multi class classification.
Binary perceptron is used to classify between three unique classes ('class-1','class-2','class-3'). The binary
perceptron is extended using the one vs rest approach for multi-class classification. The training and testing
accuracies are computed and compared for all the scenarios. L2 regularisation is added to the multi-class classifier and
trained accordingly. The accuracies are then computed for different values of regularisation coefficient.
"""
import numpy as np
import pandas as pd

# seed value is set to 30 for shuffling
SEED = 30


def clean_data(train_set, test_set, classifier):
    """Function where the train data and the test data are cleaned based on the requirement of the classifier.

    :param train_set: training dataset
    :param test_set: testing dataset
    :param classifier: the choice of classifier for cleaning the dataset
    :return: returns the cleaned train data and test data
    """
    train_data = train_set.copy()
    test_data = test_set.copy()
    np.random.seed(SEED)
    # one vs two classifier
    if classifier == '1':
        train_data = train_data[np.where(train_data[:, -1] != 'class-3')]
        train_data[train_data == 'class-1'] = 1
        train_data[train_data == 'class-2'] = -1
        test_data = test_data[np.where(test_data[:, -1] != 'class-3')]
        test_data[test_data == 'class-1'] = 1
        test_data[test_data == 'class-2'] = -1
    # two vs three classifier
    elif classifier == '2':
        train_data = train_data[np.where(train_data[:, -1] != 'class-1')]
        train_data[train_data == 'class-2'] = 1
        train_data[train_data == 'class-3'] = -1
        test_data = test_data[np.where(test_data[:, -1] != 'class-1')]
        test_data[test_data == 'class-2'] = 1
        test_data[test_data == 'class-3'] = -1
    # one vs three classifier
    elif classifier == '3':
        train_data = train_data[np.where(train_data[:, -1] != 'class-2')]
        train_data[train_data == 'class-1'] = 1
        train_data[train_data == 'class-3'] = -1
        test_data = test_data[np.where(test_data[:, -1] != 'class-2')]
        test_data[test_data == 'class-1'] = 1
        test_data[test_data == 'class-3'] = -1
    elif classifier == '4':
        # one vs rest classifier
        train_data_1 = train_set.copy()
        train_data_1[train_data_1 == 'class-1'] = 1
        train_data_1[(train_data_1 == 'class-2') | (train_data_1 == 'class-3')] = -1
        np.random.shuffle(train_data_1)
        # two vs rest classifier
        train_data_2 = train_set.copy()
        train_data_2[train_data_2 == 'class-2'] = 1
        train_data_2[(train_data_2 == 'class-1') | (train_data_2 == 'class-3')] = -1
        np.random.shuffle(train_data_2)
        # three vs rest classifier
        train_data_3 = train_set.copy()
        train_data_3[train_data_3 == 'class-3'] = 1
        train_data_3[(train_data_3 == 'class-1') | (train_data_3 == 'class-2')] = -1
        np.random.shuffle(train_data_3)
        return train_data_1, train_data_2, train_data_3
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    return train_data, test_data


def perceptron_train(data, max_iter, lamb=0):
    """Function to train the weights and bias of the perceptron

    :param data: dataset for training the perceptron
    :param max_iter: the maximum number of epochs
    :param lamb: regularisation coefficient
    :return: bias and weights
    """
    w = np.zeros(len(data[0]) - 1)
    b = 0
    for _ in range(max_iter):
        for d in data:
            feature = d[:-1]
            label = d[-1:]
            activation_score = np.dot(w, feature) + b
            # weights and bias are adjusted in case of misclassification
            if label * activation_score <= 0:
                w = (1 - 2 * lamb) * w + label * feature
                b = b + label
    return b, w


def perceptron_test(b, w, data, multi_class=False):
    """Function to test the perceptron

    :param b: bias
    :param w: weights
    :param data: dataset for testing the perceptron
    :param multi_class: boolean value to classify between binary and multi class perceptron
    :return: predicted label in case of binary and activation score in case of multi class
    """
    y_pred = np.ones(len(data))
    for d in data:
        feature = d[:-1]
        label = d[-1:]
        activation_score = np.dot(w, feature) + b
        # sign function returns 1 or -1 depending on the activation score
        y_predicted = np.sign(activation_score)
        if multi_class:
            y_pred[np.all(data == d, axis=1)] = activation_score
        else:
            y_pred[np.all(data == d, axis=1)] = y_predicted
    if not multi_class:
        # accuracy of binary perceptron
        accuracy(y_pred.astype(int), data[:, -1])
    return y_pred


def accuracy(y_p, y_t):
    """Function to calculate the accuracy of perceptron

    :param y_p: predicted label
    :param y_t: true label
    :return: accuracy of the perceptron
    """
    # print("Predicted label: ", y_p)
    # print("True label: \t", y_t)
    print("Accuracy: ", np.mean(y_p == y_t))


def convert_label(data):
    """Function to convert the class names to integers.

    :param data: the dataset for which the label is to be converted
    :return: the labels as integers
    """
    label = data[:, -1]
    label[label == 'class-1'] = 0
    label[label == 'class-2'] = 1
    label[label == 'class-3'] = 2
    return label


def multiclass_predict(w, data):
    """Function to perform the perceptron testing for multi class classifier.

    :param w: bias and weights of the models after training
    :param data: dataset to test the models
    :return: the class index with the maximum activation function
    """
    y_1 = perceptron_test(w[0], w[1], data, True)
    y_2 = perceptron_test(w[2], w[3], data, True)
    y_3 = perceptron_test(w[4], w[5], data, True)
    y_stack = np.vstack((y_1, y_2, y_3)).T
    return np.argmax(y_stack, axis=1)


def compute_weightage(lamb):
    """Function to compute the bias and weights of the models.

    :param lamb: regularisation coefficient
    :return: bias and weights of the models
    """
    t_1, t_2, t_3 = clean_data(train_set, test_set, '4')
    b_1, w_1 = perceptron_train(t_1, 20, lamb)
    b_2, w_2 = perceptron_train(t_2, 20, lamb)
    b_3, w_3 = perceptron_train(t_3, 20, lamb)
    return [b_1, w_1, b_2, w_2, b_3, w_3]


def compute_binary():
    """Function to compute the accuracy of binary perceptron.

    """
    clean_train_data, clean_test_data = clean_data(train_set, test_set, choice)
    bias, weights = perceptron_train(clean_train_data, 20)
    perceptron_test(bias, weights, clean_test_data)


def compute_multiclass(lamb):
    """Function to compute the accuracy of multiclass perceptron.

    :param lamb: regularisation coefficient
    """
    train_data = train_set.copy()
    test_data = test_set.copy()
    weightage = compute_weightage(lamb)
    print("Training Set")
    y_pred = multiclass_predict(weightage, train_data)
    y_true = convert_label(train_data)
    accuracy(y_pred, y_true)
    print("Testing Set")
    y_pred = multiclass_predict(weightage, test_data)
    y_true = convert_label(test_data)
    accuracy(y_pred, y_true)


if __name__ == '__main__':
    train_set = np.array(pd.read_csv("C:/Users/sabar/Desktop/UoL stuff/Sem2/COMP527/CA1data/train.data", header=None))
    test_set = np.array(pd.read_csv("C:/Users/sabar/Desktop/UoL stuff/Sem2/COMP527/CA1data/test.data", header=None))

    print("1. Class One vs Class Two")
    choice = '1'
    compute_binary()
    print("2. Class Two vs Class Three")
    choice = '2'
    compute_binary()
    print("3. Class One vs Class Three")
    choice = '3'
    compute_binary()
    print("4. Multiclass classifier using one vs rest approach")
    np.warnings.filterwarnings('ignore')
    choice = '4'
    compute_multiclass(0)
    print("5.1 Multiclass classifier using regularisation with a regularisation coefficient = 0.01")
    compute_multiclass(0.01)
    print("5.2 Multiclass classifier using regularisation with a regularisation coefficient = 0.1")
    compute_multiclass(0.1)
    print("5.3 Multiclass classifier using regularisation with a regularisation coefficient = 1")
    compute_multiclass(1)
    print("5.4 Multiclass classifier using regularisation with a regularisation coefficient = 10")
    compute_multiclass(10)
    print("5.5 Multiclass classifier using regularisation with a regularisation coefficient = 100")
    compute_multiclass(100)
