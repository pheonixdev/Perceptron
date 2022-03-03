
"""This is a program that implements binary perceptron and extends the binary perceptron for multi class classification.
Binary perceptron is used to classify between three unique classes ('class-1','class-2','class-3'). The binary
perceptron is extended using the one vs rest approach for multi-class classification. The training and testing
accuracies are computed and compared for all the scenarios. L2 regularisation is added to the multi-class classifier and
trained accordingly. The accuracies are then computed for different values of regularisation coefficient.
"""
import numpy as np
import pandas as pd

SEED = 30


def user_input():
    """Function to get the choice from the user based on what classifier they would like to check the accuracy on.

    :return:
        user_choice : the choice of classifier entered by the user
    """
    print("1. One vs Two\n"
          "2. Two vs Three\n"
          "3. One vs Three\n"
          "4. Multiclass classifier using one vs rest approach\n"
          "5. Multiclass classifier with L2 regularization\n")
    user_choice = input("Select the classifier you would like to try:\n")

    # if user enters an invalid choice, programs exits after printing an error message
    if user_choice not in '12345':
        print("Invalid input. Exiting program..")
        exit()
    return user_choice


def clean_data(train_set, test_set, classifier):
    """Function where the train data and the test data are cleaned based on the requirement of the classifier.

    :param train_set: training dataset
    :param test_set: testing dataset
    :param classifier: user input for the choice of classifier
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

    elif classifier in '45':
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

    else:
        print("Invalid classifier!")
        user_input()
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
            if label * activation_score <= 0:
                w = (1 - 2 * lamb) * w + label * feature
                b = b + label
    return b, w


def perceptron_test(b, w, data, multi_class=False):
    """Function to test the perceptron

    :param b:
    :param w:
    :param data:
    :param multi_class:
    :return:
    """
    y_pred = np.ones(len(data))
    for d in data:
        feature = d[:-1]
        label = d[-1:]
        activation_score = np.dot(w, feature) + b
        y_predicted = np.sign(activation_score)
        if multi_class:
            y_pred[np.all(data == d, axis=1)] = activation_score
        else:
            y_pred[np.all(data == d, axis=1)] = y_predicted
    if not multi_class:
        accuracy(y_pred.astype(int), data[:, -1])
    return y_pred


def accuracy(y_p, y_t):
    print("Predicted label: ", y_p)
    print("True label: \t", y_t)
    print("Accuracy: ", np.mean(y_p == y_t))


def convert_label(data):
    label = data[:, -1]
    label[label == 'class-1'] = 0
    label[label == 'class-2'] = 1
    label[label == 'class-3'] = 2
    return label


def multiclass_predict(w, data):
    y_1 = perceptron_test(w[0], w[1], data, True)
    y_2 = perceptron_test(w[2], w[3], data, True)
    y_3 = perceptron_test(w[4], w[5], data, True)
    y_stack = np.vstack((y_1, y_2, y_3)).T
    return np.argmax(y_stack, axis=1)


def compute_weightage(lamb):
    t_1, t_2, t_3 = clean_data(train_set, test_set, choice)
    b_1, w_1 = perceptron_train(t_1, 20, lamb)
    b_2, w_2 = perceptron_train(t_2, 20, lamb)
    b_3, w_3 = perceptron_train(t_3, 20, lamb)
    return [b_1, w_1, b_2, w_2, b_3, w_3]


if __name__ == '__main__':

    train_set = np.array(pd.read_csv("C:/Users/sabar/Desktop/UoL stuff/Sem2/COMP527/CA1data/train.data", header=None))
    test_set = np.array(pd.read_csv("C:/Users/sabar/Desktop/UoL stuff/Sem2/COMP527/CA1data/test.data", header=None))

    choice = user_input()
    if choice in '123':
        clean_train_data, clean_test_data = clean_data(train_set, test_set, choice)
        bias, weights = perceptron_train(clean_train_data, 20)
        perceptron_test(bias, weights, clean_test_data)

    elif choice == '4':
        weightage = compute_weightage(0)
        print("\nMulticlass classification on the Training Set")
        y_pred = multiclass_predict(weightage, train_set)
        y_true = convert_label(train_set)
        accuracy(y_pred, y_true)
        print("\nMulticlass classification on the Testing Set")
        y_pred = multiclass_predict(weightage, test_set)
        y_true = convert_label(test_set)
        accuracy(y_pred, y_true)

    elif choice == '5':
        np.warnings.filterwarnings('ignore')
        print("Adding L2 Regularization and computing the accuracies\n")
        print("\nRegularisation coefficient = 0.01")
        weightage = compute_weightage(0.01)
        print("Multiclass classification on the Training Set")
        y_pred = multiclass_predict(weightage, train_set)
        y_true = convert_label(train_set)
        accuracy(y_pred, y_true)
        print("\nMulticlass classification on the Testing Set")
        y_pred = multiclass_predict(weightage, test_set)
        y_true = convert_label(test_set)
        accuracy(y_pred, y_true)

        print("\nRegularisation coefficient = 0.1")
        weightage = compute_weightage(0.1)
        print("Multiclass classification on the Training Set")
        y_pred = multiclass_predict(weightage, train_set)
        y_true = convert_label(train_set)
        accuracy(y_pred, y_true)
        print("\nMulticlass classification on the Testing Set")
        y_pred = multiclass_predict(weightage, test_set)
        y_true = convert_label(test_set)
        accuracy(y_pred, y_true)

        print("\nRegularisation coefficient = 1.0")
        weightage = compute_weightage(1.0)
        print("Multiclass classification on the Training Set")
        y_pred = multiclass_predict(weightage, train_set)
        y_true = convert_label(train_set)
        accuracy(y_pred, y_true)
        print("\nMulticlass classification on the Testing Set")
        y_pred = multiclass_predict(weightage, test_set)
        y_true = convert_label(test_set)
        accuracy(y_pred, y_true)

        print("\nRegularisation coefficient = 10")
        weightage = compute_weightage(10)
        print("Multiclass classification on the Training Set")
        y_pred = multiclass_predict(weightage, train_set)
        y_true = convert_label(train_set)
        accuracy(y_pred, y_true)
        print("\nMulticlass classification on the Testing Set")
        y_pred = multiclass_predict(weightage, test_set)
        y_true = convert_label(test_set)
        accuracy(y_pred, y_true)

        print("\nRegularisation coefficient = 100")
        weightage = compute_weightage(100)
        print("Multiclass classification on the Training Set")
        y_pred = multiclass_predict(weightage, train_set)
        y_true = convert_label(train_set)
        accuracy(y_pred, y_true)
        print("\nMulticlass classification on the Testing Set")
        y_pred = multiclass_predict(weightage, test_set)
        y_true = convert_label(test_set)
        accuracy(y_pred, y_true)
