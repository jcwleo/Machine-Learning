""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    # print(np.dot(data, weights[:-1]).shape, weights[-1].shape)
    y = sigmoid(np.dot(data, weights[:-1]) + weights[-1])

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    ce = -np.sum(targets * np.log(1-y+1e-8))
    y = (y < 0.5)
    y = y.astype(int)

    frac_correct = np.mean(y == targets)
    # print(y.shape,targets.shape)

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        f, df = logistic_pen(weights, data, targets, hyperparameters)

    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    mu = hyperparameters['weight decay']

    bias = np.expand_dims(weights[-1], axis=0)  # for concatenate

    if hyperparameters['weight_regularization']:
        f = -np.sum(targets * np.log(1 - sigmoid(np.dot(data, weights[:-1]) + weights[-1])+1e-8) + (1 - targets) * (
        np.log(sigmoid(np.dot(data, weights[:-1]) + weights[-1])+1e-8))) + mu * np.sum(
            np.dot(np.transpose(weights), weights))
        df = np.concatenate(
            (np.dot(np.transpose(data), targets - (sigmoid(-(np.dot(data, weights[:-1]) + weights[-1])))), bias),
            axis=0) - mu * weights

    else:
        f = -np.sum(targets * np.log(1 - sigmoid(np.dot(data, weights[:-1]) + weights[-1])+1e-8) + (1 - targets) * (
        np.log(sigmoid(np.dot(data, weights[:-1]) + weights[-1])+1e-8)))

        df = np.concatenate(
            (np.dot(np.transpose(data), targets - (sigmoid(-(np.dot(data, weights[:-1]) + weights[-1])))), bias),
            axis=0)

    return f, df
