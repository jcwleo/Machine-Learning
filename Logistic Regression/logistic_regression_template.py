from check_grad import check_grad
from utils import *
from logistic import *
import numpy as np
import matplotlib.pyplot as plt


def run_logistic_regression(hyperparameters):
    # TODO specify training data
    if hyperparameters['train_small']:
        train_inputs, train_targets = load_train_small()
    else:
        train_inputs, train_targets = load_train()

    if hyperparameters['test']:
        test_inputs, test_targets = load_test()

    valid_inputs, valid_targets = load_valid()
    # print(train_inputs.shape)

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.randn(M + 1, 1) / 1000

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    # run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in range(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
              "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}".format(
            t + 1, f / N, cross_entropy_train, frac_correct_train * 100,
            cross_entropy_valid, frac_correct_valid * 100))
        logging[t] = [f / N, cross_entropy_train, frac_correct_train * 100, cross_entropy_valid,
                      frac_correct_valid * 100]
    if hyperparameters['test']:
        # Make a prediction on the valid_inputs.
        predictions_test = logistic_predict(weights, test_inputs)

        # Evaluate the prediction.
        cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)

        # print some stats
        print('-----------test stat--------------')
        print("TEST FRAC:{:2.2f}".format(
           frac_correct_test * 100))
    return logging


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,  # function to check
                      weights,
                      0.001,  # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
        'learning_rate': 0.01,
        'weight_regularization': True,  # boolean, True for using Gaussian prior on weights
        'num_iterations': 10000,
        'mu': 0.01,  # Regularization rate
        'train_small': False,    # boolean, True for using train small data
        'test': True,  # boolean, True for using Test data
        'plot': False   # boolean, True for plotting error graph
    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in range(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    if hyperparameters['plot']:
        # TODO generate plots
        plt.figure()
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.plot(logging[:, 0])
        plt.show()
