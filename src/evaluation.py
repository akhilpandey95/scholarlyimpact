# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# function for evaluating the model and reporting stats
def evaluate(model, option, **data):
    """
    Fit the neural network model

    Parameters
    ----------
    arg1 | model: keras.model.Model
        A trained keras neural network model
    arg2 | option: str
        A flag asserting whether to evaluate either train or test samples
    arg3 | **data: variable function arguments
        The variable argument used for pulling the training or test data

    Returns
    -------
    Array
        numpy.ndarray

    """
    try:
        if option == 'train':
            # evaluate the model and print the training stats
            evaluation = model.evaluate(data['x_train'], data['y_train'])
        else:
            # evaluate the model and print the training stats
            evaluation = model.evaluate(data['x_test'], data['y_test'])

        # return the model
        return evaluation
    except:
        return np.zeros(3)

# function for printing regression metrics for the model
def reg_metrics(model, **data):
    """
    Predict the test samples and return the regression metrics for the model

    Parameters
    ----------
    arg1 | model: keras.model.Model
        A trained keras neural network model
    arg2 | **data: variable function arguments
        The variable argument used for pulling the training or test data

    Returns
    -------
    List
        [numpy.float64, numpy.float64, numpy.float64]

    """
    try:
        # evaluate the model and print the training stats
        evaluation = model.evaluate(data['x_test'], data['y_test'])

        # predict the model
        y_pred = model.predict(data['x_test'])

        # print the total sum of squared errors and the residuals
        ssres = np.sum(np.square(data['y_test'] - list(map(lambda x: x[0], y_pred))))
        sstot = np.sum(np.square(data['y_test'] - np.mean(list(map(lambda x: x[0], y_pred)))))

        # obtain the r-squared value
        r2 = np.around(1 - np.divide(ssres, sstot), decimals=5)

        # append the r2 to the evaluation list
        evaluation.append(r2)

        # return the three regression metrics [mse, mae, r2]
        return evaluation
    except:
        return np.zeros(3)

# function for printing classification metrics for the model
def clf_metrics(model, **data):
    """
    Predict the test samples and return the classification metrics for the model
    Parameters
    ----------
    arg1 | model: keras.model.Model
        A trained keras neural network model
    arg2 | **data: variable function arguments
        The variable argument used for pulling the training or test data
    Returns
    -------
    Float
        numpy.float64
    """
    try:
        # predict the model
        y_pred= model.predict(data['x_test'])

        # predict the values
        y_pred = (y_pred > 0.5)

        # test accuracy
        acc = accuracy_score(data['y_test'], y_pred)

        # precision
        prec = precision_score(data['y_test'], y_pred)

        # recall
        rec = recall_score(data['y_test'], y_pred)

        # f-1
        f1 = f1_score(data['y_test'], y_pred)

        # return the r-squared value
        return [acc, prec, rec, f1]
    except:
        return np.zeros(4)
