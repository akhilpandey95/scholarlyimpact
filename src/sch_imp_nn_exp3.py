# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import sys
from models import PredictLogCitation
from evaluation import evaluate, metrics
from data import data_processing, prepare_X_Y
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # load the dataset
    data = data_processing('~/Downloads/sch_impact.csv')

    # prepare the X, Y
    X, Y = prepare_X_Y(data)

    # build the train and test samples
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # build the model
    regressor = PredictLogCitation()

    # train the model
    regressor = regressor.train(10, 64, X_train, X_test, Y_train, Y_test, stopping=False)

    # evaluate and print the training stats
    model_evaluation = evaluate(regressor, 'train', x_train=X_train, y_train=Y_train)

    # print training metrics
    print('Training Loss(MSE):', model_evaluation[0])
    print('Training MAE:', model_evaluation[1])

    # evaluate and print the test stats
    model_evaluation = evaluate(regressor, 'test', x_test=X_test, y_test=Y_test)

    # print test metrics
    print('Test Loss(MSE):', model_evaluation[0])
    print('Test MAE:', model_evaluation[1])

    # print the r-squared
    print('R-squared:', metrics(regressor, x_test=X_test, y_test=Y_test))

else:
    print('ERR: unable to run the script')
    sys.exit(0)
