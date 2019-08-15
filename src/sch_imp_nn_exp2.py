# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import sys
from models import PredictMedianCitationsExist
from evaluation import evaluate, clf_metrics
from data import data_processing, prepare_X_Y
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # load the dataset
    data = data_processing('~/Downloads/sch_impact.csv')

    # prepare the X, Y
    X, Y = prepare_X_Y(data, 'target_exp_2')

    # build the train and test samples
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # build the model
    classifier = PredictMedianCitationsExist()

    # train the model
    classifier = classifier.train(10, 64, X_train, X_test, Y_train, Y_test, stopping=False)

    # evaluate and print the training stats
    model_evaluation = evaluate(classifier, 'train', x_train=X_train, y_train=Y_train)

    # print training metrics
    print('Training Loss:', model_evaluation[0])
    print('Training Accuracy:', model_evaluation[1])

    # print the test set metrcs [acc, prec, recall, f1]
    model_evaluation = clf_metrics(classifier, x_test=X_test, y_test=Y_test)

    # test accuracy
    print('Test accuracy:', model_evaluation[0])

    # precision
    print('Precision:', model_evaluation[1])

    # recall
    print('Recall:', model_evaluation[2])

    # f1
    print('F-1:', model_evaluation[3])
else:
    print('ERR: unable to run the script')
    sys.exit(0)
