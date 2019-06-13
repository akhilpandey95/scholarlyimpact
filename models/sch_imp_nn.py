# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import plaidml.keras
plaidml.keras.install_backend()

import sys
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.layers import Dense
from collections import Counter
from keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# function for processing the dataset
def data_processing(file_path):
    """
    Process the dataset and prepare it for using it against the neural network model

    Parameters
    ----------
    arg1 | file_path: str
        The file path indicating the location of the dataset

    Returns
    -------
    Dataframe
        pandas.DataFrame

    """
    try:
        # read the dataset
        data = pd.read_csv(file_path, low_memory=False)

        # create the label encoder
        encoder = LabelEncoder()

        # transform all the columns
        data.Type = encoder.fit_transform(data.Type)

        # transform the column profession
        data.Profession = encoder.fit_transform(data.Profession)

        # transform the column academic status
        data.AcademicStatus = encoder.fit_transform(data.AcademicStatus)

        # transform the column platform with max mentions
        data.PlatformWithMaxMentions = encoder.fit_transform(data.PlatformWithMaxMentions)

        # create a target variable for the first experiment
        data = data.assign(target_exp_1 =
                           list(map(lambda x: 1 if x > 0 else 0, tqdm(data['citations']))))

        # create a target variable for the first experiment
        data = data.assign(target_exp_2 =
                           list(map(lambda x: 1 if x > 9 else 0, tqdm(data['citations']))))

        # create a target variable for the first experiment
        data = data.assign(target_exp_3 =
                           list(map(lambda x: np.log(1 + x), tqdm(data['citations']))))

        # drop the columns unecessary
        data = data.drop(columns=['Type', 'citations', 'citations(Log_Transformed)'])

        # return the dataframe
        return data
    except:
        return pd.DataFrame()

# function for preparing the X & Y for the dataset
def prepare_X_Y(data_frame):
    """
    Process the dataframe and return the X and Y for the experiment

    Parameters
    ----------
    arg1 | data_frame: pandas.DataFrame
        A loaded dataframe for preparing X and Y

    Returns
    -------
    Tuple
        numpy.ndarray, numpy.ndarray

    """
    try:
        # the following data columns will be considered as features
        data_columns = ['mendeley', 'citeulike', 'News', 'Blogs',
                        'Reddit', 'Twitter', 'Facebook',
                        'GooglePlus', 'PeerReviews','Wikipedia',
                        'TotalPlatforms', 'SincePublication','PlatformWithMaxMentions',
                        'Countries', 'MaxFollowers', 'Retweets','Profession',
                        'AcademicStatus', 'PostLength', 'HashTags', 'Mentions',
                        'AuthorCount']

        # set the X column
        X = data_frame.as_matrix(columns = data_columns)

        # set the target variable
        Y = data_frame.target_exp_3

        # return the tuple
        return X, Y
    except:
        return np.zeros((len(data_frame), 22)), np.zeros((len(data_frame), 22))

# function for preparing the X & Y for the dataset
def build_model():
    """
    Build the Vanilla style neural network model and compile it

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        keras.model.Model

    """
    try:
        # create the model
        model = Sequential()

        # add the first hidden layer with 64 neurons, relu activation
        model.add(Dense(64, activation='relu', input_dim=22))

        # add the second hidden layer with 64 neurons, relu activation
        model.add(Dense(64, activation='relu'))

        # add the single output layer
        model.add(Dense(1))

        # use the rmsprop optimizer
        rms = keras.optimizers.RMSprop(lr=0.001)

        # compile the model
        model.compile(optimizer=rms, loss='mean_squared_error',
                           metrics =['mean_absolute_error', 'mean_squared_error'])

        # return the model
        return model
    except:
        return keras.models.Model()

# function for training the neural network model
def train(model, X_train, X_test, Y_train, Y_test, stopping=True):
    """
    Fit the neural network model

    Parameters
    ----------
    arg1 | model: keras.model.Model
        A compiled keras neural network model to train
    arg2 | X_train: numpy.ndarray
        The training samples containing all the predictors
    arg3 | X_test: numpy.ndarray
        The test samples containing all the predictors
    arg4 | Y_train: numpy.ndarray
        The training samples containing values for the target variable
    arg5 | Y_test: numpy.ndarray
        The test samples containing values for the target variable
    arg6 | stopping: boolean
    A flag asserting if early stopping should or shouldn't be used for training

    Returns
    -------
    Neural Network Model
        keras.model.Model

    """
    try:
        if not stopping:
            # fit the model
            model.fit(X_train, Y_train, epochs=100, validation_split=0.2)
        else:
            # prepare for early stopping
            early_stopping = keras.callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0,
                                                     patience=40, verbose=0, mode='auto',
                                                     baseline=None, restore_best_weights=False)
            # fit the model
            model.fit(X_train, Y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

        # return the model
        return model
    except:
        return keras.models.Model()

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
            evaluation = model.evaluate(data['X_train'], data['Y_train'])
        else:
            # evaluate the model and print the training stats
            evaluation = model.evaluate(data['X_test'], data['Y_test'])

        # return the model
        return evaluation
    except:
        return np.zeros(3)

# function for printing metrics for the model
def metrics(model, **data):
    """
    Predict the test samples and return the metrics for the model

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
        y_pred = model.predict(data['X_test'])

        # print the total sum of squared errors and the residuals
        ssres = np.sum(np.square(data['Y_test'] - list(map(lambda x: x[0], y_pred))))
        sstot = np.sum(np.square(data['Y_test'] - np.mean(list(map(lambda x: x[0], y_pred)))))

        # calculate the r-squared
        r2 = 1 - ssres/sstot

        # return the r-squared value
        return r2
    except:
        return np.zeros(1)[0]

if __name__ == '__main__':
    # load the dataset
    data = data_processing('~/Downloads/sch_impact.csv')

    # prepare the X, Y
    X, Y = prepare_X_Y(data)

    # build the train and test samples
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # build the model
    regressor = build_model()

    # train the model
    regressor = train(regressor, X_train, X_test, Y_train, Y_test)

    # evaluate and print the training stats
    model_evaluation = evaluate(regressor, 'train', x_train=X_train, y_train=Y_train)

    # print training metrics
    print('Training Loss:', model_evaluation[0])
    print('Training MAE:', model_evaluation[1])
    print('Training MSE:', model_evaluation[2])

    # evaluate and print the test stats
    model_evaluation = evaluate(regressor, 'test', x_test=X_test, y_test=Y_test)

    # print test metrics
    print('Test Loss:', model_evaluation[0])
    print('Test MAE:', model_evaluation[1])
    print('Test MSE:', model_evaluation[2])

    # print the r-squared
    print('R-squared:', metrics(regressor, x_test=X_test, y_test=Y_test))

else:
    print('ERR: unable to run the script')
    sys.exit(0)
