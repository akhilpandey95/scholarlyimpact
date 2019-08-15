# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential

# feedforward network for predicting if citations exist or not
class PredictCitationsExist(Model):
    """
    Class object for predicting if citations for a given
    scholarly paper exist or not

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        keras.model.Model

    """
    # function for preparing the X & Y for the dataset
    def __init__(self):
        """
        Build the Vanilla style neural network model and compile it

        Parameters
        ----------
        No arguments

        Returns
        -------
        Nothing
            None

        """
        # super class the keras model
        super(PredictCitationsExist, self).__init__()

        # create the model
        self.model = Sequential()

        # add the first hidden layer with 64 neurons, relu activation
        self.model.add(Dense(512, activation='selu', input_dim=22))

        # add the single output layer
        self.model.add(Dense(1, activation='softmax'))

        # use the rmsprop optimizer
        self.rms = keras.optimizers.RMSprop(lr=0.001)

        # compile the model
        self.model.compile(optimizer=self.rms, loss='binary_crossentropy', metrics =['accuracy'])

    # function for training the neural network model
    def train(self, epochs, batch_size, X_train, X_test, Y_train, Y_test, stopping=True):
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
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size)
            else:
                # prepare for early stopping
                early_stopping = keras.callbacks.EarlyStopping(monitor='binary_cross_entropy', min_delta=0,
                                                         patience=40, verbose=0, mode='auto',
                                                         baseline=None, restore_best_weights=False)
                # fit the model
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping])

            # return the model
            return self.model
        except:
            return keras.models.Model()

# feedforward network for predicting if citations more than median or not
class PredictMedianCitationsExist(Model):
    """
    Class object for predicting if citations for a given
    scholarly paper are more than the median number of
    citations or not

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        keras.model.Model

    """
    # function for preparing the X & Y for the dataset
    def __init__(self):
        """
        Build the Vanilla style neural network model and compile it

        Parameters
        ----------
        No arguments

        Returns
        -------
        Nothing
            None

        """
        # super class the keras model
        super(PredictMedianCitationsExist, self).__init__()

        # create the model
        self.model = Sequential()

        # add the first hidden layer with 64 neurons, relu activation
        self.model.add(Dense(64, activation='sigmoid', input_dim=22))

        # add the second hidden layer with 128 neurons, relu activation
        self.model.add(Dense(128, activation='selu'))

        # add the third hidden layer with 64 neurons, relu activation
        self.model.add(Dense(64, activation='sigmoid'))

        # add the single output layer
        self.model.add(Dense(1, activation='sigmoid'))

        # use the rmsprop optimizer
        self.rms = keras.optimizers.RMSprop(lr=0.001)

        # compile the model
        self.model.compile(optimizer=self.rms, loss='binary_crossentropy', metrics =['accuracy'])

    # function for training the neural network model
    def train(self, epochs, batch_size, X_train, X_test, Y_train, Y_test, stopping=True):
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
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size)
            else:
                # prepare for early stopping
                early_stopping = keras.callbacks.EarlyStopping(monitor='binary_cross_entropy', min_delta=0,
                                                         patience=40, verbose=0, mode='auto',
                                                         baseline=None, restore_best_weights=False)
                # fit the model
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping])

            # return the model
            return self.model
        except:
            return keras.models.Model()


# feedforward network for predicting log(1 + citations)
class PredictLogCitation(Model):
    """
    Class object for predicting Log(1 + citations) for a given
    scholarly paper

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        keras.model.Model

    """
    # function for preparing the X & Y for the dataset
    def __init__(self):
        """
        Build the Vanilla style neural network model and compile it

        Parameters
        ----------
        No arguments

        Returns
        -------
        Nothing
            None

        """
        # super class the keras model
        super(PredictLogCitation, self).__init__()

        # create the model
        self.model = Sequential()

        # add the first hidden layer with 32 neurons, relu activation
        self.model.add(Dense(32, activation='selu', input_dim=22))

        # add the second hidden layer with 64 neurons, relu activation
        self.model.add(Dense(64, activation='selu'))

        # add the third hidden layer with 64 neurons, relu activation
        self.model.add(Dense(64, activation='selu'))

        # add the fourth hidden layer with 128 neurons, relu activation
        self.model.add(Dense(128, activation='selu'))

        # add the fifth hidden layer with 64 neurons, relu activation
        self.model.add(Dense(64, activation='selu'))

        # add the sixth hidden layer with 64 neurons, relu activation
        self.model.add(Dense(64, activation='selu'))

        # add the seventh hidden layer with 32 neurons, relu activation
        self.model.add(Dense(32, activation='selu'))

        # add the single output layer
        self.model.add(Dense(1, activation='selu'))

        # use the rmsprop optimizer
        self.rms = keras.optimizers.RMSprop(lr=0.001)

        # compile the model
        self.model.compile(optimizer=self.rms, loss='mean_squared_error', metrics =['mean_absolute_error'])

    # function for training the neural network model
    def train(self, epochs, batch_size, X_train, X_test, Y_train, Y_test, stopping=True):
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
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size)
            else:
                # prepare for early stopping
                early_stopping = keras.callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0,
                                                         patience=40, verbose=0, mode='auto',
                                                         baseline=None, restore_best_weights=False)
                # fit the model
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping])

            # return the model
            return self.model
        except:
            return keras.models.Model()
