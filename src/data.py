# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# function for computing sigmoid of a value
def sigmoid(value, derivative=False):
    """
    Return the sigmoid of a numeric value
    Parameters
    ----------
    arg1 | value: int
        The numeric value intended to convert into a continuos range
    Returns
    -------
    Float
        float
    """
    try:
        # compute the sigmoid
        result = 1. / (1. + np.exp(-x))

        # check if derivative is required
        if derivative:
            # return the sigmoid
            return result * (1. - result)

        # return the sigmoid
        return result
    except:
        # return zero
        return np.zeros(1)[0]

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

        # create a target variable for the second experiment
        data = data.assign(target_exp_2 =
                           list(map(lambda x: 1 if x > 9 else 0, tqdm(data['citations']))))

        # create a target variable for the third experiment
        data = data.assign(target_exp_3 =
                           list(map(lambda x: np.log(1 + x), tqdm(data['citations']))))

        # create a target variable for the third experiment
        data = data.assign(target_exp_4 = list(map(sigmoid, tqdm(data['citations']))))

        # drop the columns unecessary
        data = data.drop(columns=['Type', 'citations', 'citations(Log_Transformed)'])

        # return the dataframe
        return data
    except:
        return pd.DataFrame()

# function for preparing the X & Y for the dataset
def prepare_X_Y(data_frame, target):
    """
    Process the dataframe and return the X and Y for the experiment

    Parameters
    ----------
    arg1 | data_frame: pandas.DataFrame
        A loaded dataframe for preparing X and Y
    arg1 | target: str
        The intended target variable for the experiment

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
        Y = data_frame[target]

        # return the tuple
        return X, Y
    except:
        return np.zeros((len(data_frame), 22)), np.zeros((len(data_frame), 22))
