# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import os
import sys
import json
import dimcli
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
from dimcli.shortcuts import *
from collections import Counter, defaultdict, ChainMap

# function for initializing the dimensions cli
def init_dimcli(credentials_file):
    """
    Login to the Dimensions API and
    return the DSL object
    Parameters
    ----------
    arg1 | credentials_file: str
        The file name that contains dimensions credentials
    Returns
    -------
    Dimensions Object
        dimcli.core.api.Dsl
    """
    try:
        # use the credentials file name to obtain the account information
        with open(credentials_file) as f:
            # load the credentials from the file
            credentials = json.load(f)

            # fetch the username, pwd and URI from the credentials file
            username = credentials['username']
            password = credentials['password']
            endpoint = credentials['endpoint']

            # login to dimensions API
            dimcli.login(username, password, endpoint)

            # craete a DSL object
            dsl = dimcli.Dsl()
        
        # return the DSL object
        return dsl
    except:
        # return an empty list
        return dimcli.core.api.Dsl

# function for obtaining the altmetric id from the dump
def parse_altmetric_id(file_object, paper_id):
    """
    Parse the altmetric_id from the file
    object using hte paper id and return
    the id after processing the object
    Parameters
    ----------
    arg1 | file_object: list
        The list that contains the altmetric information of all articles
    arg2 | paper_id: int
        The paper id that is useful for extracting all of the infromation 
        about scholarly articles from the file object
    Returns
    -------
    Number
        int
    """
    try:
        # iterate over the file object for a specific paper
        paper_info = file_object[paper_id].split(',')

        # iterate over the list to parse the altmetric id
        altmetric_id = [info for info in paper_info if 'altmetric_id' in info][0].strip().split(':')[1].strip()

        # return the altmetric id
        return int(altmetric_id)
    except:
        # return a zero
        return int(0)

# function for splitting a list into n parts
def chunk_list(sequence, parts):
    """
    Create smaller sequences of a list
    that are nearly equal in length
    Parameters
    ----------
    arg1 | sequence: list
        A list of items
    arg2 | parts: int
        Number of parts to split the list
    Returns
    -------
    Array
        list
    """
    try:
        # calculate the average number of parts
        avg = len(sequence) / float(parts)
        sequences = []
        last = 0.0

        while last < len(sequence):
            sequences.append(sequence[int(last):int(last + avg)])
            last += avg

        # return the new sequence of smaller sequences
        return sequences
    except:
        # return an empty list
        return list()
    
# function for obtaining the citations using the dimensions API
def get_dimensions_citations(dsl_object, dataframe):
    """
    Use the dimensions API to obtain the citations
    for a given altmetric id associated with a scholarly article
    Parameters
    ----------
    arg1 | dsl_object: dimcli.core.api.Dsl
        The DSL dimensions object used for making CRUD operations
    arg2 | dataframe: pandas.core.frame.DataFrame
        The dataframe that has information of all scholarly articles
    Returns
    -------
    Array
        numpy.ndarray
    """
    try:
        # store the altmetric ids to the list
        seed = dataframe.altmetric_id.values.tolist()

        # chunk the list into smaller lists due to API payload restrictions
        seed = chunk_list(seed, 26149)

        # init a list for storing the citations
        citations = []

        # iterate over the batches
        for i in range(len(seed)):
            # use the dimcli object to query
            data = dsl_object.query(f"""search publications where altmetric_id in {json.dumps(seed[i])} return publications[times_cited]""")

            # return just the citations
            citations.append(data.as_dataframe()['times_cited'].values.tolist())

        # return the altmetric id
        return citations
    except:
        # return a zero
        return numpy.zeros(len(dataframe))    
    
# function for extracting the altmetric id's from the dump
def db_dump_process(dump_fname, data_fname, dsl_object):
    """
    Read the db dump containing information about
    altmetric articles and return neccesary columns
    into a dataframe
    Parameters
    ----------
    arg1 | dump_fname: str
        The file name that contains the altmetrics 2016 dump
    arg2 | data_fname: str
        The file name that contains all of the infromation about scholarly articles
    arg3 | dsl_object: dimcli.core.api.Dsl
        The DSL dimensions object used for making CRUD operations
    Returns
    -------
    Dataframe
        pandas.core.frame.DataFrame
    """
    try:
        # read the 2016 dump file
        file_dump = open(dump_fname).readlines()

        # read the dataframe
        data = pd.read_csv(data_fname, low_memory=False)

        # identify the required altmetric ids from the list
        paper_ids = data.paperid.values.tolist()

        # parse the altmetric ids from the dump
        list_of_altmetric_ids = [parse_altmetric_id(file_dump, paper_id) for paper_id in tqdm(paper_ids)]

        # add the altmetric id from the dump to the dataframe
        data = data.assign(altmetric_id = list_of_altmetric_ids)

        # parse the citations with altmetric id
        citations = get_dimensions_citations(dsl_object, data)

        # add the citations column to the dataframe
        #data = data.assign(citations_dimensions = citations)
        citations = list(itertools.chain.from_iterable(citations))

        # print the number of scholarly articles that have dimensions citations
        print("Total articles with citations: ", len(citations))

        # return the dataframe
        return data
    except:
        # return an empty list
        return pd.DataFrame()

if __name__ == '__main__':
    # run the init function
    dsl = init_dimcli('cred.json')

    #  run the db dump process function
    data = db_dump_process('/media/hector/data/datasets/db2.csv', '/media/hector/data/datasets/sch_impact.csv', dsl)

    # extract the citations
    #print(data[['altmetric_id', 'citations', 'citations_dimensions']].head())

    
