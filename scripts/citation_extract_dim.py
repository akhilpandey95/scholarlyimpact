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
from ast import literal_eval
from functools import reduce
from dimcli.shortcuts import *
from ratelimit import limits, sleep_and_retry
from collections import Counter, defaultdict, ChainMap

# class definition for Rate limiting


class RateLimiter:
    def __init__(self, maxRate=5, timeUnit=1):
        self.timeUnit = timeUnit
        self.deque = deque(maxlen=maxRate)

    def __call__(self):
        if self.deque.maxlen == len(self.deque):
            cTime = time.time()
            if cTime - self.deque[0] > self.timeUnit:
                self.deque.append(cTime)
                return False
            else:
                return True
        self.deque.append(time.time())
        return False

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
        altmetric_id = [info for info in paper_info if 'altmetric_id' in info][0].strip().split(':')[
                                                                                        1].strip()

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


def get_dimensions_citations(dsl_object, altmetric_id):
    """
    Use the dimensions API to obtain the citations
    for a given altmetric id associated with a scholarly article
    Parameters
    ----------
    arg1 | dsl_object: dimcli.core.api.Dsl
        The DSL dimensions object used for making CRUD operations
    arg2 | dataframe: int
        The altmetric id of the scholarly article
    Returns
    -------
    Number
        int
    """
    try:
        # store the altmetric ids to the list
        seed = [altmetric_id]

        # use the dimcli object to query
        data = dsl_object.query(
            f"""search publications where altmetric_id in {json.dumps(seed)} return publications[times_cited]""")

        # return just the citations
        return data.as_dataframe()['times_cited'][0]
    except:
        # return a zero
        return int(0)

# function for obtaining the citations using the dimensions web url
@sleep_and_retry
@limits(calls=30, period=60)
def get_dimensions_citations_web(credentials_file, altmetric_id):
    """
    Use the dimensions web URL and requests API to obtain the citations
    for a given altmetric id associated with a scholarly article
    Parameters
    ----------
    arg1 | credentials_file: str
        The file name that contains dimensions credentials
    arg2 | altmetric_id: int
        The altmetric id of a scholarly article
    Returns
    -------
    Dictionary
        dict
    """
    try:
        # use the credentials file name to obtain the account information
        with open(credentials_file) as f:
            # load the credentials from the file
            credentials = json.load(f)

            # fetch the username, pwd and URI from the credentials file
            login = {
                'username': credentials['username'],
                'password': credentials['password']
            }

            # login to the web API
            resp = requests.post(
                'https://app.dimensions.ai/api/auth.json', json=login)

            # make sure alerts are raised in case of failed authentication
            resp.raise_for_status()

            # Create http header using the generated token.
            headers = {
                'Authorization': "JWT " + resp.json()['token']
            }

            # create the query
            query = """search publications where altmetric_id in""" + \
                str([altmetric_id]) + \
                    """return publications[id+doi+altmetric_id+title+times_cited+authors]"""

            # Execute DSL query.
            resp = requests.post(
                'https://app.dimensions.ai/api/dsl.json',
                data=query.encode(),
                headers=headers)

            # check for 200 status
            if resp.status_code == 200 and 'publications' in resp.json() \
               and len(resp.json()['publications']) > 0:
                # just obtain the first author
                response = copy.deepcopy(resp.json()['publications'][0])

                # set the first name
                response['first_name'] = response['authors'][0]['first_name'] \
                                        + ' ' + \
                                            response['authors'][0]['last_name']

                # remove the authors key
                del response['authors']

                # return the final dict
                return response
            else:
                # return the json
                return resp.json()
    except:
        # return a empty dict
        return dict()

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
        list_of_altmetric_ids = [parse_altmetric_id(
            file_dump, paper_id) for paper_id in tqdm(paper_ids)]

        # add the altmetric id from the dump to the dataframe
        data = data.assign(altmetric_id=list_of_altmetric_ids)

        # return the dataframe
        return data
    except:
        # return an empty list
        return pd.DataFrame()

if __name__ == '__main__':
    # get the login headers for dimensions web API
    headers = dimensions_login('cred.json')

    # call the rate limiter object
    r = RateLimiter()

    # iterate over the length
    length_of_file = len(open('altmetric_ids.txt').readlines())

    # place the contents of the list into a file
    alt_list = open('altmetric_ids.txt').readlines()

    # iterate over the length of the file
    # write the results to a file
    for i in tqdm(range(length_of_file)):
        time.sleep(1.75)
        if not r():
            alt_info = open('altmetric_ids.txt', 'r+')
            cit_info = open('citations_dim.csv', 'a')
            cit_info.write(str(alt_list[i].strip(
            )) + ',' + str(get_dimensions_citations_web(headers, int(alt_list[i].strip()))))
            cit_info.write('\n')
            cit_info.close()
            alt_info.seek(0)
            alt_info.truncate()
            alt_info.writelines(alt_list[i+1:])
            alt_info.close()
        else:
            pass

    # print the remaining number of files
    print(length_of_file)
    
