#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import os
import sys
import csv
import json
import copy
import time
import dimcli
import pickle
import urllib3
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, deque

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


# function for authentication using the dimensions web url
def dimensions_login(credentials_file):
    """
    Use the dimensions web URL and requests API to login
    to the dimensions API
    Parameters
    ----------
    arg1 | credentials_file: str
        The file name that contains dimensions credentials
    Returns
    -------
    Dictionary
        dict
    """
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

        # return the headers
        return headers

# function for obtaining the citations using the dimensions web url


def get_dimensions_citations_web(headers, altmetric_id):
    """
    Use the dimensions web URL and requests API to obtain the citations
    for a given altmetric id associated with a scholarly article
    Parameters
    ----------
    arg1 | headers: str
        The login headers from dimensions to query
    arg2 | altmetric_id: int
        The altmetric id of a scholarly article
    Returns
    -------
    Dictionary
        dict
    """
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

        if 'authors' in response.keys():
            # set the first name
            response['first_name'] = response['authors'][0]['first_name'] + \
                ' ' + response['authors'][0]['last_name']

            # remove the authors key
            del response['authors']

        # return the final dict
        return response
    else:
        # return the json
        return dict()

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
