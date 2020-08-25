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
from scholarly import scholarly
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


# function for obtaining the citations using the dimensions web url


def get_gs_citations_web(title):
    """
    Use the google scholar web URL and requests API to obtain the citations
    for a given title of a scholarly article
    Parameters
    ----------
    arg1 | title: str
        The title of a scholarly article
    Returns
    -------
    Dictionary
        dict
    """
    # make the query
    query = scholarly.search_single_pub(title)

    # return the response dict
    return query

if __name__ == '__main__':
    # call the rate limiter object
    r = RateLimiter()

    # iterate over the length
    length_of_file = len(open('paper_titles.txt').readlines())

    # place the contents of the list into a file
    alt_list = open('paper_titles.txt').readlines()

    # iterate over the length of the file
    # write the results to a file
    for i in tqdm(range(length_of_file)):
        time.sleep(0.25)
        if not r():
            alt_info = open('paper_titles.txt', 'r+')
            cit_info = open('citations_gs.csv', 'a')
            cit_info.write(str(alt_list[i].strip(
            ).split('\t')[0]) + ',' + str(get_gs_citations_web(alt_list[i].strip().split('\t')[1])))
            cit_info.write('\n')
            cit_info.close()
            alt_info.seek(0)
            alt_info.truncate()
            alt_info.writelines(alt_list[i+1:])
            alt_info.close()
        else:
            pass


