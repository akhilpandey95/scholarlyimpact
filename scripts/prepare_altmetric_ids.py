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
from datetime import datetime
from collections import Counter, defaultdict, deque

# function for using the altmetric api to obtain the pubdates
# add published date of the article


def add_pubdate(alt_id):
    try:
        response = requests.get("https://api.altmetric.com/v1/id/" +
                                str(alt_id) + '?key=0674c34a5f9563d51ac1a7443d448ec3')
        return dict(json.loads(response.content))['published_on']
    except:
        return ''

# function for returning year from an epoch


def get_year_from_epoch(epoch):
    """
    Obtain the year from a given epoch string
    Parameters
    ----------
    arg1 | epoch: str
        Epoch string
    Returns
    -------
    Number
        int
    """
    return datetime.fromtimestamp(epoch).year

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
        altmetric_id = [info for info in paper_info
                        if 'altmetric_id' in info][0].strip().split(':')[1].strip()

        # return the altmetric id
        return int(altmetric_id)
    except:
        # return a zero
        return int(0)

# function for obtaining the published date from the dump


def parse_published_date(file_object, paper_id):
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

        # iterate over the list to parse the published date1
        paper_pubdate = [info for info in paper_info
                         if 'published_on' in info][0].strip().split(':')[1].strip()

        # return the pub date in epochs
        return literal_eval(paper_pubdate)
    except:
        # return a zero
        return int(0)


# read the 2016 dump file
#file_dump = open('db2.csv').readlines()

# read the dataframe
data = pd.read_csv('sch_impact_beta.csv', low_memory=False)

# remove unecessary columns
data = data.drop(columns=['Unnamed: 0'])

# identify the required altmetric ids from the list
#paper_ids = data.paperid.values.tolist()

# parse the altmetric ids from the dump
#list_of_altmetric_ids = [parse_altmetric_id(file_dump, paper_id)
#                         for paper_id in tqdm(paper_ids)]

# parse the published date from the dump
# list_of_pub_dates = [add_pubdate(paper_id)
# for paper_id in tqdm(paper_ids)]

# add the altmetric id from the dump to the dataframe
# data = data.assign(altmetric_id=list_of_altmetric_ids)

# add the published date from the dump to the dataframe
# data = data.assign(pubdate=list_of_pub_dates)

# add the published year
# data = data.assign(pubyear=list(map(get_year_from_epoch, tqdm(data.pubdate))))

# only include the 2016 articles
data = data.loc[data['pubdate'].apply(lambda x: int(x)) == 2016].reset_index(drop=True)

# grab the altmetric ids
ids = data.altmetric_id.values.tolist()

# write the ids to a text file
if len(sys.argv) > 1:
    end_seq = int(sys.argv[1])
else:
    end_seq = len(ids)

# altmetric_id for all the 130K articles
for i in tqdm(ids[: end_seq]):
    t = open('altmetric_ids.txt', 'a')
    t.write(str(i))
    t.write('\n')
    t.close()
