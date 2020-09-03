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

# function for using the altmetric api to obtain the title
# add published date of the article
def add_title(alt_id):
    try:
        response = requests.get("https://api.altmetric.com/v1/id/" +
                                str(alt_id) + '?key=0674c34a5f9563d51ac1a7443d448ec3')
        return dict(json.loads(response.content))['title']
    except:
        return ''


def parse_title(file_object, paper_id):
    """
    Parse the title of the paper from the file
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

        # iterate over the list to parse the title of the article
        title = [info for info in paper_info if 'title' in info][0].strip().split(':')[1].strip()

        # return the title
        return title
    except:
        # return a zero
        return int(0)

# read the dataset
data = pd.read_csv('sch_impact_delta_15_14_13.csv', low_memory=False)

# identify the required altmetric ids from the list
#paper_ids = data.altmetric_id.values.tolist()

# parse the published date from the dump
#list_of_titles = [add_title(paper_id) for paper_id in tqdm(paper_ids)]

# add the published date from the dump to the dataframe
#data = data.assign(title=list_of_titles)

# write the data to csv
#data.to_csv('sch_impact_delta_15_14_13.csv', encoding='utf-8', index=False)

# remove NA rows
data = data.loc[data['title'].apply(lambda x: type(x)) == str].reset_index(drop=True)

# only include the 2016 articles
data = data.loc[data['pubdate'].apply(lambda x: int(x)) == 2015].reset_index(drop=True)
print(data.shape)

# grab the titles of all the articles
titles = data.title.values.tolist()
ids = data.altmetric_id.values.tolist()

# write the ids to a text file
if len(sys.argv) > 1:
    end_seq = int(sys.argv[1])
else:
    end_seq = len(titles)

# altmetric_id for all the 130K articles
for i in tqdm( range(len(titles[: end_seq])) ):
    t = open('paper_titles.txt', 'a')
    t.write(str(ids[i]) + "\t" + str( ' '.join(titles[i].split()) ))
    t.write('\n')
    t.close()
