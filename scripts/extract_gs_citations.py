#!/usr/bin/env python3

# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE.

import os
import csv
import glob
import json
import requests
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from fp.fp import FreeProxy
from torrequest import TorRequest
from scholarly import scholarly
from collections import Counter, OrderedDict
from operator import attrgetter

# class definition for Rate limiting
class RateLimiter:
    """
    Class object for putting a rate limit on the
    number of requests made
    Parameters
    ----------
    No arguments
    Returns
    -------
    Nothing
    """
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
    while True:
        try:
            # call the lumproxy object
            scholarly.use_lum_proxy()

            # make the query
            query = scholarly.search_pubs(title)

            # come out
            break
        except Exception as e:
            # come out and try again
            break

    # return the response dict
    return next(query)

# function for assigning new IP address
def assign_new_ip(text=False):
    """
    Reset the identity using TorRequest

    Parameters
    ----------
    arg1 [OPTIONAL]| text: bool
        A boolean flag to return the IP address tuple (old, morphed)

    Returns
    -------
    boolean
        True/False

    """
    try:
        # pass the hashed password
        req = TorRequest(password='scholarly_password')

        # return the ip address
        normal_identity = requests.get('http://ipecho.net/plain')

        # reset the identity using Tor
        req.reset_identity()

        # make a request now
        morphed_identity = req.get('http://ipecho.net/plain')

        # return the status depending on the flag
        if morphed_identity != normal_identity:
            if text == True:
                # return the ip address pairs as a tuple
                return (normal_identity.text, morphed_identity.text)
            else:
                return True
        else:
            # return just the status
            return False
    except:
        return False

# function for assigning a new proxy
def set_new_proxy(text=True):
    """
    Reset the identity using FreeProxy
    Parameters
    ----------
    arg1 [OPTIONAL]| text: bool
        A boolean flag to return the IP address tuple (old, morphed)
    Returns
    -------
    Address
        fp.fp.FreeProxy
    """
    while True:
        # call the freeproxy object
        proxy = FreeProxy(rand=True, timeout=1).get()

        # allocate the proxy address to scholarly
        proxy_works = scholarly.use_proxy(http=proxy, https=proxy)

        # check it the ip address works
        if proxy_works:
            # come out
            break

    # print the ip address depending on the text argument
    if text:
        # print the working ip
        print("Working proxy:", proxy)

    # return the proxy details
    return proxy

# function for connecting tor to scholarly
def scholarly_init_connection():
    """
    Bind TorRequest to Scholarly service

    Parameters
    ----------
    No arguments
    Returns
    -------
    Nothing
    """
    while True:
        # assign new tor identity
        ips = assign_new_ip(text=True)

        # use the tor request for scholarly
        tor_req = scholarly.use_tor(tor_sock_port=9050, \
                                    tor_control_port=9051, \
                                    tor_pw="scholarly_password")
        if tor_req:
            # come out of the loop, when successful
            break

    # print the tor identity
    print("Working Tor identity:", ips[1])

# function for restarting the system tor service
def restart_tor_system_service(text=False):
    """
    Use the os module to restart the tor service
    Parameters
    ----------
    arg1 [OPTIONAL]| text: bool
        A boolean flag to return the status of the command

    Returns
    -------
    Boolean
        bool
    """
    # subprocess command for stopping the tor service
    tor_stop = subprocess.Popen(['service', 'tor', 'stop'])

    # subprocess command for restarting the tor service
    tor_restart = subprocess.Popen(['service', 'tor', 'restart'])

    # subprocess command for restarting the tor service
    tor_status = subprocess.Popen(['service', 'tor', 'status'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=True)

    # if the label is set to true then print the output
    if text:
        for output in tor_status.stdout.readlines():
            print(output.strip())

    # pipe out the stdout, stderr for the subprocess
    stdout, stderr = tor_status.communicate()

    if len(stderr) > 0:
        # return False
        return False
    else:
        # return true if successful
        return True

def get_articleInfo(title):
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
    while True:
        try:
            # init the connection with scholarly and tor
            scholarly_init_connection()

            # search for the query
            search_query = scholarly.search_pubs(title)

            # print success
            print("Got the results of the query")

            # come out of the loop
            break
        except Exception as e:
            # print error message
            print("Attempt Failed, patching new tor identity")

            # restart the system tor service
            restart_tor_system_service(text=False)

            # assign new connection again
            scholarly_init_connection()

    # obtain the bib entry of the scholarly article
    pub = next(search_query)

    # return the bib entry
    return pub

if __name__ == '__main__':
    # iterate over the length
    length_of_file = len(open('paper_titles.txt').readlines())

    # place the contents of the list into a file
    alt_list = open('paper_titles.txt').readlines()

    # iterate over the length of the file
    # write the results to a file
    for i in tqdm(range(length_of_file)):
        alt_info = open('paper_titles.txt', 'r+')
        cit_info = open('citations_gs.csv', 'a')
        cit_info.write(str(alt_list[i].strip(
        ).split('\t')[0]) + ',' + str(get_articleInfo(alt_list[i].strip().split('\t')[1])))
        cit_info.write('\n')
        cit_info.close()
        alt_info.seek(0)
        alt_info.truncate()
        alt_info.writelines(alt_list[i+1:])
        alt_info.close()
