# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:49:44 2020

@author: lukeb
"""

import autotrader_scraper as sc
import pandas as pd
import sys

"""
df = sc.scraper('https://www.autotrader.co.uk/classified/advert/new/202007010715098?journey=deals&fromHomepageNewCars=true')
search_url = 'https://www.autotrader.co.uk/car-search?postcode=BS50DN&radius=10&make=BMW&year-to=2020&onesearchad=Used&onesearchad=Nearly%20New&onesearchad=New&advertising-location=at_cars&page=1'
search = sc.search_result_scraper(search_url)
tree = sc.tree_getter(search_url)
c_scraper = sc.c_scraper(search_url)
"""
list_urls = pd.read_csv('petrol_2018.csv', header=None)

lst = list(list_urls[0])

frames = []
for i in range(len(lst)):
    entry = sc.scraper(lst[i])
    frame = pd.DataFrame(entry, index=[i])
    frames.append(frame)
    sys.stdout.write('.'); sys.stdout.flush();  # print a small progress bar
df = pd.concat(frames)

