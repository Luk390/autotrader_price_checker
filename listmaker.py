# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:53:45 2020

@author: lukeb
"""

# Listmaker
# Usage:
# ./listmaker.py https://www.autotrader.co.uk/car-search?advert... url_list.txt 
# if file exists, we append

import sys
import csv
import os

import autotrader_scraper as ff1

postcode_urls = ['BS11AA', 'WC2N5DU', 'B11BB', 'LS11AZ', 'M11AG']

for postcode in postcode_urls:
    base_url = str('https://www.autotrader.co.uk/car-search?postcode='+str(postcode)+'&radius=10&year-from=2018&year-to=2020&fuel-type=Petrol')
    out_fname = 'petrol_2018.csv'
    
    if not os.path.isfile(out_fname):
        z = open(out_fname, 'w')
        z.close()
        
    page_count = 1 
    while True:
        results = ff1.search_result_scraper(base_url + '&page=' + str(page_count))
        if len(results) > 0:
            with open(out_fname, 'a') as fcon:
                for r in results:
                    fcon.write('https://www.autotrader.co.uk' + r)
                    fcon.write('\n')
            page_count += 1
            print(f"Page Count: {page_count}, Postcode: {postcode}")
            print(len(results))
        else:
            break