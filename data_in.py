# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:00:21 2020

@author: lukeb
"""
import pickle

with open('data.pickle', 'rb') as f:
    data_in = pickle.load(f)
    