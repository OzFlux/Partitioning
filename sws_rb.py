#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:58:27 2018

@author: ian
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def soil_moisture_func(sws_series, theta_1, theta_2):
    return 1 / (1 + np.exp(theta_1 - theta_2 * sws_series))

def group_soil_moisture(df, n_cats = 30):
    
    df['theta_cats'] = pd.qcut(df.Sws, n_cats, 
                               labels = np.linspace(1, n_cats, n_cats))
    return df.groupby('theta_cats').mean()

def fit_soil_moisture(df):
    
    params, pcov = curve_fit(soil_moisture_func, df.Sws, df.rb, p0 = [3, 30])
    
    return params