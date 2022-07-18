#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:56:33 2022

@author: simonlesflex
"""

#from datetime import datetime, timedelta
import yahoo_fin.stock_info as si
import pandas as pd


import requests
import re
from bs4 import BeautifulSoup

from w3lib.html import replace_entities
import pandas as pd



def getREITs():
    gheaders = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    params = 'offset=0&count=100'
    page = requests.get('https://finance.yahoo.com/screener/predefined/REIT_Diversified', params = params, headers = gheaders)
    soup = BeautifulSoup(page.content, 'html.parser') # Parsing content using beautifulsoup
    
    
    rows = soup.find_all('a', {'data-test': re.compile('quoteLink*')})
    
    REIT_Tickers = [r.text.strip() for r in rows]
    
    print(REIT_Tickers)
    
    return REIT_Tickers


