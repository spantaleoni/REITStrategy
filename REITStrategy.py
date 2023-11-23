#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 15:56:33 2022

@author: simonlesflex
"""

#from datetime import datetime, timedelta
from yahoo_fin.stock_info import *
import yahoo_fin
import pandas as pd
from getREIT import *
import matplotlib.pyplot as plt
import telegram_send

from PortfolioOptimize import *

TelegramSend = True

G_REPORTTITLE = '** DUAL MOMENTUM in REITS **'

STARTDATE = '01/01/2017'
ENDDATE = '01/02/2022'
LOOKBACK = 252
NUMYEARS = 2
VolumeAvgDays = 21
VolumeFilter = 10000
NumStockFilter = 5
slong_mom = 365
long_mom = 60
skip_mom = 30
G_RISKMODEL = 'MAD'
G_FONTSIZE = 16

REIT_Tickers = getREITs()


df_data = pd.DataFrame()
for stock_ticker in REIT_Tickers:
    try: 
        if ENDDATE == '':
            data = get_data(stock_ticker, start_date=STARTDATE)
        else:
            data = get_data(stock_ticker, start_date=STARTDATE, end_date=ENDDATE)
    except:
        pass
    #adj_closed = data['close'][- (LOOKBACK * NUMYEARS):]
    adj_closed = data['close']
    print(len(adj_closed))
    volume = ( data['volume'][- (VolumeAvgDays):].sum() ) / VolumeAvgDays
    #tickdata = pd.concat([adj_closed, volume], axis=1)
    if volume > VolumeFilter and len(adj_closed) > (LOOKBACK * NUMYEARS):
        df_data[stock_ticker] = adj_closed
#data = pd.DataFrame([df_data], index=[0]).dropna()
data = pd.DataFrame(df_data).dropna()

print(data)

momreits = [ ]

for stockhist in data.columns:
    start_price = data[stockhist][-slong_mom]
    startm_price = data[stockhist][-long_mom]
    end_price = data[stockhist][-skip_mom]
    momentum = round(((end_price-startm_price)/startm_price) + ((end_price-start_price)/start_price), 4)
    print(momentum)
    momreits.append([stockhist, momentum])
    
dfmom = pd.DataFrame(momreits, columns=['Ticker', 'Value'])
dfmom_sort = dfmom.sort_values(by=['Value'])

TargetEQ = dfmom_sort.Ticker[-NumStockFilter:]
TargetEQ = TargetEQ.tolist()

print(TargetEQ)

TargetEQW = PortfolioOptimizeMV(TargetEQ, G_RISKMODEL)


portfoliopie = pd.DataFrame({'Tickers': TargetEQW.columns,
                             'Weight': TargetEQW.iloc[-1]}
                             )

portfoliopie.groupby(['Tickers']).sum().plot(kind='pie', fontsize=G_FONTSIZE, y='Weight', shadow=True, autopct='%1.1f%%', title=G_REPORTTITLE)
plt.savefig('PortfolioPIE.jpeg')

if TelegramFLAG is True:
    telegram_send.send(messages=[G_REPORTTITLE])
    telegram_send.send(messages=[TargetEQ])
    with open("PortfolioPIE.jpeg", "rb") as fpie:
        telegram_send.send(images=[fpie])
    with open("EfficientFrontier.jpeg", "rb") as fpie:
        telegram_send.send(images=[fpie])
