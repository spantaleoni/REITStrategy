#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:35:55 2022
https://builtin.com/data-science/portfolio-optimization-python
https://github.com/dcajasn/Riskfolio-Lib
https://riskfolio-lib.readthedocs.io/en/latest/plot.html?highlight=plot
@author: simonlesflex -- RATES COUNTRIES
            Switzerland
            Denmark
            Japan
            Sweden
            Europe
            Australia
            Israel
            United States
            Canada
            Great Britain
            Norway
            New Zealand
            Saudi Arabia
            South Korea
            Poland
            Hungary
            Czech Republic
            South Africa
            China
            India
            Chile
            Mexico
            Indonesia
            Russia
            Brazil
            Turkey
"""

import numpy as np
import pandas as pd
from yahoo_fin.stock_info import *
import warnings

import riskfolio as rp
# importing matplotlib library
import matplotlib.pyplot as plt

import telegram_send


warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

TelegramFLAG = True
LOOKBACK = 252
NUMYEARS = 10
G_FONTSIZE = 18
G_PORTFOLIOTITLE = '** Portfolio Optimization **'

# Tickers of assets
assets = ['RSP', 'AAPL', 'TLT', 'IEF', 'TIP']
assets.sort()


def GetGlobalRates(country = 'United States'):
            
    try:
        globalRates = pd.read_csv('/home/simonlesflex/PythonProjects/GetInterestRates/global_interest_rates.csv')
        #print(globalRates)
    except:
        print("!!ERROR LOADING GLOBAL RATES from CSV!!")
    
    #print(globalRates[globalRates.Name == 'United States'])
    if country == "all":
        Rate = globalRates
    else:
        Rate = float(globalRates[globalRates.Name == country].Value)
    
    print(Rate)
    return Rate
    

def PortfolioOptimizeMV(assets, RiskModel='MAD'):
    
    df_data = {}
    for stock_ticker in assets: 
        data = get_data(stock_ticker)
        adj_closed = data['close'][- (LOOKBACK * NUMYEARS):]
        df_data[stock_ticker] = adj_closed
    data = pd.DataFrame(df_data).dropna()
    # Calculating returns
  
    Y = data[assets].pct_change().dropna()
    display(Y.head())
    
    # Building the portfolio object
    port = rp.Portfolio(returns=Y)
    
    # Calculating optimal portfolio
    # Select method and estimate input parameters:
    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='hist' # Method to estimate covariance matrix based on historical data.
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    
    # Estimate optimal portfolio:
    model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    #rm = 'MAD' # Risk measure used, this time will be variance
    rm = RiskModel
    #obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True # Use historical scenarios for risk measures that depend on scenarios
    rf = 0 # Risk free rate
    l = 0 # Risk aversion factor, only useful when obj is 'Utility'
    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    display(w.T)
    
    points = 50 # Number of points of the frontier

    frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
    
    # Plotting the efficient frontier

    label = 'Max Risk Adjusted Return Portfolio' # Title of point
    mu = port.mu # Expected returns
    cov = port.cov # Covariance matrix
    returns = port.returns # Returns of the assets
    
    ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)
    plt.savefig('EfficientFrontier.jpeg')
    
    #ax2 = rp.plot_table(returns=returns, w=w, MAR=0, alpha=0.05, ax=None)
    
    
    return w.T
    
    # Plotting the composition of the portfolio
    
    #ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                     #height=6, width=10, ax=None)


def PortfolioOptimizeALL(assets, RiskModelT=['MAD', 'FLPM', 'CDaR', 'UCI']):

    # Risk Measures available:
    #
    # 'MV': Standard Deviation.
    # 'MAD': Mean Absolute Deviation.
    # 'MSV': Semi Standard Deviation.
    # 'FLPM': First Lower Partial Moment (Omega Ratio).
    # 'SLPM': Second Lower Partial Moment (Sortino Ratio).
    # 'CVaR': Conditional Value at Risk.
    # 'EVaR': Entropic Value at Risk.
    # 'WR': Worst Realization (Minimax)
    # 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
    # 'ADD': Average Drawdown of uncompounded cumulative returns.
    # 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
    # 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
    # 'UCI': Ulcer Index of uncompounded cumulative returns.
    
    df_data = {}
    for stock_ticker in assets: 
        data = get_data(stock_ticker)
        adj_closed = data['close'][- (LOOKBACK * NUMYEARS):]
        df_data[stock_ticker] = adj_closed
    data = pd.DataFrame(df_data).dropna()
    # Calculating returns
  
    Y = data[assets].pct_change().dropna()
    display(Y.head())
    
    # Building the portfolio object
    port = rp.Portfolio(returns=Y)
    
    # Calculating optimal portfolio
    # Select method and estimate input parameters:
    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='hist' # Method to estimate covariance matrix based on historical data.
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    
    # Estimate optimal portfolio:
    model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = 'MV' # Risk measure used, this time will be variance
    obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True # Use historical scenarios for risk measures that depend on scenarios
    rf = 0 # Risk free rate
    l = 0 # Risk aversion factor, only useful when obj is 'Utility'    
    
    rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
           'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']
    
    w_s = pd.DataFrame([])
    
    
    for i in rms:
        w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
        w_s = pd.concat([w_s, w], axis=1)
        w.plot(kind='pie', y='weights', shadow=True, autopct='%1.1f%%', title=str(i))
        if TelegramFLAG is True:
            filename = 'PortfolioOptimization' + str(i) + '.jpeg'
            plt.savefig(filename)
            with open(filename, "rb") as fmarket:
                telegram_send.send(images=[fmarket])
    
    #w = port.optimization(model=model, rm=RiskModel, obj=obj, rf=rf, l=l, hist=hist)
    #w_s = pd.concat([w_s, w], axis=1)
        
    w_s.columns = RiskModelT
    
    display(w_s)
    return w_s
    #if TelegramFLAG is True:
        #w_s.to_csv("w_s.txt", header=True, index=True, sep=',', mode='w')
        #with open("w_s.txt", "rb") as fmarket:
            #telegram_send.send(files=[fmarket])
