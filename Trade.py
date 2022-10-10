from email import header
from operator import index
from textwrap import indent
import requests
import xlwings as xw
import investpy as iv
import yfinance as yf
import pandas as pd
wb = xw.Book()
ws = wb.sheets["Sheet 1"]
# link="https://www.binance.com/vi/markets"
data = yf.download(tickers='UBER', period='5d',interval='5m')
ws["A1"].options(pd.DataFrame, header=1,index=True,expand='table').value=data
# data=iv.get_crypto_historical_data(crypto='bitcoin', from_date="01/10/2022", to_date='10/10/2022')