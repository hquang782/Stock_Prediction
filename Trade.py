from cProfile import label
from turtle import color
import  matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import xlwings as xw
wb = xw.Book()
ws = wb.sheets[0]
data = yf.download(tickers="UBER", period='30d',interval='1d')
ws['A1'].options(pd.DataFrame, header=1,index=True,expand='table').value= data
x=[]
for i in range(30):
    x.append(i)
h=data.__getitem__('High')
l=data.__getitem__('Low')
fig, ax = plt.subplots()
ax.plot(x,h,label='high',color='red')
ax.plot(x,l,label='low')
ax.set(title='BTL PY', xlabel='date',ylabel='giá trị')
plt.show()
