from cProfile import label
from this import d
from turtle import color
import  matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import xlwings as xw
wb = xw.Book()
ws = wb.sheets[0]
data = yf.download(tickers="UBER", period='30d',interval='1d')
print(data)
ws['A1'].options(pd.DataFrame, header=1,index=True,expand='table').value= data
x=[]
for i in range(30):
    x.append(i)
h=data.__getitem__('High')
l=data.__getitem__('Low')
m=(h+l)/2
fig, ax = plt.subplots()
ax.plot(x,m,label='TB',color='yellow')
ax.plot(x,h,label='high',color='blue')
ax.plot(x,l,label='low',color='red')
ax.set(title='BTL PY', xlabel='date',ylabel='giá trị')
plt.legend()
plt.show()