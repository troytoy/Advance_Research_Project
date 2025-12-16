#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


start = '2010-01-01'
end = '2024-12-31'

data = yf.download('EURUSD=X', start=start, end=end)


# In[3]:


data.head()

