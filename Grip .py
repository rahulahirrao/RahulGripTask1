#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML 
# #TheSpark Grip Task 1
#                                                                    #By Mr. Rahul Dilip Ahirrao

# In[2]:


import pandas as pd
import numpy as np  

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[5]:


path = r"C:\Users\rahir\Downloads\student_scores.csv"
s_data = pd.read_csv(path)


# In[6]:


s_data.head()


# In[7]:


s_data


# In[8]:


s_data.columns


# In[9]:


s_data.shape


# In[11]:


train,test = train_test_split(s_data,test_size=0.25,random_state=123)


# In[12]:


train.shape


# In[13]:


test.shape


# In[14]:


train_x=train.drop("Scores",axis=1)
train_y=train["Scores"]


# In[15]:


test_x=test.drop("Scores",axis=1)
test_y=test["Scores"]


# In[16]:


lr=LinearRegression()


# In[17]:


lr.fit(train_x,train_y)


# In[18]:




lr.coef_


# In[19]:




lr.intercept_


# In[20]:


# Plotting the regression line # formula for line is y=m*x + c
line = lr.coef_*train_x+lr.intercept_

# Plotting for the test data
plt.scatter(train_x,train_y)
plt.plot(train_x, line);
plt.show()


# In[21]:




pr=lr.predict(test_x)


# In[22]:




list(zip(test_y,pr))


# In[23]:


from sklearn.metrics import mean_squared_error


# In[24]:


mean_squared_error(test_y,pr,squared=False)


# In[25]:


hour =[9.25]
own_pr=lr.predict([hour])
print("No of Hours = {}".format([hour]))
print("Predicted Score = {}".format(own_pr[0]))


# In[ ]:




