#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[3]:


tweets = pd.DataFrame.from_csv('/Users/mohammedawan/Downloads/School/2018-2 Fall/Info Retrieval:Knowledge Discovery [CSI 5810]/Project 2/output.csv')


# In[4]:


x = tweets.loc[:, :'entities'].values
y = tweets['engagement'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[52]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth = 5)
dtr.fit(x_train, y_train)


# In[53]:


dpredictionr2 = r2_score(y_test, dtr.predict(x_test), multioutput='raw_values')
print("DTR R2 Score: " + str(dpredictionr2))
print("Exmaple Y: " + str(y_test[0:4]))
print("Prediction: " + str(dtr.predict(x_test[0:4]).astype(int)))


# In[7]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)


# In[32]:


rpredictionr2 = r2_score(y_test, rfr.predict(x_test), multioutput='raw_values')
print("RFR R2 Score: " + str(rpredictionr2))
print("Exmaple Y: " + str(y_test[0:4]))
print("Prediction: " + str(rfr.predict(x_test[0:4]).astype(int)))


# In[9]:


avgd = np.array(dtr.feature_importances_)
avgdt = avgd[0:401]
avgdl = avgd[401:441]
avgdt = np.average(avgdt)
avgdl = np.average(avgdl)
avgd = avgd[441:]
avgd = np.insert(avgd,0,avgdl)
avgd = np.insert(avgd,0,avgdt)
print(avgd)


# In[11]:


avgd2 = np.array(rfr.feature_importances_)
avgdt2 = avgd2[0:401]
avgdl2 = avgd2[401:441]
avgdt2 = np.average(avgdt2)
avgdl2 = np.average(avgdl2)
avgd2 = avgd2[441:]
avgd2 = np.insert(avgd2,0,avgdl2)
avgd2 = np.insert(avgd2,0,avgdt2)
print(avgd2)


# In[12]:


avgdtotal = []
for i in range(len(avgd)):
    avgdtotal.append(np.average([avgd[i],avgd2[i]]))
print(avgdtotal)


# In[13]:


labels = ['Text', 'Lang', 'Verified', 'Length', ]
avgdtotal2 = []
for a in avgdtotal:
    if(a <=0.05):
        avgdtotal2.append(a)
y_pos = np.arange(len(labels))
for i in range(len(labels)):
        plt.bar(i,avgdtotal2[i], align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.ylabel('Weights')
plt.title('Feature Importance') 
plt.show()


# In[56]:


plt.hist([w for w in y if w < 10000], bins = 50)
plt.ylabel('# of tweets')
plt.xlabel('# of retweets')
plt.title('Tweet distribution') 
plt.show()


# In[ ]:




