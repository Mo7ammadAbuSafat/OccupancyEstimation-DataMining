#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("Occupancy_Estimation.csv");
data.head()


# In[2]:


data.info()


# In[3]:


data.dropna();
data=data.drop_duplicates()
data.info()


# In[4]:


data.corr()


# In[5]:


import seaborn as see

see.heatmap(data.corr())


# In[6]:


import numpy as np

output=np.array(data["Room_Occupancy_Count"],dtype=int).reshape(-1,1)
input=np.array(data.iloc[:,2:18],dtype=float).reshape(-1,16)

print(input.shape);
print(output.shape);


# In[7]:


from sklearn.preprocessing import MinMaxScaler

s=MinMaxScaler();
outputnew= s.fit_transform(output);
inputnew= s.fit_transform(input);

inputnew


# In[8]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(inputnew,outputnew,train_size=0.7);

print(x_train.shape)
print(y_train.shape)


# In[9]:


from sklearn import svm

mod_svm = svm.SVR()
mod_svm.fit(x_train, y_train)

print(mod_svm.score(x_test,y_test));


# In[10]:


from sklearn.ensemble import RandomForestRegressor

mod_Random= RandomForestRegressor();
mod_Random.fit(x_train,y_train);
print(mod_Random.score(x_test,y_test));


# In[11]:


from sklearn.cluster import KMeans
from sklearn import metrics

mod_KMeans= KMeans(n_clusters=4)
mod_KMeans.fit(x_train)
pred = mod_KMeans.predict(x_test)

contingecyMatrix = metrics.cluster.contingency_matrix(y_test, pred)
print (contingecyMatrix)


# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(16,), activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10,batch_size=10)

