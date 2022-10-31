#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Implementation of Simple Linear Regression model 


# In[18]:


#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn import linear_model


# In[7]:


#Reading the dataset using Pandas
df_simple = pd.read_csv('C:\\Users\\supra\\Desktop\Data_sets\\Salary_Data.csv')


# In[8]:


#Displaying top 5 rows in the data
df_simple.head()


# In[9]:


# Some information regarding the columns in the data
df_simple.info()


# In[10]:


# this describes the basic stat behind the dataset used 
df_simple.describe()


# In[11]:


#Plots that help to explain the values and how they are scattered

plt.figure(figsize=(12,6))
sns.pairplot(df_simple,x_vars=['YearsExperience'],y_vars=['Salary'],size=7,kind='scatter')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()


# In[12]:


# Cooking the data
X = df_simple['YearsExperience']
X.head()


# In[13]:


# Cooking the data
y = df_simple['Salary']
y.head()


# In[15]:


# Import Segregating data from scikit learn
from sklearn.model_selection import train_test_split


# In[16]:


# Split the data for train and test 
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)


# In[19]:


# Create new axis for x column
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]


# In[21]:


# Importing Linear Regression model from scikit learn
from sklearn.linear_model import LinearRegression


# In[22]:


# Fitting the model
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[23]:


# Predicting the Salary for the Test values
y_pred = lr.predict(X_test)


# In[24]:


# Plot the actual and predicted values

c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle='-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()


# In[25]:


# plotting the error
c = [i for i in range(1,len(y_test)+1,1)]
plt.plot(c,y_test-y_pred,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()


# In[26]:


# Importing metrics for the evaluation of the model
from sklearn.metrics import r2_score,mean_squared_error


# In[28]:


#calculate Mean square error
mse = mean_squared_error(y_test,y_pred)


# In[29]:


# Calculate R square vale
rsq = r2_score(y_test,y_pred)


# In[30]:


print('mean squared error :',mse)
print('r square :',rsq)


# In[31]:


# Just plot actual and predicted values for more insights
plt.figure(figsize=(12,6))
plt.scatter(y_test,y_pred,color='r',linestyle='-')
plt.show()


# In[32]:


# Intecept and coeff of the line
print('Intercept of the model:',lr.intercept_)
print('Coefficient of the line:',lr.coef_)


# In[ ]:




