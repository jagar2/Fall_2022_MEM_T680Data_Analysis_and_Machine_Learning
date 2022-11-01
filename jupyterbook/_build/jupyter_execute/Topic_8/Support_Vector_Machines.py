#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# # Support Vector Machines
# 

# - A type of discriminate machine learning model
# 

# - They try and build a plane in space that separates examples that belong to different classes with the widest possible margin
# 

# - New examples that are mapped onto that space get "classified" based on which side of the boundary they fall on
# 

# ## Example: Random Data
# 

# In[2]:


# creating datasets X containing n_samples 
# Y containing two classes 
X, Y = make_blobs(n_samples=500, centers=2, 
                  random_state=0, cluster_std=0.40) 
  
# plotting scatters  
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='RdBu'); 
plt.show()  


# In this case, it is very easy to classify the points with a linear boundary
# 

# In[3]:


# creating line space between -1 to 3.5  
xfit = np.linspace(-1, 3.5) 
  
# plotting scatter 
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='RdBu') 
  
# plot a line between the different sets of data 
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]: 
    yfit = m * xfit + b 
    plt.plot(xfit, yfit, '-k') 
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',  
    color='#AAAAAA', alpha=0.4) 
  
plt.xlim(-1, 3.5); 
plt.show() 


# - SVMs try and find the widest possible perpendicular distance between the dividing vector and the points
# 

# ## Example: Breast Cancer Data
# 

# ### Step 1: Load the Data
# 
# We'll use the built-in breast cancer dataset from `Scikit-Learn`. We can get with the load function:
# 

# In[4]:


cancer = load_breast_cancer()


# ### Step 2: Understanding the Data
# 
# The data set is presented in a dictionary form:
# 

# In[5]:


cancer.keys()


# The `DESCR` contains a description of the information in the dataset
# 

# In[6]:


print(cancer['DESCR'])


# In[7]:


# A list of the feature names
cancer['feature_names']


# ### Step 3: Building the Data Structure for Training
# 

# In[8]:


df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()


# In[9]:


# Views how the labels are structured
cancer['target']


# In[10]:


df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])


# ### Step 4: Validate the Data Structure
# 

# In[11]:


df_feat.head()


# **Generally, at this stage, one would visualize that data using domain knowledge**
# 

# ### Step 5: Test-Train Split
# 

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)


# ### Step 6: Train the Model
# 

# In[13]:


model = SVC(gamma='auto')
model.fit(X_train,y_train)


# In[14]:


X_train


# ### Step 7: Visualize the Results
# 

# In[15]:


predictions = model.predict(X_test)
cm = confusion_matrix(y_test,predictions)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
classNames = ['Negative','Positive']
plt.title('Cancer Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


# In[16]:


print(classification_report(y_test,predictions))


# **This is not good ... Our Machine Thinks Everyone has Cancer!!!!**
# 

# **What did we do wrong?**
# 

# ### Step 4a: Scaling the Data
# 

# When models fail:
# 
# 1. Check how the data was normalized
# 2. Optimize the model parameters
# 

# In[17]:


df_feat.head()


# - The features have vastly different scales which makes them weighted unequally
# 

# #### Standard Scaler
# 
# Removes the mean and scales to a unit variance
# 
# $$ z = \frac{x - \mu}{\sigma}$$
# 
# - $\mu$ is the mean
# - $\sigma$ is the standard deviation
# 
# **The result is all features will have a mean of 0 and a variance of 1**
# 

# In[18]:


# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.


scaler = StandardScaler()
df_feat = scaler.fit_transform(df_feat)
df_feat = pd.DataFrame(df_feat,columns=cancer['feature_names'])


# In[19]:


df_feat.head()


# ### Step 5: Test-Train Split
# 

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)


# ### Step 6: Train the Model
# 

# In[21]:


model = SVC(gamma='auto')
model.fit(X_train,y_train)


# ### Step 7: Visualize the Results
# 

# In[22]:


predictions = model.predict(X_test)
cm = confusion_matrix(y_test,predictions)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
classNames = ['Negative','Positive']
plt.title('Cancer Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


# **That looks way better**
# 

# In[ ]:




