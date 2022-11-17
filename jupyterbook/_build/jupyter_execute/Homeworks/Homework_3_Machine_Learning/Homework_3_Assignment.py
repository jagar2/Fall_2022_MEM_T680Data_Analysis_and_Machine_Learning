#!/usr/bin/env python
# coding: utf-8

# # Homework 3: Machine Learning Tasks (XX/125 points)
# ## Due Monday 12/7/2022 11:59 pm
# ## About Dataset
# ### Data from a semi-conductor manufacturing process
# 
# Number of Instances: 1567 <br>
# Area: Computer<br>
# Attribute Characteristics: Real<br>
# Number of Attributes: 591<br>
# Date Donated: 2008-11-19<br>
# Associated Tasks: Classification, Causal-Discovery<br>
# Missing Values? Yes<br>
# 
# A complex modern semi-conductor manufacturing process is normally under consistent
# surveillance via the monitoring of signals/variables collected from sensors and or
# process measurement points. However, not all of these signals are equally valuable
# in a specific monitoring system. The measured signals contain a combination of
# useful information, irrelevant information as well as noise. It is often the case
# that useful information is buried in the latter two. Engineers typically have a
# much larger number of signals than are actually required. If we consider each type
# of signal as a feature, then feature selection may be applied to identify the most
# relevant signals. The Process Engineers may then use these signals to determine key
# factors contributing to yield excursions downstream in the process. This will
# enable an increase in process throughput, decreased time to learning and reduce the per unit production costs.
# 
# To enhance current business improvement techniques the application of feature
# selection as an intelligent systems technique is being investigated.
# 
# The dataset presented in this case represents a selection of such features where
# each example represents a single production entity with associated measured
# features and the labels represent a simple pass/fail yield for in-house line testing, figure 2, and associated date time stamp. Where -1 corresponds to a pass
# and 1 corresponds to a fail and the data time stamp is for that specific test
# point.
# 
# This homework assignment will walk you through how to tackle this real problem. 
# 
# It is worth noting that this is an actual dataset and thus the problem is not fully tractable. Like many real problems, sometimes you do not have the necessary information to make a perfect solution. We want a useful and informative solution. 

# In[1]:


## Here are some packages and modules that you will use. Make sure they are installed.

# for basic operations
import numpy as np 
import pandas as pd 

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for modeling 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import plotly_express as px

from imblearn.over_sampling import SMOTE

# to avoid warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ## Loading the data (5 points)
# 
# 1. The data file is in a file called `uci-secom.csv` scikit-learn works well with pandas. It is recommended that you read the csv into a pandas array.
# 2. It is useful to print the shape of the output array to know what the data is that you are working with
# 

# In[2]:


# Your code goes here


# 3. Pandas has a built-in method called head that shows the first few rows, this is useful to see what the data looks like 

# In[3]:


# Your code goes here


# ## Filtering Data (5 points)
# 
# Real data is usually a mess. There could be missing points, outliers, and features that could have vastly different values and ranges. 
# * Machine learning models are influenced heavily by these problems 
# 

# ### Fixing Missing Values (5 Points)
# 
# 1. It is not uncommon that some of the features have only a few entries. These are not helpful for machine learning. We should just remove these features from the data
# 
# It is good to visualize how many missing values each feature has.

# ### Plotting the Missing Data Entries (5 points)
# 
# Hint: you can find the nan values with the `.isna()` method, and the sum using `.sum()`
# 
# You can plot the data using `px.histogram`

# In[4]:


# Your code goes here


# ### Removing Sparse Features (10 points)
# 
# We can remove the features that have more than 100 missing entries. 
# 
# You can find the location where a condition is met in a Pandas array using `data.loc[:,:]` with traditional numpy-like indexing

# In[5]:


# Your code goes here


# Remove these columns in the dataframe using the `.drop()` method, make sure inplace is set to True

# In[6]:


# Your code goes here


# It is useful to check the shape to make sure that the operation worked

# In[7]:


# Your code goes here


# It is useful to see how many data points have missing information
# 
# Hint: you can change the axis of the `isna()`
# 
# You can use the built-in method `value_counts()` to view the number of samples with bad rows.

# In[8]:


# Your code goes here


# Since it is not that many we can remove these rows.
# * you can just index the data that you want using the built-in method `.loc()`

# In[9]:


# Your code goes here


# You should grab the features and labels you can do this by:
# 1. Using the Pandas `drop` built-in method, you can also drop the time
# 2. You should set the the prediction to be if the dataset passed or failed
# 3. It is a good idea to replace the -1 values with 0 using the pandas built-in method `.replace`

# In[10]:


# Your code goes here


# ## Test-train split (5 points)
# 
# Use the `train_test_split` method to split the data.
# 
# For consistency make the `test_size = .3`, and the `random_state = 42`.

# In[11]:


# Your code goes here


# ## Machine Learning (5 points)
# 
# It is always good to try a quick machine learning model. If your data is simple it might just work.
# 
# * Implement a `LogisticRegression` from scikit-learn
# * Fit the model

# In[12]:


# Your code goes here


# It is always a good idea to see if the model fit the data well.
# * uses the `.predict()` method to predict on the training data
# * use the sklearn function `classification_report` to evaluate the model
# * use the sklearn function `confusion_matrix`, you can plot this in plotly using `px.imshow()`
# 
# You will reuse these lines of code to visualize your results

# In[13]:


# Your code goes here


# Now use the same approach to visualize the test results

# In[14]:


# Your code goes here


# <span style="color:blue">Question: Describe what might be wrong with this model, does it provide any practical value? (5 points)</span>

# ### Try Another Model (5 points)
# 
# It could be that we just selected a bad model for the problem, try with a random forest classifier as implemented in scikit-learn
# * Instantiate the model
# * Fit the data

# In[15]:


# Your code goes here


# Validate the model on the training and testing dataset

# In[16]:


# Your code goes here


# In[17]:


# Your code goes here


# That still does not do anything meaningful

# ## Normalizing the Data (10 points)
# 
# Machine learning models prefer features with a mean of 0 and a standard deviation of 1. This makes the optimization easier.
# 
# 1. Make a histogram of the mean and standard deviation you can use the built-in method `.mean()` and `.std()`
# 2. You can plot this using `px.histogram`

# In[18]:


# Your code goes here


# In[19]:


# Your code goes here


# Scikit-learn has a utility function for conducting standard scalars `StandardScaler()`
# 
# We could implement the standard scaler in steps but it is more convenient to do it with a pipeline

# ### Scaled Logistic Regression (5 points)
# 
# 1. Use the `Pipeline` utility to create a machine learning model that:
#     - Computes the standard scalar of the data
#     - Conducts logistic regression
# 2. Fit the model

# In[20]:


# Your code goes here


# Visualize the results as you have done before

# In[21]:


# Your code goes here


# In[22]:


# Your code goes here


# ### Standard Scaled Random Forest (5 points)

# 1. Use the `Pipeline` utility to create a machine learning model that:
#     - Computes the standard scalar of the data
#     - Conducts Random Forest
# 2. Fit the model

# In[23]:


# Your code goes here


# In[24]:


# Your code goes here


# <span style="color:blue"> Question: Explain what is going on with the random forest model? Why are the results so bad? (5 points)</span>

# ## Feature Reduction
# 
# ### Logistic Regression (5 Points)
# 
# We can use PCA to reduce the number of features such that highly covariant features are combined. This helps deal with the curse of dimensionality.
# 
# Add PCA to the pipeline for the logistic regression, and visualize the results as we have done before

# In[25]:


# Your code goes here to build and fit the model


# In[26]:


# Your code goes here to validate the training performance


# In[27]:


# Your code goes here to validate the test performance


# ### Random Forest (5 points)
# 
# Add PCA to the pipeline for the logistic regression, and visualize the results as we have done before

# In[28]:


# Your code goes here to build and fit the feature reduced random forest classifier


# In[29]:


# Your code goes here to visualize the training results


# In[30]:


# Your code goes here to visualize the test results


# <span style="color:blue"> Question: Explain if adding PCA helped, explain why you think PCA helped or did not help. (5 points)</span> 

# ## Hyperparameter Tuning 
# 
# To improve a machine learning model you might want to tune the hyperparameters.
# 
# Scikit-learn has automated tools for cross-validation and hyperparameter search. You can just define a dictionary of values that you want to search and it will try all of the fits returning the best results

# ### Logistic Regression (7.5 points)
# 
# Conduct build a pipeline and build a parameter grid to search the following hyperparameters:
# 1. C = [ 0.001, .01, .1, 1, 10, 100]
# 2. penalty = ['l1', 'l2']
# 3. class_weight = ['balanced']
# 4. solver = ['saga']
# 5. PCA n_components = [2, 3, 4, 5, 8, 10]
# 
# To conduct the fitting you should build the classifier with GridSearchCV. This conducts a grid search with cross-folds. See the documentation for more information. 
# 
# For the GridSearchCV set the scoring to 'f1', cv=5. If you want to monitor the status you can set verbose=10. 

# In[31]:


# Your code goes here


# You should look and see what the best estimator was from the search. 
# 
# You can use the `best_estimator_` method

# In[32]:


# your code to evaluate the training process


# In[33]:


# Your code goes here for validating the model


# ### Random Forest Classifier (7.5 points)
# 
# Conduct build a pipeline and build a parameter grid to search the following hyperparameters:
# 1. Random Forest Criterion = [ "gini", "entropy", "log_loss"]
# 2. max depth = [4, 8, 12]
# 3. max features = ['sqrt', 'log2']
# 4. PCA n_components = [4, 8, 10, 20]
# 
# To conduct the fitting you should build the classifier with GridSearchCV. This conducts a grid search with cross-folds. See the documentation for more information. 
# 
# For the GridSearchCV set the scoring to 'f1', cv=5. If you want to monitor the status you can set verbose=10. 

# In[34]:


# Your code goes here


# In[35]:


# Your code to show the best estimator goes here


# In[36]:


# Your code to evaluate the training process


# In[37]:


# Your code to evaluate the performance of the model


# ## Balancing in the Data (10 points)
# 
# This is a classification problem, it is useful to see if the classes are balanced as this affects model training. 
# 
# If you have a highly unbalanced dataset you can train a model to predict the most common classes but getting the uncommon classes wrong has little effect on the model performance metrics.
# 
# View the ratio of the class outcomes. 
# 
# The class outcomes are stored in the ['pass/fail'] column, you can view the values and counts using the `.value_count()` built-in method

# In[38]:


# Your code goes here


# Use `SMOTE(`)` to balance the dataset

# In[39]:


# Your code goes here


# In[40]:


# Check that this worked as expected


# Using the balanced dataset repeat the analysis done with the hyperparameter search
# 
# ### Logistic Regression (5 points)

# In[41]:


# Your fitting code here


# In[42]:


# Your code to show the best estimator


# In[43]:


# Your code to check the training process


# In[44]:


# Your code to validate the training process


# ### Random Forest Classifier (5 Points)

# In[45]:


# Your code to conduct the classifier


# In[46]:


# Your code to show the best estimator


# In[47]:


# Your code to validate the training process


# In[48]:


# Your code to validate the model performance


# <span style="color:blue">Question: Given that you are trying to predict and determine the underlying features responsible for producing products that "pass" which model would be better and why? (5 points)</span>

# ## Bonus (10 Points): 
# 
# Use any method available to you to get a better validation F1 score for the Pass Classification

# A trick to always get better results is to use ensemble methods
