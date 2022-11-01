#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd


from sklearn import set_config
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

set_config(display="diagram")


# # Hyperparameter tuning by grid-search
# 
# There are many parameters that need to be tuned when building a machine-learning model.
# 
# - It is non-intuitive what these parameters should be
# 

# ## Example: Adults Census Data
# 
# This is a dataset that contains information from the 1994 Census. The goal is to predict if a person makes more than \$50,000 per year based on their demographic information.
# 

# ### Step 1: Load the Data
# 

# In[2]:


adult_census = pd.read_csv("./data/adult-census.csv")


# ### Step 2: Visualize and Clean the Data
# 

# We extract the column containing the target.
# 

# In[3]:


target_name = "class"
target = adult_census[target_name]
target


# We drop from our data the target and the `"education-num"` column which
# duplicates the information from the `"education"` column.
# 

# In[4]:


data = adult_census.drop(columns=[target_name, "education-num"])
data.head()


# ### Step 3: Test Train Split
# 

# Once the dataset is loaded, we split it into training and testing sets.
# 

# In[5]:


data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)


# ### Step 4: Building a Pipeline
# 
# Within Scikit-learn there are tools to build a pipeline to automate preprocessing and data analysis.
# 

# We will define a pipeline. It will handle both numerical and categorical features.
# 
# The first step is to select all the categorical columns.
# 

# In[6]:


categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)


# Here we will use a tree-based model as a classifier
# (i.e. `HistGradientBoostingClassifier`). That means:
# 
# - Numerical variables don't need scaling;
# - Categorical variables can be dealt with by an `OrdinalEncoder` even if the
#   coding order is not meaningful;
# - For tree-based models, the `OrdinalEncoder` avoids having high-dimensional
#   representations.
# 
# We now build our `OrdinalEncoder` by passing it to the known categories.
# 

# In[7]:


categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)


# We then use a `ColumnTransformer` to select the categorical columns and apply the `OrdinalEncoder` to them.
# 
# - The `ColumnTransformer` applies a transformation to the columns of a Pandas dataframe.
# 

# In[8]:


preprocessor = ColumnTransformer([
    ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)


# Finally, we use a tree-based classifier (i.e. histogram gradient-boosting) to
# predict whether or not a person earns more than 50 k$ a year.
# 

# In[9]:


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))])
model


# ### Step 5: Tuning the Model Using GridSearch
# 
# - `GridSearchCV` is a scikit-learn class that implements a very similar logic with less repetitive code.
# 
# Let's see how to use the `GridSearchCV` estimator for doing such a search. Since the grid search will be costly, we will only explore the combination learning rate and the maximum number of nodes.
# 

# In[10]:


get_ipython().run_cell_magic('time', '', "param_grid = {\n    'classifier__learning_rate': (0.01, 0.1, 1, 10),\n    'classifier__max_leaf_nodes': (3, 10, 30)}\nmodel_grid_search = GridSearchCV(model, param_grid=param_grid,\n                                 n_jobs=2, cv=2)\nmodel_grid_search.fit(data_train, target_train)\n")


# ### Step 6: Validation of the Model
# 

# Finally, we will check the accuracy of our model using the test set.
# 

# In[11]:


accuracy = model_grid_search.score(data_test, target_test)
print(
    f"The test accuracy score of the grid-searched pipeline is: "
    f"{accuracy:.2f}"
)


# <div class="admonition warning alert alert-danger">
# <p class="first admonition-title" style="font-weight: bold;">Warning</p>
# <p>Be aware that the evaluation should normally be performed through
# cross-validation by providing <tt class="docutils literal">model_grid_search</tt> as a model to the
# <tt class="docutils literal">cross_validate</tt> function.</p>
# <p class="last">Here, we used a single train-test split to evaluate <tt class="docutils literal">model_grid_search</tt>.
# If interested you can look into more detail about nested cross-validation,
# when you use cross-validation both for hyperparameter tuning and model
# evaluation.</p>
# </div>
# 

# The `GridSearchCV` estimator takes a `param_grid` parameter which defines
# all hyperparameters and their associated values. The grid search will be in
# charge of creating all possible combinations and testing them.
# 

# The number of combinations will be equal to the product of the
# number of values to explore for each parameter (e.g. in our example 4 x 3
# combinations). Thus, adding new parameters with their associated values to be
# explored become rapidly computationally expensive.
# 

# Once the grid search is fitted, it can be used as any other predictor by
# calling `predict` and `predict_proba`. Internally, it will use the model with
# the best parameters found during the `fit`.
# 

# Get predictions for the 5 first samples using the estimator with the best
# parameters.
# 

# In[12]:


model_grid_search.predict(data_test.iloc[0:5])


# You can know about these parameters by looking at the `best_params_`
# attribute.
# 

# In[13]:


print(f"The best set of parameters is: "
      f"{model_grid_search.best_params_}")


# In addition, we can inspect all results which are stored in the attribute
# `cv_results_` of the grid-search. We will filter some specific columns
# from these results.
# 

# In[14]:


cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
cv_results.head()


# Let us focus on the most interesting columns and shorten the parameter
# names to remove the `"param_classifier__"` prefix for readability:
# 

# In[15]:


# get the parameter names
column_results = [f"param_{name}" for name in param_grid.keys()]
column_results += [
    "mean_test_score", "std_test_score", "rank_test_score"]
cv_results = cv_results[column_results]


# In[16]:


def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


cv_results = cv_results.rename(shorten_param, axis=1)
cv_results


# ### Step 7: Visualizing Hyperparameter Tuning
# 

# With only 2 parameters, we might want to visualize the grid search as a
# heatmap. We need to transform our `cv_results` into a dataframe where:
# 
# - the rows will correspond to the learning-rate values;
# - the columns will correspond to the maximum number of leaf;
# - the content of the dataframe will be the mean test scores.
# 

# In[17]:


pivoted_cv_results = cv_results.pivot_table(
    values="mean_test_score", index=["learning_rate"],
    columns=["max_leaf_nodes"])

pivoted_cv_results


# We can use a heatmap representation to show the above dataframe visually.
# 

# In[18]:


ax = sns.heatmap(pivoted_cv_results, annot=True, cmap="YlGnBu", vmin=0.7,
                 vmax=0.9)
ax.invert_yaxis()


# The above tables highlight the following things:
# 
# - for too high values of `learning_rate`, the generalization performance of the
#   model is degraded and adjusting the value of `max_leaf_nodes` cannot fix
#   that problem;
# - outside of this pathological region, we observe that the optimal choice
#   of `max_leaf_nodes` depends on the value of `learning_rate`;
# - in particular, we observe a "diagonal" of good models with an accuracy
#   close to the maximal of 0.87: when the value of `max_leaf_nodes` is
#   increased, one should decrease the value of `learning_rate` accordingly
#   to preserve a good accuracy.
# 

# For now, we will note that, in general, **there are no unique optimal parameter
# setting**: 4 models out of the 12 parameter configurations that reach the maximal
# accuracy (up to small random fluctuations caused by the sampling of the
# training set).
# 

# In[ ]:




