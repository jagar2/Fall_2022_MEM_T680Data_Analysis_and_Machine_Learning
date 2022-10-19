#!/usr/bin/env python
# coding: utf-8

# # What is NumPy?
# 

# - NumPy is the fundamental package for scientific computing in Python
# - It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
# 

# At the core of the NumPy package, is the ndarray object.
# 
# - This encapsulates n-dimensional arrays of homogeneous data types, with many operations being performed in compiled code for performance.
# 

# There are several important differences between NumPy arrays and the standard Python sequences:
# 

# - NumPy arrays have a fixed size at creation, unlike Python lists (which can grow dynamically). Changing the size of an ndarray will create a new array and delete the original.
# 

# - The elements in a NumPy array are all required to be of the same data type, and thus will be the same size in memory.
#   - The exception: one can have arrays of (Python, including NumPy) objects, thereby allowing for arrays of different sized elements.
# 

# - NumPy arrays facilitate advanced mathematical and other types of operations on large numbers of data. Typically, such operations are executed more efficiently and with less code than is possible using Python’s built-in sequences.
# 

# - A growing plethora of scientific and mathematical Python-based packages are using NumPy arrays; though these typically support Python-sequence input, they convert such input to NumPy arrays prior to processing, and they often output NumPy arrays.
# 

# NumPy (Numerical Python) is an open source Python library that’s used in almost every field of science and engineering.
# 

# - It’s the universal standard for working with numerical data in Python, and it’s at the core of the scientific Python and PyData ecosystems.
# - NumPy users include everyone from beginning coders to experienced researchers doing state-of-the-art scientific and industrial research and development.
# - The NumPy API is used extensively in Pandas, SciPy, Matplotlib, scikit-learn, scikit-image and most other data science and scientific Python packages.
# 

# ## Exploring How Numpy Speeds Computation Up
# 
# Consider the case of multiplying each element in a 1-D sequence with the corresponding element in another sequence of the same length. If the data are stored in two Python lists, a and b, we could iterate over each element:
# 

# In[1]:


get_ipython().run_cell_magic('time', '', 'a = range(0,1000000)\nb = range(0,1000000)\n\nc = []\nfor i in range(len(a)):\n    c.append(a[i]*b[i])')


# We get the right answer but it takes a long time. There are ways that we could write this in C that are much more efficient. The good thing about python is that someone has done this for you.
# 

# NumPy gives us the best of both worlds: element-by-element operations are the “default mode” when an ndarray is involved, but the element-by-element operation is speedily executed by pre-compiled C code. In NumPy:
# 

# ```{note}
# This is a common thread in Python. Python provides a high-level language to access very efficient source code.
# ```
# 

# In[2]:


import numpy as np

a = np.array(a)
b = np.array(b)


# In[3]:


get_ipython().run_cell_magic('time', '', '\nc = a * b')


# ## Why is NumPy so Fast?
# 
# Vectorization describes the absence of any explicit looping, indexing, etc., in the code - these things are taking place, of course, just “behind the scenes” in optimized, pre-compiled C code. Vectorized code has many advantages, among which are:
# 

# - Vectorized code is more concise and easier to read
# 
# - Fewer lines of code generally means fewer bugs
# 
# - The code more closely resembles standard mathematical notation (making it easier, typically, to correctly code mathematical constructs)
# 
# - Vectorization results in more “Pythonic” code. Without vectorization, our code would be littered with inefficient and difficult to read for loops.
# 
# - The speedup is achieved using broadcasting operations
# 

# In[ ]:




