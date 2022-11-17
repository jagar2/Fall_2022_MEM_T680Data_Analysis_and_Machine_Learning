#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


# # Transforms
# 
# Data does not always come in its final processed form that is required for
# training machine learning algorithms. We use **transforms** to perform some
# manipulation of the data and make it suitable for training.
# 

# All TorchVision datasets have two parameters -`transform` to modify the features and
# `target_transform` to modify the labels - that accept callables containing the transformation logic.
# The [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) module offers
# several commonly-used transforms out of the box.
# 

# The FashionMNIST features are in PIL Image format, and the labels are integers.
# For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors.
# To make these transformations, we use `ToTensor` and `Lambda`.
# 

# In[2]:


ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)


# ## ToTensor()
# 
# [ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor)
# converts a PIL image or NumPy `ndarray` into a `FloatTensor`. and scales
# the image's pixel intensity values in the range [0., 1.]
# 

# ## Lambda Transforms
# 
# Lambda transforms apply any user-defined lambda function. Here, we define a function
# to turn the integer into a one-hot encoded tensor.
# It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls
# [scatter\_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html) which assigns a
# `value=1` on the index as given by the label `y`.
# 

# In[3]:


target_transform = Lambda(
    lambda y: torch.zeros(10, dtype=torch.float).scatter_(
        dim=0, index=torch.tensor(y), value=1
    )
)


# ### Further Reading
# 
# - [torchvision.transforms API](https://pytorch.org/vision/stable/transforms.html)
# 
