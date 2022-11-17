#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision.models as models


# # Save and Load the Model
# 
# In this section we will look at how to persist model state with saving, loading and running model predictions.
# 

# ## Saving and Loading Model Weights
# PyTorch models store the learned parameters in an internal
# state dictionary, called ``state_dict``. These can be persisted via the ``torch.save``
# method:
# 
# 

# In[2]:


model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')


# To load model weights, you need to create an instance of the same model first, and then load the parameters
# using ``load_state_dict()`` method.
# 
# 

# In[3]:


model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# <div class="alert alert-info"><h4>Note</h4><p>be sure to call ``model.eval()`` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.</p></div>
# 
# 

# ## Saving and Loading Models with Shapes
# When loading model weights, we needed to instantiate the model class first, because the class
# defines the structure of a network. We might want to save the structure of this class together with
# the model, in which case we can pass ``model`` (and not ``model.state_dict()``) to the saving function:
# 
# 

# In[4]:


torch.save(model, 'model.pth')


# We can then load the model like this:
# 
# 

# In[5]:


model = torch.load('model.pth')


# <div class="alert alert-info"><h4>Note</h4><p>This approach uses Python [pickle](https://docs.python.org/3/library/pickle.html) module when serializing the model, thus it relies on the actual class definition to be available when loading the model.</p></div>
# 
# 

# ## Related Tutorials
# [Saving and Loading a General Checkpoint in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
# 
# 
