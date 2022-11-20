#!/usr/bin/env python
# coding: utf-8

# # Final Exam Assignment (40 points - Total of 45 is Possible)
# ## Due December 7, 2022, @ 8:00 am
# Note that there will be no extensions given for this assignment as there is a tight timeline for grading. 
# 
# For this assignment, I have provided each of you with your own training dataset. Your goal is to train a deep neural network to uncover the code image provided to you. 
# 
# I will provide you with instructions throughout. 

# In[1]:


# Add your import statements here


# In[ ]:


# This is a tool I have provided you to help you download your file.

def download_file(url, filename):
    """
    A function that downloads the data file from a URL
    Parameters
    ----------
    url : string
        url where the file to download is located
    filename : string
        location where to save the file
    reporthook : function
        callback to display the download progress
    """
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename, reporthook)
        
def reporthook(count, block_size, total_size):
    """
    A function that displays the status and speed of the download
    """

    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration + 0.0001))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


# In[ ]:


# You can download your file by typing your first name into the name block
# The name used is the first part of your first name as listed in BB learn
# If you have problems downloading the data please reach out to me

name = 'Your First Name Here'
download_file(f'https://zenodo.org/record/7339649/files/data_{name}.npz?download=1','data.npz')


# ## Loading the Data (3 points)
# The data is provided to you as a compressed NumPy array saved as 'data.npz'. When working with real data you might need to figure out how data is stored. Use the information on 'npz' files to figure out what data you have. The data file contains three NumPy arrays. 
# 1. The features for the training dataset
# 2. The regression values for the training dataset
# 3. The validation features that contain your code

# In[2]:


# Your Code goes here


# ## Preprocessing the Data (5 points)
# 
# You should explore the data and figure out the best way to preprocess the data. 
# 
# Hints: 
# 1. For the regression values, these at the end will represent colors in RGB space from [0,1]. It is recommended to use a max-min scalar between 0 and 1. 
# 2. For the training features, you should look at the data and determine the best scaling method. Look at our class notes for a reminder of what other scaler might be useful. 

# In[3]:


# Your code goes here


# ## Building the Dataset (5 points)
# 
# When training neural networks it is important to build a dataset that allows the machinery to sample the data. This also can be used to conduct some preprocessing of the data to make it work with PyTorch. 
# 
# I have provided you with the framework for a Dataset Class. 
# 
# You should:
# 1. Convert the x and y data to a tensor 'float32' and put it on the GPU.
# 2. Save the len of the data
# 3. Add the code so when `__getitem__` is called it returns the x and y values
# 3. make it so `__len__` returns the lenght when calle
# 

# In[ ]:


class Data(Dataset):
  '''Dataset Class to store the samples and their corresponding labels, 
  and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
  '''

  def __init__(self, X: np.ndarray, y: np.ndarray, device = 'cuda') -> None:

    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = # here
    self.y = # here
    self.len = # here
  
  def __getitem__(self, index: int) -> tuple:
    return # here

  def __len__(self) -> int:
    return #here


# ## Train-test Split (3 points)
# 
# 1. You should conduct a train-test split of the training data so you can make sure that your model does not overfit the data. A good ratio is 66/33 train 
# 2. You should instantiate the training dataset using the data class implemented above.
# 

# In[4]:


# Your code goes here


# ## Build the Dataloader (3 points)
# 
# Pytorch uses DataLoaders to efficiently sample from a training dataset. Instantiate a Pytorch DataLoader using the dataset. 
# 
# You should set the following parameters:
# 1. Batch size = 64
# 2. Shuffle = True

# In[5]:


# Your code goes here


# ## Building a Neural Network (5 points)
# 
# Using the provided class framework which inherits the `nn.Module` type in PyTorch builds a 4-layer neural network to complete the multiple regression.
# 
# 

# In[ ]:


class Neural_Network(nn.Module):
  ''' Regression Model
  ''' 

  # note, you can ignore the `:int` and `-> None` this is just more advanced doctring syntax
  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
      '''The network has 4 layers
            - input layer
            - ReLu
            - hidden layer
            - ReLu
            - hidden layer
            - ReLu
            - output layer
      '''
      super(Neural_Network, self).__init__()
      # in this part you should intantiate each of the layer components
      # Type your code here

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      # In this part you should build a model that returns the 3 outputs of the regression
      # Type your code here
      
      return x


# ## Instantiate the Model (3 points)
# 
# 

# In[ ]:


# number of features (len of X cols)
input_dim = 
# number of hidden layers set this to 50
hidden_layers = 
# Add the number of output dimensions
output_dim =


# In[ ]:


# initiate the regression model
# make sure to put it on your GPU
model = 
print(model)

# criterion to computes the loss between input and target
# Choose a good criteria

# optimizer that will be used to update weights and biases
# you can choose any optimizer. I would recommend ADAM.
# This problem should not be hard to optimize. A good starting learning rate is 3e-5. 


# ## Train the Model (5 points)
# 
# Training the model is conducted in a number of steps using loops.
# 
# 1. Set up a loop for each epoch
# 2. Set a parameter to save the running loss
# 3. Set up a nested loop that goes through the batches from the DataLoader you built
#     - I would recommend using enumerate to include the counts in the loop
#     - The dataloader will return a tuple that is the inputs and the labels
# 4. Conduct the forward propagation of the model
#     - Give the model the inputs and compute the outputs
#     - Compute the loss given the criteria. 
# 5. Use the zero gradient method to remove the gradients from the optimizer
# 6. Use the backward method to compute the gradients
# 7. Use the step method in the optimizer to take an optimization step
# 8. Compute the running loss by calling the item method and adding it to the running loss for each minibatch
# 9. For each epoch print the epoch and the loss
# 
# Note: If you find this challenging I would recommend that you look at examples of other pytorch training loops online. This is a very standard workflow. 

# In[ ]:


# start training
epochs = # sets the number of epochs to train 20 should be sufficent.
# This should take about 5-10 minutes to train.

# Your code should go here


# ## Validate the Model (3 points)
# 
# Use the test dataset from the train-test split to make sure your model is not overfitting
# 
# 1. You can build a dataloader as you did before, this time with the test data.
# 2. Build a validation loop, which you should use `with torch.no_grad()` to make sure you do not modify the gradients, or weights. This will fix your model. 
# 3. Instantiate the loss to be 0.
# 4. Build a similar loop to grab the validation dataset. 
# 5. Compute the predictions with the model.
# 6. Compute the loss using your loss criteria.
# 7. Print the final loss determined.

# In[6]:


# Your code goes here


# <p style="color:blue"> Question: Is your model overfitting or not? How do you know? (3 points) </p>

# Type your response here

# ## Crack Your Code (3 points)
# 
# 1. You can build a dataloader as you did before, this time with the validation features to view your code.
# 2. Build a loop, you should use `with torch.no_grad()` to make sure you do not modify the gradients or weights. This will fix your model. 
# 3. Compute the predictions of your model. 
#     - Make sure you do all the same preprocess, the data has the same datatype, and is on the same device

# In[7]:


# Your code goes here


# ## Reveal Your Code (3 points)
# 
# Your code is an image. there are (65536, 3) predictions this corresponds to a (256,256,3) RGB image. 
# 1. Use the detach() method to remove the gradients from the tensor
# 2. Transfer the tensor back to the 'cpu' if you had it on a GPU
# 3. Reshape the image into a 256,256,3 array. 
# 4. Plot your successful result.

# In[8]:


# Your Code goes here

