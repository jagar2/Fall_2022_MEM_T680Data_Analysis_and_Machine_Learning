#!/usr/bin/env python
# coding: utf-8

# # Setting up your Computing Environment
# 
# Python is highly flexible, thus there are a lot of good ways to set up your computing environment, however, each have their own advantages and disadvantages.

# ## Introduction to the JupyterLab and Jupyter Notebooks
# 
# This is a short introduction to two of the flagship tools created by [the Jupyter Community](https://jupyter.org).

# ### JupyterLab 🧪
# 
# **JupyterLab** is a next-generation web-based user interface for Project Jupyter. It enables you to work with documents and activities such as Jupyter notebooks, text editors, terminals, and custom components in a flexible, integrated, and extensible manner. It is the interface that you're looking at right now.
# 
# > **See Also**: For a more in-depth tour of JupyterLab with a full environment that runs in the cloud, see [the JupyterLab introduction on Binder](https://mybinder.org/v2/gh/jupyterlab/jupyterlab-demo/HEAD?urlpath=lab/tree/demo).

# ### Jupyter Notebooks 📓
# 
# **Jupyter Notebooks** are a community standard for communicating and performing interactive computing. They are a document that blends computations, outputs, explanatory text, mathematics, images, and rich media representations of objects.
# 
# JupyterLab is one interface used to create and interact with Jupyter Notebooks.

# In this course we will primarily use Jupyter Notebooks as an interface. Jupyter is extremely powerful. All the content in this course (including the book) was built with Jupyter. We will discuss more about using Jupyter later. First, we need a way to access Jupyter.

# ## Google Colab
# 
# ### What is Colab?
# 
# Colab, or "Colaboratory", allows you to write and execute Python in your browser, with 
# - Zero configuration required
# - Access to GPUs free of charge
# - Easy sharing
# 
# Whether you're a **student**, a **data scientist** or an **AI researcher**, Colab can make your work easier. Watch [Introduction to Colab](https://www.youtube.com/watch?v=inN8seMm7UI) to learn more.

# ### Advantages:
# * You can just go to [colab](https://colab.research.google.com/) and you are off to the races

# * The interface works like a google docs, there are good autosaving and colaboration features

# * The interface is a standard jupyter notebook, files are fully-interoperable with other platforms

# * The interface is a standard jupyter notebook, files are fully-interoperable with other platforms

# * Many of the most important software packages are pre-installed, you can also install your own packages

# * Google provides you free access to a decent GPU for machine learning applications

# * Since all the computation happens on google's servers you can run colab on any device, even a cellphone

# ### Disadvantages:
# * Limited access to the command line interface

# * File system is either temporary or restricted to google drive

# * Requires a Google account

# * Limited number of active sessions per users

# * Provides limited CPU cores

# * Has built-in timeouts (premium services can improve this)

# * Limited access to terminal and interactive python functionalities

# ## JupyterHub
# 
# If you have a server you can setup a JupyterHub. This is hosting for a python instance with preconfigured environments.

# At Drexel we have setup a small JupyterHub. To access the JupyterHub.
# 
# 1. Log into the Drexel network or connect using the [VPN](https://drexel.edu/it/help/a-z/VPN/).
# 1. The JupyterHub can be accessed at [https://jupyterhub.coe.drexel.edu](https://jupyterhub.coe.drexel.edu)
# 1. To get access you will need to have an account made for you. Please contact [Andrew Marx](atm26@drexel.edu) regarding how to create an account.
# 1. Once up and running you will have access to a JupyterLab instance

# ### Advantages:
# * This is a standard Jupyter interface accessible from a web browsers

# * Can be configured with custom packages installed

# * Provides sufficient command line and file system access

# * Can provide burstible computational capabilities 

# ### Disadvantages
# * Restricted to the web browser, independent graphical applications cannot be used.
# * The Drexel server is underpowered, there is currently no GPU access. 

# ## Local Installation

# A major advantage of Python is it is open-source and free. Unlike Matlab, anyone in the world can download and use Python!
# 
# Python is available for Windows, Linux, and Mac. 
# 
# Since Python is open source, there are many ways to install it, and various distributions. 

# ### Recommended Installation (Anaconda)

# I recommend installing the anaconda distribution of python, as this comes with:
# * Many of the useful packages pre-installed
# * Conda package manager
# * Useful tools for managing python environments

# #### Installation on Windows

# In[1]:


from IPython.display import IFrame

IFrame(src="https://docs.anaconda.com/anaconda/install/windows/", width=1200, height=600)


# #### MacOS

# In[2]:


IFrame(src="https://docs.anaconda.com/anaconda/install/mac-os/", width=1200, height=600)


# #### Linux

# In[3]:


IFrame(src="https://docs.anaconda.com/anaconda/install/linux/", width=1200, height=600)


# #### Running Jupyter

# To run Jupyter Notebook you can type `jupyter notebook` in the Anaconda Prompt. If you want to run JupyterLab you can type `jupyter lab`

# ### Advantages:
# 
# * You are in full control of your Python environment

# * All features are available

# * You can install packages (this is rare) That requires admin access

# * You have access to your file system

# * You can have as much computing capabilities as you can afford. 

# ### Disadvantages
# 
# * You can do things (if you copy paste commands blindly from online sites) that can mess up your python instance or mess with apps on your computer

# * You have to manage your Python environment

# * You have to work in the command line

# * You have to use your computing resources - Most people today rely on laptops as their primary computer. These rarely have GPUs for machine learning. If they do, it will burn through your battery fast. 

# ## Managing a Python Environment

# With conda, you can create, export, list, remove, and update
# environments that have different versions of Python and/or packages
# installed in them. Switching or moving between environments is called
# activating the environment. You can also share an environment file.

# It is important to create environments because certain scripts require specific packages and versions. Environments ensure that scripts can run.

# There are many options available for the commands described on this
# page. For details, see [Command Reference](https://conda.io/projects/conda/en/latest/commands.html)

# ### Creating an Environment

# 1. Using the terminal or anaconda prompt

# ```conda create --name myenv```

# ```{note}
# Replace `myenv` with the name of the env
# ```

# 2. When conda asks you to proceed, type `y`:

# ```
# proceed ([y]/n)?
# ```

# This creates the myenv environment in `/envs/`. No packages will be installed in this environment.

# 3. To create an environment with a specific version of Python:

# ```
# conda create -n myenv python=3.9
# ```

# ```{note}
# At a minimum you should always be using Python Version 3
# ```

# 4. To create an environment with a specific package:

# ```
# conda create -n myenv scipy
# ```
# 

# or, if you want a specific version of the package

# ```
# conda install -n myenv scipy=0.17.3
# ```

# ### Creating an Environment from the `environment.yml` file

# Sometimes, you want to ship or use a previously developed environment this can be done by creating a `.yml` file

# 1. Create the environment from the `environment.yml` file:

# ```
# conda env create -f environment.yml
# ```

# For details see [Creating an environment file manually](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually).

# 2. You can check that the env was installed or view env using:

# ```
# conda env list
# ```

# 3. Activating an env:

# ```
# conda activate env
# ```

# More information on managing environments can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)

# ## Using Interactive Development Environments (IDE)
# 
# IDEs are software that provides a more complex development environment than Jupyter. These are great for building large scale projects but sometimes are overwhelming for beginners. 
# 
# There are many good, and free IDEs: 
# 1. [VScode](https://code.visualstudio.com/download)
# 1. [Atom](https://atom.io/)
# 1. [Pycharm](https://www.jetbrains.com/pycharm/download)
# 
# My current preference is to use VScode, however, all are good. 

# ### Advantages:
# 
# * Better systems to manage files and large packages

# * Integration with Python locally or even through remote SSH

# * Tons of addons for connecting to apps and services (e.g., github, Jupyter, Docker, etc.)

# * Tools for formatting including spellcheck, autocomplete, search, autoformatting, etc.

# ### Disadvantages
# 
# * More complex interface
# * Bigger learning curve

# ## Recommendations
# 
# It is up to you and your comfort level what tools you use in this course. All material (with possibly the exception of your project) are designed to be conducted in Google Colab. 

# I will list my preferred options:
# 1. Google Colab - I would recommend using google colab. For the sections on deep learning I would recommend purchasing colab pro ($10/month). Remember there is no book or software that you have to buy for this course.
# 1. Local Instance - A local instance could be okay, particularly if you have a laptop or desktop with a high-powered NVIDIA GPU. If you choose to go this route expect it to be more challenging.
# 1. Drexel Jupyter Hub - This service would be sufficient for the beginning of the course where computational requirements are minimal. 

# If your project is related to research and you require further computing capabilities other options exist:
# 1. You can pay for cloud services through Google, Azure, AWS, etc.
# 1. URCF has GPUs resources that can be used for a fee
# 1. You can talk with me and I might have some additional GPU resources available
