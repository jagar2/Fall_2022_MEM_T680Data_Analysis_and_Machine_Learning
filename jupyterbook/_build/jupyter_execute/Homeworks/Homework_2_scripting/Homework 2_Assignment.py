#!/usr/bin/env python
# coding: utf-8

# # Homework 2: Learning How to Script in Python
# 
# ## Due: 10/11/2022 @ 5:00 pm
# 

# ### Question 1: Building Rocks, Paper, Scissors (40 Points)
# 
# Rock paper scissors (also known by other orderings of the three items, with "rock" sometimes being called "stone," or as Rochambeau, roshambo, or ro-sham-bo) is a hand game originating from China, usually played between two people, in which each player simultaneously forms one of three shapes with an outstretched hand. These shapes are "rock" (a closed fist), "paper" (a flat hand), and "scissors" (a fist with the index finger and middle finger extended, forming a V). "Scissors" is identical to the two-fingered V sign (also indicating "victory" or "peace") except that it is pointed horizontally instead of being held upright in the air.
# 
# A simultaneous, zero-sum game, it has three possible outcomes: a draw, a win or a loss. A player who decides to play rock will beat another player who has chosen scissors ("rock crushes scissors" or "breaks scissors" or sometimes "blunts scissors"), but will lose to one who has played paper ("paper covers rock"); a play of paper will lose to a play of scissors ("scissors cuts paper"). If both players choose the same shape, the game is tied and is usually immediately replayed to break the tie. The game spread from China while developing different variants in signs over time.
# 
# It is your job to figure out how to implement this code. It should have the following functions:
# 
# 1. Allows the user to select the number of games that defines a win (e.g., first to 3) (5-points).
# 1. Allows the user to select their choice on trial (5-points).
# 1. Print a statement saying if the player or computer won (the computer's choice is random) a single game (5-points).
# 1. Simple error handling if an input is not a valid choice (5-points).
# 1. Control to exit if the user types an "exit" command (5-points).
# 1. Tells the player who won the overall game (5-points).
# 
# For a working and playable game (10-points).
# 
# Hints:
# 
# 1. The `randint` package can be used to generate a random integer
# 2. `break` can be used to exit a loop and return
# 3. The `input` function can be used to ask the user for an input
# 4. `isinstance` command can tell you if a variable is of a specific type
# 
# You should always think about fail cases. How can you make the code fail gracefully when someone does something not 100% correct?
# 
# **Make sure to comment on your code**
# 

# In[ ]:


# Type your code here.


# ### Question 2: Automating Data Analysis (60 points)
# 
# Note in this problem you can earn a total of 70 points, it will be graded out of 60.
# 
# It is quite common to want to conduct a similar analysis on a large set of experimental data. Dielectrics are an important class of materials used for signal processing and resonators. Their voltage dependent properties are important to determine their function. It is your job to determine which material is best for a specific application. You should determine:
# 
# 1. Which material is best for a voltage-tunable dielectric?
#    - This material is used for RF communications
#    - You should be looking for a material where the dielectric constant changes significantly with applied electric fields
# 1. Which material is the best high-frequency dielectric?
#    - These are used for filters and as gate dielectrics
# 1. Which material is a ferroelectric near the morphotropic phase boundary?
#    - This is a material exhibiting a maximum in the dielectric response
#    - This material should have two regimes of dielectric response. The first is where there are extrinsic domain wall contributions, the second is after they are quenched
# 1. Which material is a ferroelectric far from the morphotropic phase boundary?
#    - This is a material with a lower dielectric constant
#    - This material should show two regimes of response, an intrinsic and extrinsic
# 1. Which material is a non-ferroelectric dielectric?
#    - This material should have a lower dielectric response
#    - This material will only have one type of behavior
# 
# For help you can refer to [the helpful paper link](https://m3-learning.com/wp-content/uploads/2018/11/Tuning-Susceptibility-via-Misfit-Strain-in-Relaxed-Morphotropic-.pdf)
# 
# We will walk you through the steps in completing this analysis. Some code will be provided. You will have to finish the code.
# 

# In[1]:


# imports some tools you will use

# numpy the matrix multiplication and basic mathematics package
import numpy as np

# matplotlib - this is the most common plotting library 
import matplotlib.pyplot as plt


# #### Use the `np.load` function to extract the data (5 points)
# 
# - Hint: This data was saved as a pickle
# - Hint: Since the data was stored in a dictionary use `.item()` built-in method to extract the dictionary
# 

# In[2]:


# Your code goes here


# #### Data Storage
# 
# The data is stored in a structure called a dictionary. Items in a dictionary can be accessed by typing `Data['name']`
# 
# ##### Description of data
# 
# `data['dc_voltage_vector']` - DC bias applied to the sample
# 
# `data['ac_frequency']` - frequency of the measurement
# 
# `data['sample_1']` - data for sample 1
# 
# `data['sample_2']` - data for sample 2
# 
# `data['sample_3']` - data for sample 3
# 

# You can view the shape of the data by typing `data['sample_1'].shape`
# 

# In[ ]:


data['sample_1'].shape


# Each sample is measured at 50 DC-voltages and 100 frequencies
# 

# #### Converting the Data to Dielectric Permittivity
# 
# The measurement consists of measuring the capacitance of a parallel plate capacitor. Units are given in Farads.
# 
# In the markdown box below write the equation to convert the capacitance to the relevant geometry independent materials parameter.
# 
# You can read about how to calculate the Dielectric Permittivity from Capacitance [here](http://hyperphysics.phy-astr.gsu.edu/hbase/electric/pplate.html)
# 
# Please use LaTex for this.
# 
# Needed syntax:
# 
# LaTex equation - `$write your equation here$`
# 
# Fraction - `\frac{a}{b}` $\frac{a}{b}$
# 
# epsilon - `\varepsilon` $\varepsilon$
# 
# subscript - `a_{b}` $a_{b}$
# 

# #### Task 1: Using Markdown write the expression for a parallel plate capacitor (5 points)
# 

# 

# #### Task: Assuming that all capacitors have a radius of 50 $\mu$ m and are 200 nm thick write a function that applys the necessary conversion (5 points)
# 

# In[3]:


# Your code goes here


# #### Task: Visualizing the Data (15 Points)
# 
# When conducting data analysis is it always good to visualize the data in its raw form.
# 
# In this dataset we have 3 samples, on each of which we conducted 50 measurements with different DC electric fields (labeled by the vector dc_voltage_vector). Within each of these measurements datapoints were collected at 100 different frequencies (labeled by the vector ac_frequency.
# 
# We should plot all 150 of these curves. This would take a lot of time with excel. Use the code provided below to complete these graphs.
# 
# 1. Start by commenting on the existing code (5 points)
# 2. Create a function that plots all the data for the different samples (10 points)
#    1. The plot should be frequency vs. dielectric constant
#    2. The plot should be a semilogx as the frequency is generally represented in a log scale. Test your google skills to learn how to do this using matplotlib
#    3. Add the x and y labels to the graph
#    4. Make sure you add the color from the color scale we defined and make sure to add labels to each of the plots so the legend generates correctly
# 

# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(12,4))

color = np.flipud(plt.cm.viridis(np.linspace(0,1,50)))

# Add your code for your loop here. 


chartBox = axs[2].get_position()
axs[2].set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
axs[2].legend(loc='upper right', bbox_to_anchor=(2.5, 1), ncol=3)


# #### Task: Fitting the Data (10 Points)
# 
# Now that you can see the plots you want to conduct a linear fit on all of these curves to extract the zero-frequency permittivity (the intercept) at each dc voltage.
# 
# Make a nested for loop to linear fit all of the data. Make sure to save the fit results.
# 
# Syntax: to fit the data use `np.polyfit(x, y, 1)`. Where 1 represents the order of the polynomial (in this case linear)
# 
# The first index `[0]` is the slope. The second index `[1]` is the intercept.
# 
# Hints:
# 
# - It is helpful to preallocate an array to store the results
#   `fit_results = np.zeros()` # 3 samples, 50 curves, 2 parameter
# 

# In[4]:


# Your code goes here


# #### Task 4: Plotting the results (10 points)
# 
# We did not talk about this in class yet, but you can make a very simple line plot using `matplotlib`. We already imported the package `matplotlib.pyplot as plt` now you can use this package to plot your results.
# 
# 1. Use the `subplot` module to create 3 graphs (2 points)
# 1. Use a loop that loops around the axis and plots the DC voltage vs the calculated intercept (2 points)
# 1. Add labels to the graph using `set_xlabel`, `set_ylabel`, `set_title` (2 points)
# 1. Make sure the title of the graph indicates the sample number (2 points)
# 1. You can use `plt.tight_layout()` to make your graph loop pretty (2 points)
# 

# In[5]:


# Your code goes here


# #### Task 5: Fitting regions of a plot (20 points)
# 
# If you look at these plots a few of them have two linear regimes. This means that there are two different mechanisms of dielectric response.
# 
# You should conduct linear fits within these regions.
# 
# Hint: to do this you want to set the cutoff values for where you conduct the linear fit.
# 
# This can be done by finding the index you within your selected range using:
# `idx = (data['dc_voltage_vector']>0)*(data['dc_voltage_vector']<500)`
# 
# Then you can select those values by indexing the vector:
# `data['dc_voltage_vector'][idx]`
# 
# For this problem, I want to challenge you to think programmatically. Here are some high-level hints.
# 
# 1. You want to define the ranges where the response is linear. This could be automated but it is acceptable to do it manually here (2 points)
# 2. You want to build a figure that contains 3 graphs (2 points)
# 3. You want to conduct the fits in each of the regions (2 points)
# 4. You want to plot the graphs and the fit results. To see the fit results it is recommended that you use a different color so that it can be seen (2 points)
# 5. Add labels and titles to the graph (2 points)
# 6. Add code that will put a label with the fit results on the graph (2 points)
# 7. Save the figure to a file. You can look up how to do this within `matplotlib` (2 points)
# 8. Make sure to comment your code (4 points)
# 
# Completed assignment (4 points)
# 

# In[6]:


# Your code goes here

