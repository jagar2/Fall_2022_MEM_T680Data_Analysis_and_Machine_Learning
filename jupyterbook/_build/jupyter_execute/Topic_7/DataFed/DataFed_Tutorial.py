#!/usr/bin/env python
# coding: utf-8

# # DataFed Tutorial

# ## Getting Started with DataFed
# 
# In this notebook we will be going over basic concepts such as `contexts`, `projects`, understanding how DataFed responds to function calls, etc.
# 
# To get started, we need to import only the `API` class from `CommandLab` in the datafed package.
# 

# In[1]:


from datafed.CommandLib import API


# Next, we need to instantiate the `API` class as:
# 

# In[2]:


df_api = API()


# ### First DataFed function
# 
# Let's try out the DataFed API by simply asking DataFed for a list of all projects that we are part of using the `projectList()` function:
# 

# In[3]:


pl_resp = df_api.projectList()
print(pl_resp)


# ### DataFed Messages
# 
# DataFed responds using `Google Protocol Buffer` or `protobuf` messages
# 
# Let's take a closer look at this response:
# 

# In[4]:


print(type(pl_resp), len(pl_resp))


# As you can see, the reply is a tuple containing two objects, namely the protobuf message reply itself, `[0]` and the type of reply received, `ListingReply` at `[1]`. We can confirm this by checking the response type:
# 

# In[5]:


type(pl_resp[0])


# ### Contexts
# 
# DataFed allows us to work within multiple different "data spaces" – such as our own `Personal Data`, and those of our `Project`s. Let's find out what `context` DataFed automatically put us into using `getContext()`:
# 

# In[6]:


print(df_api.getContext())


# By default, DataFed sets our working space or `context` to our own `Personal Data` (`u/username`).
# 

# ### Specifying contexts:
# 
# If we want to take a look at the `root` collection of the Training Project, we need to specify that using the `context` keyword argument:
# 

# In[7]:


print(df_api.collectionView("root", context="p/trn001"))


# Here's what we get when we give the same `collectionView` request without the project context:
# 

# In[8]:


print(df_api.collectionView("root"))


# ### Subscripting and Iterating messages
# 
# Let us take a look at the contents of the Project (its `root` Collection) using `collectionItemsList()`
# 

# In[9]:


ls_resp = df_api.collectionItemsList("root", context="p/trn001")
print(ls_resp)


# Much like the `projectList()`, we get a `ListingReply` in this case as well
# 

# ### Subscripting
# 
# The listing reply `item` behaves similarly to a python list in terms of subscriptability. We can use indexing:
# 

# In[10]:


ls_resp[0].item[-2].title


# ### Iterating
# 
# These messages also mirror python lists in their iterability.
# 
# We can iterate through the items in this listing and use the subscripting capability to only extract the `id` and `alias` fields of each of the collections
# 

# In[11]:


for record in ls_resp[0].item:
    print(record.id, "\t", record.alias)


# ## Aliases and IDs
# 
# Let's try taking a closer look at the `PROJSHARE` collection using its `alias`:
# 

# In[ ]:


df_api.collectionView("projshare")


# The above request failed because we asked DataFed to look for a Collection with alias: `projshare` without specifying the `context`. Naturally, DataFed assumed that we meant our own `Personal Data` rather than the training `Project`.
# 

# If we want to address an item by its `alias`, we need to be careful about its `context` since:
# 
# **An `alias` is unique only within a given `context`** such as `Personal Data` or a `Project`
# 

# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Correct the above function call to view `projshare` collection: </span>
# 

# In[12]:


# Your Code Goes Here


# In[13]:


df_api.collectionView("projshare", context="p/trn001")


# Alternatively, we can view the correct collection by referring to it using its `id`:
# 

# In[14]:


df_api.collectionView("c/34559108")


# The above command worked even though we did not specify a `context` because:
# 
# **`ID`s are unique across DataFed and do not need a `context`**
# 

# ### Setting Context:
# 
# Having to specify the context for every function call can be tiring if we are sure we are working within a single context.
# 
# We can set the context via the `setContext()` function:
# 

# In[15]:


df_api.setContext("p/trn001")


# ```{note}
# ``setContext()`` is valid within the scope of a single python session. You would need to call the function each time you instantiate the DataFed ``CommandLib.API`` class. E.g. - at the top of every notebook
# ```
# 

# Let's attempt to view the `projshare` Collection via its `alias` **without** specifying the `context` keyword argument:
# 

# In[16]:


df_api.collectionView("projshare")


# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Using the DataFed API's `collectionView()`, extract the create time (`ct`) of your own personal collection within the training project or `projshare`. <br><br> <b> Bonus: </b> Consider using the built-in `df_api.timestampToStr()` function or `datetime.datetime.fromtimestamp` from the `datetime` package to convert the unix formatted time to a human readable string </span>
# 

# In[17]:


# Your Code Goes Here


# In[18]:


personal_collection_id = "c/34559108"
create_time = df_api.collectionView(personal_collection_id)[0].coll[0].ct

# Bonus:
df_api.timestampToStr(create_time)


# ## Data Records
# 
# We will be going over how to create, add metadata and other contextual information, edit, establish relationships between `Data Records` - the fundamental unit in DataFed
# 

# ### Specifying context:
# 
# Since we want to work within the `context` of the Training Project:
# 

# In[19]:


df_api.setContext("p/trn001")


# To begin with, you will be working within your own private collection whose `alias` is the same as your DataFed username.
# 

# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Enter your username into the `parent_collection` variable </span>
# 

# In[20]:


# Your Code Goes Here


# In[21]:


parent_collection = "jca318"


# ### Creating Data Records:
# 
# Data Records can hold a whole lot of contextual information about the raw data.
# 
# - Ideally, we would get this metadata from the headers of the raw data file or some other log file that was generated along with the raw data.
# 

# ```{note}
# DataFed expects scientific metadata to be specified **like** a python dictionary.
# ```
# 

# For now, let's set up some dummy metadata:
# 

# In[22]:


parameters = {
    "a": 4,
    "b": [1, 2, -4, 7.123],
    "c": "Something important",
    "d": {"x": 14, "y": -19},  # Can use nested dictionaries
}


# DataFed currently takes metadata as a JSON file, this can be achieved using `json.dumps`
# 

# In[23]:


import json

json.dumps(parameters)


# We use the `dataCreate()` function to make our new record, and the `json.dumps()` function to format the python dictionary to JSON:
# 

# In[24]:


dc_resp = df_api.dataCreate(
    "my important data",
    metadata=json.dumps(parameters),
    parent_id=parent_collection,
    # The parent collection, whose alias is your username
)
dc_resp


# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Extract the `ID` of the data record from the message returned from `dataCreate()` for future use: </span>
# 

# In[25]:


# Your Code Goes Here


# In[26]:


record_id = dc_resp[0].data[0].id
print(record_id)


# Data Records and the information in them are not static and can always be modified at any time
# 

# ### Updating Data Records
# 
# Let's add some additional metadata and change the title of our record:
# 

# In[27]:


du_resp = df_api.dataUpdate(
    record_id,
    title="Some new title for the data",
    metadata=json.dumps({"appended_metadata": True}),
)
print(du_resp)


# ### Viewing Data Records
# 
# We can get full information about a data record including the complete metadata via the `dataView()` function. Let us use this function to verify that the changes have been incorporated:
# 

# In[28]:


dv_resp = df_api.dataView(record_id)
print(dv_resp)


# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Try isolating the updated metadata and converting it to a python dictionary. </span>
# 
# Hint - `json.loads()` is the opposite of `json.dumps()`
# 

# In[29]:


# Your Code Goes Here


# In[30]:


metadata = json.loads(dv_resp[0].data[0].metadata)
print(metadata)


# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Now try to **replace** the metadata. <br><br>Hint: look at the `metadata_set` keyword argument in the docstrings. </span>
# 
# You can make the new metadata `new_metadata = {"key": "value", "E": "mc^2"}`
# 

# ```{tip}
# With the cursor just past the starting parenthesis of ``dataUpdate(``, simultaneously press the ``Shift`` and ``Tab`` keys once, twice, or four times to view more of the documentation about the function.
# ```
# 

# In[31]:


# Your Code Goes Here


# In[32]:


new_metadata = {"key": "value", "E": "mc^2"}
du_resp = df_api.dataUpdate(
    record_id, metadata=json.dumps(new_metadata), metadata_set=True
)
dv_resp = df_api.dataView(record_id)
print(json.loads(dv_resp[0].data[0].metadata))


# ```{note}
# DataFed currently does not support version control of metadata. If you wanted to implement this it could be really valuable
# ```
# 

# ## Provenance
# 
# Along with in-depth, detailed scientific metadata describing each data record, DataFed also provides a very handy tool for tracking data provenance, i.e. recording the relationships between Data Records which can be used to track the history, lineage, and origins of a data object.
# 

# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Create a new record meant to hold some processed version of the first data record. <br> **Caution**: Make sure to create it in the correct Collection.</span>
# 

# In[33]:


# Your Code Goes Here


# In[34]:


new_params = {"hello": "world", "counting": [1, 2, 3, 4]}

dc2_resp = df_api.dataCreate(
    "Subsequent Record", metadata=json.dumps(new_params), parent_id=parent_collection
)

clean_rec_id = dc2_resp[0].data[0].id
print(clean_rec_id)


# ### Specifying Relationships
# 
# Now that we have two records, we can specify the second record's relationship to the first by adding a **dependency** via the `deps_add` keyword argument of the `dataUpdate()` function.
# 

# ```{note}
# Dependencies must be specified as a ``list`` of relationships. Each relationship is expressed as a ``list`` where the first item is a dependency type (a string) and the second is the data record (also a string)
# ```
# 

# DataFed currently supports three relationship types:
# 
# - `der` - Is derived from
# - `comp` - Is comprised of
# - `ver` - Is new version of
# 

# In[35]:


dep_resp = df_api.dataUpdate(clean_rec_id, deps_add=[["der", record_id]])
print(dep_resp)


# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green"> Take a look at the records on the DataFed Web Portal in order to see a graphical representation of the data provenance. </span>
# 

# ### <span style="color:green"> Exercise: </span>
# 
# <span style="color:green">1. Create a new data record to hold a figure in your journal article. <br>2. Extract the record ID. <br>3. Now establish a provenance link between this figure record and the processed data record we just created. You may try out a different dependency type if you like. <br>4. Take a look at the DataFed web portal to see the update to the Provenance of the records</span>
# 

# In[36]:


# Your Code Goes Here


# In[37]:


# 1
reply = df_api.dataCreate("Figure 1", parent_id=parent_collection)
# 2
fig_id = reply[0].data[0].id
# 3
provenance_link = df_api.dataUpdate(fig_id, deps_add=[("comp", clean_rec_id)])
print(provenance_link[0].data[0])


# ## Transferring Data in DataFed
# 

# In[38]:


# imports necessary packages

import json
import os
import time
from datafed.CommandLib import API


# Instantiate the DataFed API and set `context` to the Training project:
# 

# In[39]:


df_api = API()
df_api.setContext("p/trn001")


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Enter your username to work within your personal Collection. </span>
# 

# In[40]:


# Your Code Goes Here


# In[41]:


parent_collection = "jca318"  # your username here


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Copy over the ID for the record you created previously  </span>
# 

# In[42]:


# Your Code Goes Here


# In[43]:


record_id = "d/412681057"


# ## Uploading raw data
# 
# We will learn how to upload data to the record we just created. For demonstration purposes, we will just create a simple text file and use this as the raw data for the Data Record
# 

# ```{note}
# DataFed does not impose any restrictions on the file extension / format for the raw data
# ```
# 

# In[44]:


datapath = './datapath'

# This just writes some text
with open(datapath + "/raw_data.txt", mode="w") as file_handle:
    file_handle.write("This is some data")


# ```{note}
# Ensure that your Globus endpoint is active and that your files are located in a directory that is visible to the Globus Endpoint
# ```
# 

# Uploading data files to DataFed is done using the `dataPut` command.
# 

# In[45]:


put_resp = df_api.dataPut(
    record_id,
    datapath + "/raw_data.txt",
    wait=True,  # Waits until transfer completes.
)
print(put_resp)


# We get two components in the response:
# 
# - Information about the Data Record, data was uploaded to
# - Information about the data transfer `task` - more on this later
# 

# The `dataPut()` method **initiates a Globus transfer** on our behalf from the machine **wherever** the file was present to wherever the default data repository is located. In this case, the file was in our local file system and on the same machine where we are executing the command.
# 

# ```{note}
# The above data file was specified by its relative local path, so DataFed used our pre-configured default Globus endpoint to find the data file. As long as we have the id for any *active* Globus endpoint that we have authenticated access to, we can transfer data from that endpoint with its full absolute file path – even if the file system is not attached ot the local machine. Look for more information on this in later examples.
# ```
# 

# Let's view the data record now that we've uploaded our data. Pay attention to the `ext` and `source` fields which should now populated:
# 

# In[46]:


dv_resp = df_api.dataView(record_id)
print(dv_resp)


# ## Downloading raw data
# 
# DataFed is also capable of getting data stored in a DataFed repository and placing it in the local or other Globus-visible filesystem via the `dataGet()` function.
# 

# Let us download the content in the data record we have been working on so far for demonstration purposes
# 

# In[47]:


get_resp = df_api.dataGet(
    record_id,
    datapath,  # directory where data should be downloaded
    orig_fname=False,  # do not name file by its original name
    wait=True,  # Wait until Globus transfer completes
)
print(get_resp)


# In the response we only get back information about the data transfer `task` - more on this shortly
# 
# `dataGet()` reveals its capability to **download multiple data records or even Collections.**
# 
# Let's confirm that the data has been downloaded successfully:
# 

# In[48]:


os.listdir(datapath)


# In[49]:


expected_file_name = os.path.join(datapath, record_id.split("d/")[-1]) + ".txt"
print("Does a file with this name: " + expected_file_name + " exist?")
print(os.path.exists(expected_file_name))


# ## Tasks
# 

# ```{note}
# A DataFed task may itself contain / be responsible for several Globus file transfers, potentially from / to multiple locations
# ```
# 

# DataFed makes it possible to check on the status of transfer tasks in an easy and programmatic manner.
# 

# Before we learn more about tasks, first lets try to get the `id` of the task in `get_resp` from the recent `dataGet()` function call:
# 

# In[50]:


task_id = get_resp[0].task[0].id
print(task_id)


# ### Viewing Tasks
# 
# We can get more information about a given transfer via the `taskView()` function:
# 

# In[51]:


task_resp = df_api.taskView(task_id)
print(task_resp)


# We get a new kind of message - a `TaskDataReply`.
# Key fields to keep an eye on:
# 
# - `status`
# - `msg`
# - `source`
# - `dest`
# 

# If we are interested in monitoring tasks, triggering activities or subsequent steps of workflows based on transfers, we would need to know how to get the `status` property from the `TaskDataReply`:
# 

# In[52]:


task_resp[0].task[0].status


# Even though the message above says `TS_SUCCEEDED`, we see that this task status codes to the integer `3`.
# 

# ```{note}
# Cheat sheet for interpreting task statuses: 
# 
# ``2``: in progress
# 
# ``3``: complete
# 
# anything else - problem
# ```
# 

# ### Listing Tasks
# 
# We can request a listing of all our recently initiated tasks:
# 

# In[53]:


df_api.taskList()


# The output of this listing would be very helpful for the exercise below
# 

# ### <span style="color:green">Example scenario - Simulations </span>
# 
# <span style="color:green"> Let's say that we want to run a series of simulations where one or more parameters are varied and each simulation is run with a unique set of parameters. Let's also assume that our eventual goal is to build a surrogate model for the computationally expensive simulation using machine learning. So, we **want to capture the metadata and data associated with the series of simulations** to train the machine learning model later on.</span>
# 
# <span style="color:green"> We have set up skeleton functions and code snippets to help you mimic the data management for such a simulation. We would like you to take what you have learnt so far and fill in the blanks </span>
# 

# #### Fake simulation
# 
# Here, we have simulated a computationally "expensive" simulation that simply sleeps for a few seconds.
# 

# In[54]:


import time


def expensive_simulation():
    time.sleep(5)
    # Yes, this simulation is deterministic and always results in the same result:
    path_to_results = datapath + "/408461737.txt"
    # The simulation uses the same combination of parameters
    metadata = {"a": 1, "b": 2, "c": 3.14}
    return path_to_results, metadata


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Define a function that: <br> 1. creates a new Data Record with the provided metadata (as a dictionary) and other details, <br> 2. extracts the record id, <br> 3. puts the raw data into the record, <br> 4. extracts <br> 5. Returns the task ID. <br><br> Feel free to print any messages that may help you track things. </span>
# 
# ```{note}
# Pay attention to the ``wait`` keyword argument when putting the raw data into record
# ```
# 

# In[55]:


# Your Code Goes Here


# In[56]:


def capture_data(
    simulation_index,  # integer - counter to signify the Nth simulation in the series
    metadata,  # dictionary - combination of parameters used for this simulation
    raw_data_path,  # string - Path to the raw data file that needs to be put into the receord
    parent_collection=parent_collection,  # string - Collection to create this Data Record into
):

    # 1. Create a new Data Record with the metadata and use the simulation index to provide a unique title
    rec_resp = df_api.dataCreate(
        "Simulation_" + str(simulation_index),
        metadata=json.dumps(metadata),
        parent_id=parent_collection,
    )

    # 2. Extract the record ID from the response
    this_rec_id = rec_resp[0].data[0].id

    # 3. Put the raw data into this record:
    put_resp = df_api.dataPut(this_rec_id, raw_data_path, wait=False)

    # 4. Extract the ID for the data transfer task
    task_id = put_resp[0].task.id

    # 5. Return the task ID
    return task_id


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Try out this function to make sure it works. See what it does on the **DataFed web portal**. </span>
# 

# In[57]:


path_to_results, metadata = expensive_simulation()

task_id = capture_data(14, metadata, path_to_results)
task_id


# We might want a simple function to monitor the status of all the data upload tasks. Define a function that accepts a list of task IDs and returns their status after looking them up on DataFed
# 

# In[58]:


def check_xfer_status(task_ids):

    # put singular task ID into a list
    if isinstance(task_ids, str):
        task_ids = [task_ids]

    # Create a list to hold the statuses of each of the tasks
    statuses = list()

    # Iterate over each of the task IDs
    for this_task_id in task_ids:

        # For each task ID, get detailed information about it
        task_resp = df_api.taskView(this_task_id)

        # Extract the task status from the detailed information
        this_status = task_resp[0].task[0].status

        # Append this status to the list of statuses
        statuses.append(this_status)

    # Return the list of statuses
    return statuses


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Try out your function using the IDs of the recent `dataPut()` and `dataGet()` functions. </span>
# 

# In[59]:


check_xfer_status(task_id)


# ### Run the series of simulations:
# 
# Use the three functions defined above to mimic the process of exploring a parameter space using simulations, where for each iteration, we:
# 
# 1. Run a simulation
# 2. Capture the data + metadata into DataFed
# 3. Monitor the data upload tasks.
# 

# In[60]:


xfer_tasks = list()
for ind in range(3):
    print("Starting simulation #{}".format(ind))
    # Run the simulation.
    path_to_results, metadata = expensive_simulation()
    # Capture the data and metadata into DataFed
    task_id = capture_data(ind, metadata, path_to_results)
    # Append the task ID for this data upload into xfer_tasks
    xfer_tasks.append(task_id)
    # Print out the status of the data transfers
    print("Transfer status: {}".format(check_xfer_status(xfer_tasks)))
    print("")

print("Simulations complete! Waiting for uploads to complete\n")

while True:
    time.sleep(5)
    statuses = check_xfer_status(xfer_tasks)
    print("Transfer status: {}".format(statuses))
    if all([this == 3 for this in statuses]):
        break

print("\nFinished uploading all data!")


# ```{note}
# It is recommended to perform data orchestration (especially large data movement - upload / download) operations outside the scope of heavy / parallel computation operations in order to avoid wasting precious wall time on compute clusters
# ```
# 

# ## Collections and Queries in DataFed
# 

# In this notebook, we will be going over creating Collections, viewing contained items, organizing Collections, downloading Collections, and searching for data
# 

# Import necessary libraries
# 

# In[61]:


import os
import json
from datafed.CommandLib import API


# Instantiate the DataFed API and set `context` to the Training project
# 

# In[62]:


df_api = API()
df_api.setContext("p/trn001")


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Reset this variable to your username or Globus ID so that you work within your own collection by default </span>
# 

# In[63]:


# Your Code Goes Here


# In[64]:


parent_collection = "jca318"  # Name of this user


# ## Example use case:
# 
# Let us assume that we are working on a machine learning problem aimed at putting together training data for a machine learning model. For illustration purposes, we will assume that we aim to train a classifier for classifying animals
# 

# #### Create Collection
# 
# First, let us create a collection to hold all our data.
# 
# We will be using the `collectionCreate()` function:
# 

# In[65]:


coll_resp = df_api.collectionCreate(
    "Image classification training data", parent_id=parent_collection
)
print(coll_resp)


# In this case we got back a `CollDataReply` object. This is somewhat similar to what you get from `dataCreate()` we just saw.
# 

# Now, let's Extract the `id` of this newly created collection:
# 

# In[66]:


train_coll_id = coll_resp[0].coll[0].id
print(train_coll_id)


# #### Populate with training data
# 
# Now that we have a place to put the training data, let us populate this collection with examples of animals
# 

# ##### Define a function to generate (fake) training data:
# 
# We need a function to:
# 
# - Create a Data Record
# - Put data into this Data Record
# 

# For simplicity we will use some dummy data from a public Globus Endpoint This information has been filled in for you via the `raw_data_path` variable.
# 

# We have a skeleton function prepared for you along with comments to guide you
# 

# In[67]:


import random

def generate_animal_data(is_dog=True):
    this_animal = "cat"
    if is_dog:
        this_animal = "dog"
    # Ideally, we would have more sensible titles such as "Doberman", "Poodle", etc. instead of "Dog_234"
    # To mimic a real-life scenario, we append a number to the animal type to denote
    # the N-th example of a cat or dog. In this case, we use a random integer.
    title = this_animal + "_" + str(random.randint(1, 1000))
    # Create the record here:
    rec_resp = df_api.dataCreate(
        title, metadata=json.dumps({"animal": this_animal}), parent_id=train_coll_id
    )

    # Extract the ID of the Record:
    this_rec_id = rec_resp[0].data[0].id

    # path to the file containing the (dummy) raw data
    raw_data_path = datapath + "/raw_data.txt"

    # Put the raw data into the record you just created:
    put_resp = df_api.dataPut(this_rec_id, raw_data_path)

    # Only return the ID of the Data Record you created:
    return this_rec_id


# ##### Generate 5 examples of cats and dogs:
# 

# In[68]:


cat_records = list()
dog_records = list()
for _ in range(5):
    dog_records.append(generate_animal_data(is_dog=True))
    time.sleep(0.1)
for _ in range(5):
    cat_records.append(generate_animal_data(is_dog=False))
    time.sleep(0.1)


# In[69]:


print(cat_records)


# In[70]:


print(dog_records)


# ## Listing items in a Collection:
# 
# Let us take a look at the training data we have assembled so far using the `colectionItemsList()` function:
# 

# In[71]:


coll_list_resp = df_api.collectionItemsList(train_coll_id, offset=5)
print(coll_list_resp)


# ```{note}
# If we had several dozens, hundreds, or even thousands of items in a Collection, we would need to call ``collectionItemsList()`` multiple times by stepping up the ``offset`` keyword argument each time to get the next “page” of results.
# ```
# 

# ```{admonition} Discussion
# Let's say that we are only interested in finding records that have cats in this (potentially) large collection of training data. How do we go about doing that?
# ```
# 

# ## Data Query / Search
# 
# Use the DataFed web interface to:
# 
# - Search for cats
# - Specifically in your collection
# - Save the query
# 

# ![](figs/saving_a_search.png)
# 

# ```{note}
# Saved queries can be found in the bottom of the navigation (left) pane under ``Project Data`` and ``Saved Queries``
# ```
# 

# ## Finding Saved Queries:
# 
# We can list all saved queries via `queryList()`:
# 

# In[72]:


ql_resp = df_api.queryList()
print(ql_resp)


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Get the `id` of the desired query out of the response: </span>
# 

# In[73]:


# Your Code Goes Here


# In[74]:


id_ = [i.title for i in ql_resp[0].item].index("cat2")
query_id = ql_resp[0].item[id_].id
print(query_id)


# ### View the saved query
# 
# Use the `queryView()` function:
# 

# In[75]:


df_api.queryView(query_id)


# ### Run a saved query
# 
# Use the `queryExec()` function:
# 

# In[76]:


query_resp = df_api.queryExec(query_id)
print(query_resp)


# Yet again, we get back the `ListingReply` message.
# 

# Now let us extract just the `id`s from each of the items in the message:
# 

# In[77]:


cat_rec_ids = list()
for record in query_resp[0].item:
    cat_rec_ids.append(record.id)

# one could also use list comprehensions to get the answer in one line:
# cat_rec_ids = [record.id for record in query_resp[0].item]
print(cat_rec_ids)


# We already have the ground truth in `cat_records`. Is this the same as what we got from the query?
# 

# In[78]:


# Note, you might get a false response if you have run this script more than once
print(set(cat_rec_ids) == set(cat_records))


# ## Separating cats from dogs
# 
# Our goal now is to gather all cat Data Records into a dedicated Collection
# 

# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Create a new collection to hold the Cats record </span>
# 

# In[79]:


# Your Code Goes Here


# In[80]:


coll_resp = df_api.collectionCreate("Cats", parent_id=train_coll_id)


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Extract the `id` for this Collection: </span>
# 

# In[81]:


# Your Code Goes Here


# In[82]:


cat_coll_id = coll_resp[0].coll[0].id
print(cat_coll_id)


# ## Adding Items to Collection
# 
# Now let us add only the cat Data Records into this new collection using the `collectionItemsUpdate()` function:
# 

# In[176]:


cup_resp = df_api.collectionItemsUpdate(cat_coll_id, add_ids=cat_rec_ids)
print(cup_resp)


# Unlike most DataFed functions, this function doesn't really return much
# 

# Now, let us view the contents of the Cats Collection to make sure that all Cat Data Records are present in this Collection.
# 

# Just to keep the output clean and short, we will only extract the ID and title of the items
# 

# In[177]:


ls_resp = df_api.collectionItemsList(cat_coll_id)
# Iterating through the items in the Collection and only extracting a few items:
for obj in ls_resp[0].item:
    print(obj.id, obj.title)


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> View the contents of the main training data Collection. <br> You may use the snippet above if you like and modify it accordingly </span>
# 

# In[83]:


# Your Code Goes Here


# In[179]:


ls_resp = df_api.collectionItemsList(train_coll_id)
# Iterating through the items in the Collection and only extracting a few items:
for obj in ls_resp[0].item:
    print(obj.id, obj.title)


# ```{note}
# Data Records can exist in **multiple** Collections just like video or songs can exist on multiple playlists
# ```
# 

# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Remove the cat Data Records from the training data collection. They already exist in the "Cats" Collection. <br> **Hint**: The function call is very similar to the function call for adding cats to the "Cats" collection </span>
# 

# In[84]:


# Your Code Goes Here


# In[181]:


cup_resp = df_api.collectionItemsUpdate(train_coll_id, rem_ids=cat_rec_ids)
print(cup_resp)


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> View the contents of the training data Collection. <br> You may reuse a code snippet from an earlier cell. <br> Do you see the individual cat Data Records in this collection? </span>
# 

# In[85]:


# Your Code Goes Here


# In[182]:


ls_resp = df_api.collectionItemsList(train_coll_id)
# Iterating through the items in the Collection and only extracting a few items:
for obj in ls_resp[0].item:
    print(obj.id, obj.title)


# ## Search or Organize?
# 
# If you could always search for your data, what is the benefit to organizing them into collections?
# 

# If you have a collection it is very easy to download the entire collection!
# 

# ### Download entire Collection
# 

# ```{note}
# DataFed can download arbitrarily large number of Records regardless of the physical locations of the DataFed repositories containing the data.
# ```
# 

# Let us first make sure we don't already have a directory with the desired name:
# 

# In[86]:


dest_dir = datapath + "/cat_data"

if os.path.exists(dest_dir):
    import shutil

    shutil.rmtree(dest_dir)


# ### <span style="color:green"> Exercise </span>
# 
# <span style="color:green"> Download the entire Cat Collection with a single DataFed function call. <br> **Hint:** You may want to look at a 
# 

# In[87]:


# Your Code Goes Here


# In[186]:


df_api.dataGet(cat_coll_id, datapath + "/cat_data", wait=True)


# Let's verify that we downloaded the data:
# 

# In[187]:


os.listdir(dest_dir)

