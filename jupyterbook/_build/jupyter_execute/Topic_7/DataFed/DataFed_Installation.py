#!/usr/bin/env python
# coding: utf-8

# # Installing DataFed

# ## Getting Started
# 
# Please follow this guide to get started with DataFed
# 

# ### Get a Globus account
# 
# Follow only step 1 of instructions [here](https://docs.globus.org/how-to/get-started/) to get a Globus account.
# 

# ### Get a Globus ID
# 
# Ensure that your `globus ID` is linked with your institutional ID in your globus account:
# 
# 1. Log into [globus.org](https://www.globus.org/)
# 

# 2. Click on `Account` on the left hand pane
# 

# 3. Select the `Identities` tab in the window that opens up
# 
# You should see (at least these) two identities:
# 
# - One from your home institution (that is listed as primary with a crown)
# 
# - Globus ID (your_username@globusid.org)
# 
# - If you do not see the `Globus ID`, click on `Link another identity`. Select `Globus ID` and link this ID.
# 

# ### Register at DataFed
# 
# 1. Once you have a Globus ID, visit the [DataFed web portal](https://datafed.ornl.gov/).
# 

# 2. Click on the `Log in / Register` button on the top right of the page.
# 

# 3. Follow the steps to register yourself with DataFed.
# 

# 4. Though you can log into the DataFed web portal with your institution’s credentials, you will need the username and password you set up during your registration for scripting.
# 

# ```{note}
# Your institutional credentials are not the same as your DataFed credentials. The latter is only required for using DataFed via python / CLI.
# ```
# 

# ### Get data allocations
# 
# As the name suggests, a data allocation is just the data storage space that users and projects can use to store and share data of their own. Though you can start to use DataFed at this point to view and get publicly shared data, it would not be possible to create or manipulate data of your own unless you have a data allocation in a DataFed data repository.
# 
# You can request a small allocation from Prof. Agar. If you would like to use DataFed for your research please email [Prof. Agar](jca92@drexel.edu)
# 

# ### Install a Globus Endpoint
# 
# You will need a [Globus endpoint](https://docs.cades.ornl.gov/#data-transfer-storage/globus-endpoints/) on every machine where you intend to download / upload data.
# 
# - Most computing facilities already have a Globus endpoint
# 

# ### Using Personal Computers and Workstations
# 
# 1. Install [Globus Personal Connect](https://www.globus.org/globus-connect-personal)
# 

# 2. When conducting the install make note of the endpoint name
# 

# 3. Log into Globus: Drexel does not have an organizational login, you may choose to either Sign in with Google or Sign in with ORCiD iD.
# 
# ![](figs/globus_google.png)
# 

# 4. Check your managed endpoints to make sure your endpoint is visible.
#    - You want to copy the UUID - this is the ID to the endpoint
# 
# ![](figs/Id.png)
# 

# 5. Installing DataFed
# 
#    `pip install datafed`
# 
# Note, if you used the `requirements.txt` file this was already installed. You can just verify that it was installed by running `pip install command`
# 

# 6. Ensure the bin Directory is in the Path
# 
# If you do not see an error when you type `datafed` in your terminal, you may skip this step.
# 
# If you encounter errors stating that `datafed was an unknown command`, you would need to add DataFed to your path.
# 
# - First, you would need to find where datafed was installed. For example, on some compute clusters, datafed was installed into directories such as `~/.local/MACHINE_NAME/PREFIXES-anaconda-SUFFIXES/bin`
# 
# - Next, add DataFed to the `PATH` variable.
# 
# Here is an [external guide](https://www.makeuseof.com/python-windows-path/) on adding Python to the `PATH` on Windows machines
# 

# 7. Basic Configuration
# 
# - Type the following command into shell:
# 
# `datafed setup`
# 
# It will prompt you for your username and password.
# 

# - Enter the credentials you set up when registering for an account on DataFed (not your institutional credentials you use to log into the machine)
# 

# - Add the Globus endpoint specific to this machine / file-system as the default endpoint via:
# 
# `datafed ep default set endpoint_name_here`
# 
# ```{note}
# If you are using Globus Connect Personal, visit the Settings or Preferences of the application to inspect which folders Globus has write access to. Consider adding or removing directories to suit your needs.
# ```
# 

# ## Checking DataFed Installation and Configuration
# 

# In[1]:


# Import packages

import os
import getpass
import subprocess
from platform import platform
import sys

try:
    datapath = os.mkdir("./datapath")
except:
    datapath = "./datapath"


# ### 0. Machine information:
# 

# Python version:
# 

# In[2]:


sys.version_info


# In[3]:


platform()


# ### 1. Verify DataFed installation:
# 

# In[4]:


try:

    # This package is not part of anaconda and may need to be installed.
    from datafed.CommandLib import API

except ImportError:
    print("datafed not found. Installing from pip.")
    subprocess.call([sys.executable, "-m", "pip", "install", "datafed"])
    from datafed.CommandLib import API

from datafed import version as df_ver

if not df_ver.startswith("1.4"):
    print("Attempting to update DataFed.")
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "datafed"])
    print(
        "Please restart the python kernel or upgrade manually to V 1.1.0:1 if you are repeatedly seeing this message via"
        "\n\tpip install --upgrade datafed"
    )
else:
    df_api = API()
    print("Success! You have DataFed: " + df_ver)


# ### 2. Verify user authentication:
# 

# In[5]:


if df_api.getAuthUser():
    print(
        "Success! You have been authenticated into DataFed as: " + df_api.getAuthUser()
    )
else:
    print("You have not authenticated into DataFed Client")
    print(
        'Please follow instructions in the "Basic Configuration" section in the link below to authenticate yourself:'
    )
    print("https://ornl.github.io/DataFed/user/client/install.html#basic-configuration")


# ### 3. Ensure Globus Endpoint is set:
# 

# In[6]:


if not df_api.endpointDefaultGet():
    print("Please follow instructions in the link below to find your Globus Endpoint:")
    print(
        "https://ornl.github.io/DataFed/system/getting_started.html#install-identify-globus-endpoint"
    )
    endpoint = input(
        "\nPlease enter either the Endpoint UUID or Legacy Name for your Globus Endpoint: "
    )
    df_api.endpointDefaultSet(endpoint)

print("Your default Globus Endpoint in DataFed is:\n" + df_api.endpointDefaultGet())


# ### 4. Test Globus Endpoint:
# 
# This will make sure you have write access to the folder
# 

# In[7]:


# This is a dataGet Command
dget_resp = df_api.dataGet("d/35437908", os.path.abspath(datapath), wait=True)
dget_resp


# You can see that a file was downloaded.
# 

# In[8]:


if dget_resp[0].task[0].status == 3:
    print("Success! Downloaded a test file to your location. Removing the file now")
    os.remove(datapath + "/35437908.md5sum")
else:
    if dget_resp[0].task[0].msg == "globus connect offline":
        print(
            "You need to activate your Globus Endpoint and/or ensure Globus Connect Personal is running.\n"
            "Please visit https://globus.org to activate your Endpoint"
        )
    elif dget_resp[0].task[0].msg == "permission denied":
        print(
            "Globus does not have write access to this directory. \n"
            "If you are using Globus Connect Personal, ensure that this notebook runs within"
            "one of the directories where Globus has write access. You may consider moving this"
            "notebook to a valid directory or add this directory to the Globus Connect Personal settings"
        )
    else:
        NotImplementedError(
            "Get in touch with us or consider looking online to find a solution to this problem:\n"
            + dget_resp[0].task[0].msg
        )


# ### (Optional) for Windows - Test for Admin privileges
# 
# Admin privileges may be necessary for some operations. On Windows when you start your Anaconda Console you can `right-click` and select `run as administrator`
# 

# In[9]:


import ctypes, os

try:
    is_admin = os.getuid() == 0
except AttributeError:
    is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

value = ""
if not is_admin:
    value = "not "

print(f"You are {value}running as an admin")

