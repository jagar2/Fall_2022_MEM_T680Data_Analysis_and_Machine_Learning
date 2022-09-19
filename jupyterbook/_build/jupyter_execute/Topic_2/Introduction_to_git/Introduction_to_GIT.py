#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import IFrame


# # Introduction to GitHub

# * GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

# ## Creating a Repository
# **repository** is usually used to organize a single project. 
# * Contains: folders and files, images, video, spreadsheets, and data.

# ### Standard Files
# 
# #### Readme
# * Contains information about the repository

# #### License file
# * Sets the license that allows others to use your software

# ## Common license

# ### MIT License
# Copyright \<YEAR> \<COPYRIGHT HOLDER>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ### GNU General Public License 
# <one line to give the program's name and a brief idea of what it does.>
# Copyright (C) <year>  <name of author>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License

# ### Apache License, Version 2
# Copyright \[yyyy] [name of copyright owner]
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ## Making a respository
# 
# www.github.com

# ## Why use GIT? 
# ### Have you ever saved versions of a file like this: 
# * `Graph`
# * `Graph_V2`
# * `Graph_V2_final`
# * `Graph_V2_final_V3`
# 
# Branches accomplish a similar goal in a GitHub Repository
# * Work on code in `branch` ... once verified merge it to the `master`

# Distributed Version control system
# * Every developer has a backup of the entire repository
# ![figure](figs/distributed.png)

# ### Check that git is installed and the verison
# `git --version`

# ### If git is not installed
# `sudo apt install git-all`

# ### set your own configuration
# `git config --global user.name "Joshua Agar"` <br>
# `git config --global user.email "jca92@drexel.edu"`<br>
# `git config --list`

# ### How to find help?
# `git config --help`

# ## Have a local codebase that you want to start tracking
# 
# `mkdir Make_New`
# 
# `cd Make_New`

# ### initiates a new repository
# `git init`

# ### Makes an ignore file
# 
# This file is useful if you want git not to track spectific files
# 
# `touch .gitignore`

# ### Make a file from the command line
# 
# You can add any files to the git repository using any method that has access to your file system
# 
# `echo This is some text > myfile.txt`

# ### Add files to remote
# 
# `git add .` 

# ### Commit the files
# `git commit -m "initial commit"`

# ### Check Status
# 
# `git status`
# `git log`

# ## Connecting to an Online Repository

# ### Making a remote repo
# 
# #### General example
# 
# ```git remote add origin https://github.com/user/repo.git ``` <br>
# 
# `git push origin master`

# ## Cloning a Repo
# 
# Let's start by forking a [Hello World Repository](https://github.com/octocat/Hello-World.git)
# 
# Forking a repository allows you to make changes locally.
# 
# `git clone https://github.com/octocat/Hello-World.git`
# 
# ### see it with 
# 
# `ls -la`

# ## Modify Some File
# 
# Let's modify the Hello World Readme.md file, and look at the difference.
# 
# Check your modifications
# `git diff`

# ### Check your status
# `git status`

# ### Stage files to update
# `git add -A`
# or
# `git add .`
# or
# `git add README.md`

# ## Advanced File Adding
# 
# ### Adding files one by one
# `git add filename`
# 
# ### Adding all files in the current directory
# `git add -A`
# 
# ### Adding all files changes in the current directory
# `git add .`
# 
# ### Choosing what changes to add (this will got through all your  changes and you can 'Y' or 'N' the changes)
# `git add -p`

# ## Commit Files
# `git commit -m "INSERT TEXT HERE"` 
# <br>
# it is important to add comment so you know what your commit is
# 

# ## Advanced Commits
# 
# ### Commit staged file(s) 
# ### This is typically used for shorter commit messages
# `git commit -m 'commit message'`
# 
# ### Add file and commit in one shot
# `git commit filename -m 'commit message'`
# 
# ### Add file and commit staged file
# `git commit -am 'insert commit message'`
# 
# ### Changing your most recent commit message
# `git commit --amend 'new commit message'` 
# 
# ### Combine a sequence of commits together into a single one 
# #### You might use this to organize a messy commit history
# `git rebase -i`
# 
# ### This will give you an interface on your core editor:
# #### Commands:
# `p`, `pick` = use commit <br>
# `r`, `reword` = use commit, but edit the commit message <br>
# `e`, `edit` = use commit, but stop for amending <br>
# `s`, `squash` = use commit, but meld into previous commit <br>
# `f`, `fixup` = like "squash", but discard this commit's log message <br>
# `x`, `exec` = run command (the rest of the line) using shell

# ## Pull Commits
# In case someone else has made changes to the repository while you are working you should pull those changes to your local repository <br>
# 
# `git pull origin master` <br>
# 
# `origin` - means the repository you are in

# ## Push to Online repository
# `git push origin master`

# ## Branching
# **Branching** is the way to work on different versions of a repository at the same time
# By default your repository has a branch named `master`. 
# * Good to use branch to experiment and create edits before committing them to the `master`
# * When working on a branch you are taking the `master` at a point in time
#     - If others make changes to master you can `pull` them to your branch

# ## Diagram of workflow
# ![figure](figs/branch.png)
# Diagram shows:
# * The `master` branch
# * A new branch called `feature`
# * The journey that `feature` takes before it is merged with the `master`

# Branches accomplish a similar goal in a GitHub Repository
# * Work on code in `branch` ... once verified merge it to the `master`

# ## Syntax for branching
# 
# ### Create a local branch to work on
# `git checkout -b branchname`
# 
# ### Switching between 2 branches 
# `git checkout branch_1`<br>
# `git checkout branch_2`
# 
# ### Pushing your new local branch to remote as backup
# `git push -u origin branch_2`
# 
# ### Deleting a local branch 
# - this won't let you delete a branch that hasn't been merged yet
# 
# `git branch -d branch_2`
# 
# ### Deleting a local branch 
# - this WILL delete a branch even if it hasn't been merged yet!
# 
# `git branch -D branch_2`
# 
# ### Viewing all current branches for the repository
# - Includes both local and remote branches. Great to see if you already have a branch for a particular feature addition, especially on bigger projects
# 
# `git branch -a`
# 
# ### Viewing all branches that have been merged into your current  branch
# - including local and remote. Great for seeing where all your code has come from!
# `git branch -a --merged`
# 
# ### Viewing all branches that haven't been merged into your current branch
# - including local and remote
# `git branch -a --no-merged`
# 
# ### Viewing all local branches
# `git branch`
# 
# ### Viewing all remote branches
# `git branch -r`
# 
# ### Rebase master branch into local branch
# `git rebase origin/master`
# 
# ### Pushing local branch after rebasing master into local branch
# `git push origin branchname`

# ## Example with Branches
# `git branch new_branch`

# ### list the location of the repository
# `git remote -v`
# 
# ### lists the branches local or remote
# `git branch -a`

# ### Switching or checking out a branch
# `git checkout new_branch`

# Let's make a copy of the `readme.md` file in the branch

# `git add -A` <br>
# Only effects the local branch `new_branch`

# `git commit -m "new branch changes"`

# ### Push branch to online repository
# `u` - means associates the local branch with the online branches <br>
# `git push -u origin new_branch`

# `git branch -a`

# ### Checkout Master
# `git checkout master`

# ### Good idea to pull again
# `git pull`

# ### Merge branches
# 
# #### check the branches that we have merged thus far
# `git branch --merged`

# #### Merge branches
# `git merge new_branch`

# #### Push changes
# `git push origin master`

# ### Deleting old branch

# `git branch --merged`

# ### Deletes the local branch: <br>
# 
# `git branch -d new_branch`

# `git branch -a`

# ### Deletes the remote branch <br>
# `git push origin --delete new_branch`

# ## Fixing Common Mistakes

# ### Switch to the version of the code of the most recent commit
# `git reset HEAD` <br>
# `git reset HEAD -- filename` - for a specific file

# ### Switch to the version of the code before the most recent commit
# `git reset HEAD^ -- filename`<br>
# `git reset HEAD^` -- filename # for a specific file

# ### Switch back 3 or 5 commits
# `git reset HEAD~3` -- filename <br>
# `git reset HEAD~3` -- filename - for a specific file <br>
# `git reset HEAD~5` -- filename <br>
# `git reset HEAD~5` -- filename - for a specific file

# ### Soft reset
# #### Switch back to a specific commit
#  Where the '0766c053' is the commit ID
#  
# `git reset 0766c053 -- filename`
# 
# `git reset 0766c053 -- filename` -for a specific file

# ### Hard reset
# The previous commands were what's known as "soft" resets. Your  code is reset, but git will still keep a copy of the other code  handy in case you need it. On the other hand, the --hard flag tells Git to overwrite all changes in the working directory.
# `git reset --hard 0766c053`

# ### Error to the commit message
# `git commit --amend -m "completed subtract function"` <br>
# Note this will change the hash

# ## How to delete all changes
# `git reset --hard [commit]`

# ## Git Stash
# Stores files that you are not ready to commit but you want to save for a later use
# * Good when you have uncommited files that you want to move between branches

# ### Saves the stash <br>
# `git stash save "message"`

# ### Checking the stash <br>
# `git stash list`

# ### Applying the stash <br>
# `git stash apply`

# ### Deleting stash <br>
# `git stash drop [stash number]` <br>
# `git stash clear`

# In[2]:


# Git Cheat Sheet
IFrame(src='https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf', width=1000, height=1000)


# ## Other Useful Tricks

# ### Searching
# 
# Searches for parts of strings in a directory <br>
# `git grep 'something'`
# 
# Searches for parts of strings in a directory and the -n prints out the line numbers where git has found matches<br>
# `git grep -n 'something'`
# 
# Searches for parts of string with some context (some lines  before and some after the 'something' we are looking for) <br>
# `git grep -C<number of lines> 'something'`
# 
# Searches for parts of string and also shows lines BEFORE it <br>
# `git grep -B<number of lines> 'something'`
# 
# Searches for parts of string and also shows lines AFTER it <br>
# `git grep -A<number of lines> 'something'`

# ### Checking Authorship
# Show alteration history of a file with the name of the author <br>
# `git blame 'filename'`
# 
# Show alteration history of a file with the name of the author and the git commit ID <br>
# `git blame 'filename' -l`

# ### Logging
# Show a list of all commits in a repository. This command shows  everything about a commit, such as commit ID, author, date and  commit message. <br>
# `git log`
# 
# List of commits showing only commit messages and changes<br>
# `git log -p`
# 
# List of commits with the particular string you are looking for<br>
# `git log -S 'something'`
# 
# List of commits by author<br>
# `git log --author 'Author Name'`
# 
# Show a summary of the list of commits in a repository. This    shows a shorter version of the commit ID and the<br> commit message.
# `git log --oneline`
# 
# Show a list of commits in a repository since yesterday<br>
# `git log --since=yesterday`
# 
# Shows log by author and searching for specific term inside the  commit message<br>
# `git log --grep "term" --author "name"`
