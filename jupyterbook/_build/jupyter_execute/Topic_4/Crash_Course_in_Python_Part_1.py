#!/usr/bin/env python
# coding: utf-8

# # A Crash Course in Python: Part 1
# 

# ## Why Python?
# 

# - Python is a dynamic, interpreted (bytecode-compiled) language
# 

# - There are no type declarations of variables, parameters, functions, or methods in source code
# 

# - Python code short and flexible and quick to use
# 

# - Python tracks the types of all values at runtime and flags code that does not make sense as it runs
# 

# - Python is open source so there are no paywalls (e.g., Matlab)
# 

# - There is a massive open source developer community
# 

# ## Numbers
# 
# - You can type an expression at it and it will write the value
# 

# In[1]:


25


# ## Variables
# 
# Used to store values in a human readable form
# 

# ### Rules for variables
# 
# - Must start with a letter
# - Can only contain letters and numbers
# - Python considers `_` as a letter
# - Best practices is for variables to be lowercase
# 

# ## Operators in Python
# 
# - Operators are ways to have expressions interact with one another
# 

# ### Mathematical Operators
# 
# - Used for mathematical operations
# 

# ![title](figs/Operators_1.png)
# 

# #### Addition
# 

# In[2]:


5 + 5


# #### Subtraction
# 

# In[3]:


10 - 7


# #### Division
# 

# In[4]:


8 / 5


# ```{note}
# division always returns a floating point number
# ```
# 

# #### Parenthesis
# 

# In[5]:


(50 - 5 * 6) / 4


# #### Modulus
# 
# Modulus returns the remainder after division
# 

# In[6]:


29 % 3


# #### Exponential
# 

# In[7]:


3**9


# #### Floor Division
# 

# In[8]:


29 // 3


# ### Comparison Operators
# 

# - Used for comparison between expressions
# 

# ![](figs/Comparison_Operators.png)
# 

# #### Equal to
# 

# In[9]:


print(10 == 10)
print(10 == 11)


# #### Not Equal to
# 

# In[10]:


print(10 != 10)
print(10 != 11)


# #### Greater than
# 

# In[11]:


print(9 > 10)
print(10 > 10)
print(11 > 10)


# #### Less than
# 

# In[12]:


print(9 < 10)
print(10 < 10)
print(11 < 10)


# #### Greater than Equal to
# 

# In[13]:


print(9 >= 10)
print(10 >= 10)
print(11 >= 10)


# #### Less than Equal to
# 

# In[14]:


print(9 <= 10)
print(10 <= 10)
print(11 <= 10)


# ### Assignment Operators
# 
# - This allows you to assign a value to a variable
# 

# ![title](figs/Assignment_Operators.png)
# 

# #### Equals
# 

# In[15]:


five = 5

print(five)


# #### Adds to
# 

# In[16]:


five += 5

print(five)


# #### Subtracts
# 

# In[17]:


print(five)

five -= 5

print(five)


# #### Multiply
# 

# In[18]:


print(five)

five *= 5

print(five)


# #### Divide
# 

# In[19]:


print(five)

five /= 5

print(five)


# #### Modulus
# 

# In[20]:


twenty_nine = 29
print(twenty_nine)

twenty_nine %= 3
print(twenty_nine)


# #### Exponent
# 

# In[21]:


print(five)

five**5

print(five)


# #### Floor Divide
# 

# In[22]:


twenty_nine = 29
print(twenty_nine)

twenty_nine //= 3

print(twenty_nine)


# ### Logical Operators
# 
# - Used to evaluate if a condition is met
# 

# ![title](figs/Logical_Operators.png)
# 

# #### Logical And (`and`)
# 

# In[23]:


print(True and True)


# In[24]:


print(False and False)


# In[25]:


print(False and False)


# #### Logical Or (`or`)
# 

# In[26]:


print(True or True)


# In[27]:


print(False or False)


# In[28]:


print(False or False)


# #### Logical Not (`not`)
# 

# In[29]:


print(not (True and True))


# In[30]:


print(not (True or False))


# In[31]:


print(not (False and False))


# ### Membership Operators
# 
# - Determines if a value is within an expression
# 

# ![](figs/Membership_Operators.png)
# 

# #### `in` operator
# 

# In[32]:


print("p" in "apple")


# In[33]:


print("i" in "apple")


# In[34]:


print("ap" in "apple")


# In[35]:


print("pa" in "apple")


# #### `not in` operator
# 

# In[36]:


print("x" not in "apple")


# In[37]:


print("a" not in "apple")


# #### `in` in lists
# 

# In[38]:


print("a" in ["apple", "pear", "peach"])


# In[39]:


print("apple" in ["apple", "pear", "peach"])


# ## Updating Expressions
# 
# - You can update an expression
# 

# In[40]:


x = 10
x = x + 10
print(x)


# ## Data Types
# 

# ### Strings
# 
# **Strings** are arrays of bytes representing Unicode characters
# 
# Example `'Drexel University'` or `"Drexel University"`
# 

# In[41]:


print("This is my first string")


# ### Accessing characters in Strings
# 
# Individual characters can be accessed by indexing:
# 
# - Positive numbers from the front of the string (starting with 0)
# - Negative numbers from the back of the string (starting with -1)
# - Indexing is only possible with a `int`
# 

# In[42]:


# Python Program to Access
# characters of String

String1 = "This is my second string"
print("Initial String: ")
print(String1)


# In[43]:


# Printing First character
print("First character of String is: ")
print(String1[0])


# In[44]:


# Printing Last character
print("Last character of String is: ")
print(String1[-1])


# ### String Slicing
# 
# A range of characters can be accessed using the `:` operator
# 

# In[45]:


# Python Program to
# demonstrate String slicing

# Creating a String
String1 = "This is my third string"
print("Initial String: ")
print(String1)


# In[46]:


# Printing 3rd to 12th character
print("Slicing characters from 3-12: ")
print(String1[3:12])


# In[47]:


# Printing characters between
# 3rd and 2nd last character
print("Slicing characters between " + "3rd and 2nd last character: ")
print(String1[3:-2])


# ### Updating Strings
# 
# Strings cannot be directly updated. They need to be replaced or reformed
# 

# In[48]:


# Python Program to Update
# character of a String

String1 = "This is my fourth string"
print("Initial String: ")
print(String1)


# In[49]:


# Updating a character of the String
## As python strings are immutable, they don't support item updates directly
### there are following two ways
# 1
list1 = list(String1)
list1[2] = "2"
String2 = "".join(list1)
print("Updating character at 2nd Index: ")
print(String2)


# In[50]:


# 2
String3 = String1[0:2] + "2" + String1[3:]
print(String3)


# #### Replacing Strings
# 

# In[51]:


# Python Program to Update
# entire String

String1 = "Hello, I'm a Drexel Dragon"
print("Initial String: ")
print(String1)


# In[52]:


# Updating a String
String1 = "Welcome to Drexel"
print("Updated String: ")
print(String1)


# ### Formatting Strings
# 
# - There are times where you want to create a string from a variable
# - If using Python 3, f-strings should be used
# 

# In[53]:


name = "Eric"
age = 74
f"Hello, {name}. You are {age}."


# #### Number formats
# 
# Sometimes it is nice to format numbers in different ways, there are conventions to do this.
# 

# In[54]:


# Formatting of Integers
String1 = f"{16:b}"
print("Binary representation of 16 is ")
print(String1)


# In[55]:


# Formatting of Floats
String1 = f"{165.6458:e}"
print("Exponent representation of 165.6458 is ")
print(String1)


# In[56]:


# Rounding off Integers
String1 = f"{1/6:.2f}"
print("one-sixth is : ")
print(String1)


# ## Lists
# 
# Lists are dynamically sized arrays
# 

# In[57]:


Var = ["I'm", "a", "Drexel", "Engineer"]
print(Var)


# - List are powerful because they can contain data of a variety of types (e.g., `strings`, `floats`, `ints`)
# - They are mutable, this means they can be altered after they are created
# 

# ### Creating a list in Python
# 

# In[58]:


# Python program to demonstrate
# Creation of List

# Creating a List
List = []
print("Blank List: ")
print(List)


# In[59]:


# Creating a List of numbers
List = [10, 20, 14]
print("List of numbers: ")
print(List)


# In[60]:


# Creating a List of strings and accessing
# using index
List = ["I'm", "a", "Drexel", "Engineer"]
print("List Items: ")
print(List[0])
print(List[2])


# ### Creating a list with multiple elements
# 

# In[61]:


# Creating a List with
# the use of Numbers
# (Having duplicate values)
List = [1, 2, 4, 4, 3, 3, 3, 6, 5]
print("List with the use of Numbers: ")
print(List)


# In[62]:


# Creating a List with
# mixed type of values
# (Having numbers and strings)
List = [1, 2, "Drexel", 4, "For", 6, "Engineers"]
print("List with the use of Mixed Values: ")
print(List)


# ### Accessing Elements of a List
# 

# - Use the index operator `[ ]` to access an item in a list
# - The index must be an integer
# - Nested lists are accessed using nested indexing.
# 

# #### Example: Accessing Elements of a List
# 

# In[63]:


# Python program to demonstrate
# accessing of element from list

# Creating a List with
# the use of multiple values
List = ["I'm", "a", "Drexel", "Engineer"]


# ##### Instructions:
# 
# 1. Print a statement that says "Accessing an element from a list"
# 1. Print the first item of a list
# 1. Print the third item of a list
# 

# In[64]:


## Type your code here




# In[65]:


# accessing a element from the
# list using index number
print("Accessing an element from the list")
print(List[0])
print(List[2])


# #### Example: Multidimensional Indexing of a List
# 

# In[66]:


# Creating a Multi-Dimensional List
# (By Nesting a list inside a List)
List = [["I`m", "a", "Drexel"], ["Engineer"]]


# ##### Instructions:
# 
# 1. Print a statement that says Accessing an element from a "Multi-dimensional list"
# 1. Print Drexel from the list
# 1. Print Engineer from the list
# 

# In[67]:


# accessing an element from the
# Multi-Dimensional List using
# index number
print("Accessing an element from a Multi-Dimensional list")
print(List[0][2])
print(List[1][0])


# ### Getting the Size of a Python List
# 
# - Python `len()` can be used to get the size of list
# 

# In[68]:


# Creating a List
List1 = []
print(len(List1))


# In[69]:


# Creating a List of numbers
List2 = [10, 20, 14]
print(len(List2))


# ### Adding and Removing Elements to a Python List
# 
# - Elements can be added to a list using the `append()` function
# - Only one element can be added to a list at a time
# - `Tuples` can be added to a list since they are immutable
# - You can add a list to a list
# 

# In[70]:


# Python program to demonstrate
# Addition of elements in a List

# Creating a List
List = []
print("Initial blank List: ")
print(List)


# In[71]:


# Addition of Elements
# in the List
List.append(1)
List.append(2)
List.append(4)
print("List after Addition of Three elements: ")
print(List)


# In[72]:


# Adding elements to the List
# using Iterator
for i in range(1, 4):
    List.append(i)
print("List after Addition of elements from 1-3: ")
print(List)


# In[73]:


# Adding Tuples to the List
List.append((5, 6))
print("List after Addition of a Tuple: ")
print(List)


# In[74]:


# Addition of List to a List
List2 = ["Drexel", "Engineer"]
List.append(List2)
print("List after Addition of a List: ")
print(List)


# #### Example: Using the Insert Method
# 
# - `insert(position, value)` insert allows you to insert a new value at a specific index
# 

# In[75]:


# Python program to demonstrate
# Addition of elements in a List

# Creating a List
List = [1, 2, 3, 4]
print("Initial List: ")
print(List)


# In[76]:


# Addition of Element at
# specific Position
# (using Insert Method)
List.insert(3, 12)
List.insert(0, "Drexel")
print("List after performing Insert Operation: ")
print(List)


# #### Example: Removing Elements from a List
# 
# - `remove()` removes a single value from a list, multiple values can be removed with an iterator
# 

# In[77]:


# Python program to demonstrate
# Removal of elements in a List

# Creating a List
List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
print("Initial List: ")
print(List)


# In[78]:


# Removing elements from List
# using Remove() method
List.remove(5)
List.remove(6)
print("List after Removal of two elements: ")
print(List)


# In[79]:


# Python program to demonstrate
# Removal of elements in a List with an iterator

# Creating a List
List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
print("Initial List: ")
print(List)


# In[80]:


# Removing elements from List
# using Remove() method
for i in [5, 6]:
    List.remove(i)
print("List after Removal of two elements: ")
print(List)


# ### Tuples
# 
# - Tuple is a collection of Python objects much like a list
# - The sequence of values stored in a tuple can be of any type, and they are indexed by integers
# 

# #### Creating a Tuple
# 
# - Tuples are created by placing a sequence of values separated by `comma` with or without the use of parentheses for grouping the data sequence.
# 

# In[81]:


# Creating an empty Tuple
Tuple1 = ()
print("Initial empty Tuple: ")
print(Tuple1)


# In[82]:


# Creating a Tuple
# with the use of string
Tuple1 = ("Drexel", "Engineering")
print("Tuple with the use of String: ")
print(Tuple1)


# In[83]:


# Creating a Tuple with
# the use of list
list1 = [1, 2, 4, 5, 6]
print("Tuple using List: ")
print(tuple(list1))


# In[84]:


# Creating a Tuple
# with the use of built-in function
Tuple1 = tuple("Drexel")
print("Tuple with the use of function: ")
print(Tuple1)


# #### Accessing Tuples
# 

# - Tuples are immutable, and usually, they contain a sequence of heterogeneous elements that are accessed via unpacking or indexing (or even by attribute in the case of named tuples)
# 

# ```{note}
# In unpacking of tuple number of variables on the left-hand side should be equal to a number of values in given tuple a
# ```
# 

# In[85]:


# Accessing Tuple
# with Indexing
Tuple1 = tuple("Drexel")
print("First element of Tuple: ")
print(Tuple1[0])


# In[86]:


# Tuple unpacking
Tuple1 = ("Drexel", "is", "Engineering")


# In[87]:


# This line unpack
# values of Tuple1
a, b, c = Tuple1
print("Values after unpacking: ")
print(a)
print(b)
print(c)


# #### Concatenation of Tuples
# 
# - The process of joining two or more Tuples
# - Concatenation is done by the use of `+` operator
# - Concatenation of tuples is done always from the end of the original tuple
# - Other arithmetic operations do not apply on Tuples
# 

# In[88]:


# Concatenation of tuples
Tuple1 = (0, 1, 2, 3)
Tuple2 = ("Drexel", "is", "Engineering")


# In[89]:


Tuple3 = Tuple1 + Tuple2

# Printing first Tuple
print("Tuple 1: ")
print(Tuple1)


# In[90]:


# Printing Second Tuple
print("Tuple2: ")
print(Tuple2)


# In[91]:


# Printing Final Tuple
print("Tuples after Concatenation: ")
print(Tuple3)


# #### Slicing of Tuples
# 
# - Slicing of a Tuple is done to fetch a specific range or slice of sub-elements from a Tuple
# 

# In[92]:


# Slicing of a Tuple

# Slicing of a Tuple
# with Numbers
Tuple1 = tuple("DrexelEngineering")


# In[93]:


# Slicing of a Tuple

# Slicing of a Tuple
# with Numbers
Tuple1 = tuple("DrexelEngineering")

# Removing First element
print("Removal of First Element: ")
print(Tuple1[1:])


# In[94]:


# Reversing the Tuple
print("Tuple after sequence of Element is reversed: ")
print(Tuple1[::-1])


# In[95]:


# Printing elements of a Range
print("Printing elements between Range 4-9: ")
print(Tuple1[4:9])


# ### Dictionary
# 
# - Collection of keys values, used to store data values like a map
# 

# #### Example of a Dictionary
# 

# In[96]:


Dict = {1: "Drexel", 2: "is", 3: "Engineering"}
print(Dict)


# #### Creating a Dictionary
# 
# - A dictionary can be created by placing a sequence of elements within curly `{}` braces, separated by `comma`.
# - Dictionary holds pairs of values, one being the Key and the other corresponding pair element being its Key:value.
# - Values in a dictionary can be of any data type and can be duplicated, whereas keys can’t be repeated and must be immutable.
# 

# In[97]:


# Creating a Dictionary
# with Integer Keys
Dict = {1: "Geeks", 2: "For", 3: "Geeks"}
print("Dictionary with the use of Integer Keys: ")
print(Dict)


# In[98]:


# Creating a Dictionary
# with Mixed keys
Dict = {"Name": "Geeks", 1: [1, 2, 3, 4]}
print("Dictionary with the use of Mixed Keys: ")
print(Dict)


# #### Example Nested Dictionaries
# 

# In[99]:


# Creating a Nested Dictionary
# as shown in the below image
Dict = {1: "Drexel", 2: "For", 3: {"A": "Welcome", "B": "To", "C": "Drexel"}}

print(Dict)


# #### Adding Elements to a Dictionary
# 
# - Value at a time can be added to a Dictionary by defining value along with the key e.g. Dict[Key] = ‘Value’
# - Updating an existing value in a Dictionary can be done by using the built-in update() method
# - Nested key values can also be added to an existing Dictionary
# 

# In[100]:


# Creating an empty Dictionary
Dict = {}
print("Empty Dictionary: ")
print(Dict)


# In[101]:


# Adding elements one at a time
Dict[0] = "Drexel"
Dict[2] = "For"
Dict[3] = 1
print("Dictionary after adding 3 elements: ")
print(Dict)


# In[102]:


# Adding set of values
# to a single Key
Dict["Value_set"] = 2, 3, 4
print("Dictionary after adding 3 elements: ")
print(Dict)


# In[103]:


# Updating existing Key's Value
Dict[2] = "Welcome"
print("Updated key value: ")
print(Dict)


# In[104]:


# Adding Nested Key value to Dictionary
Dict[5] = {"Nested": {"1": "Drexel", "2": "Engineering"}}
print("Adding a Nested Key: ")
print(Dict)


# #### Accessing Elements of a Dictionary
# 

# In[105]:


# Python program to demonstrate
# accessing a element from a Dictionary

# Creating a Dictionary
Dict = {1: "Drexel", "name": "is", 3: "Engineering"}

# accessing a element using key
print("Accessing a element using key:")
print(Dict["name"])


# In[106]:


# accessing a element using key
print("Accessing a element using key:")
print(Dict[1])


# In[107]:


# Creating a Dictionary
Dict = {1: "Drexel", "name": "For", 3: "Engineering"}

# accessing a element using get()
# method
print("Accessing a element using get:")
print(Dict.get(3))


# #### Accessing Element of a Nested Dictionary
# 

# In[108]:


# Creating a Dictionary
Dict = {"Dict1": {1: "Drexel"}, "Dict2": {"Name": "For"}}

# Accessing element using key
print(Dict["Dict1"])
print(Dict["Dict1"][1])
print(Dict["Dict2"]["Name"])


# #### Dictionary Methods
# 
# - [`clear()`](https://www.geeksforgeeks.org/python-dictionary-clear/) – Remove all the elements from the dictionary
# - [`copy()`](https://www.geeksforgeeks.org/python-dictionary-copy/) – Returns a copy of the dictionary
# - [`get()`](https://www.geeksforgeeks.org/get-method-dictionaries-python/) – Returns the value of specified key
# - [`items()`](https://www.geeksforgeeks.org/python-dictionary-items-method/) – Returns a list containing a tuple for each key value pair
# - [`keys()`](https://www.geeksforgeeks.org/python-dictionary-keys-method/) – Returns a list containing dictionary’s keys
# - [`pop()`](https://www.geeksforgeeks.org/python-dictionary-pop-method/) – Remove the element with specified key
# - [`popitem()`](https://www.geeksforgeeks.org/python-dictionary-popitem-method/) – Removes the last inserted key-value pair
# - [`update()`](https://www.geeksforgeeks.org/python-dictionary-update-method/) – Updates dictionary with specified key-value pairs
# - [`values()`](https://www.geeksforgeeks.org/python-dictionary-values/) – Returns a list of all the values of dictionary
# 

# In[109]:


# demo for all dictionary methods
dict1 = {1: "Python", 2: "Java", 3: "Ruby", 4: "Scala"}

# copy() method
dict2 = dict1.copy()
print("A copy of a string:")
print(dict2)


# In[110]:


# clear() method
dict1.clear()
print(dict1)


# In[111]:


# get() method
print(dict2.get(1))


# In[112]:


# items() method
print(dict2.items())


# In[113]:


# keys() method
print(dict2.keys())


# In[114]:


# pop() method
dict2.pop(3)
print(dict2)


# In[115]:


# popitem() method
dict2.popitem()
print(dict2)


# In[116]:


# update() method
dict2.update({3: "Scala"})
print(dict2)


# In[117]:


# values() method
print(dict2.values())


# ### Arrays
# 
# - An array is a collection of items stored at contiguous memory locations
# - The idea is to store multiple items of the same type together
# - This makes it easier to calculate the position of each element by simply adding an offset to a base value, i.e., the memory location of the first element of the array (generally denoted by the name of the array)
# 

# - Array can be handled in Python by a module named array
# - They can be useful when we have to manipulate only a specific data type values
# - A user can treat lists as arrays. However, user cannot constraint the type of elements stored in a list.
# 

# - If you create arrays using the array module, all elements of the array must be of the same type.
# - array(data_type, value_list) is used to create an array with data type and value list specified in its arguments.
# 

# #### Creating an Array
# 

# In[118]:


# Python program to demonstrate
# Creation of Array

# importing "array" for array creations
import array as arr

# creating an array with integer type
a = arr.array("i", [1, 2, 3])


# In[119]:


# printing original array
print("The new created array is : ", end=" ")
for i in range(0, 3):
    print(a[i], end=" ")
print()


# In[120]:


# creating an array with float type
b = arr.array("d", [2.5, 3.2, 3.3])

# printing original array
print("The new created array is : ", end=" ")
for i in range(0, 3):
    print(b[i], end=" ")


# #### Complexion of Arrays
# 
# ![](figs/Complexion_of_arrays.png)
# 

# #### Adding Elements to an Array
# 
# - Elements can be added to the Array by using built-in `insert()` function
# - Insert is used to insert one or more data elements into an array
# - `append()` can be used to add a value at the end of the array
# 

# In[121]:


# Python program to demonstrate
# Adding Elements to a Array

# importing "array" for array creations
import array as arr

# array with int type
a = arr.array("i", [1, 2, 3])


print("Array before insertion : ", end=" ")
for i in range(0, 3):
    print(a[i], end=" ")
print()


# In[122]:


# inserting array using
# insert() function
a.insert(1, 4)

print("Array after insertion : ", end=" ")
for i in a:
    print(i, end=" ")
print()


# In[123]:


# array with float type
b = arr.array("d", [2.5, 3.2, 3.3])

print("Array before insertion : ", end=" ")
for i in range(0, 3):
    print(b[i], end=" ")
print()


# In[124]:


# adding an element using append()
b.append(4.4)

print("Array after insertion : ", end=" ")
for i in b:
    print(i, end=" ")
print()


# #### Accessing Elements from an Array
# 
# - Use the index operator `[ ]` to access an item in a array
# 

# In[125]:


# Python program to demonstrate
# accessing of element from list

# importing array module
import array as arr

# array with int type
a = arr.array("i", [1, 2, 3, 4, 5, 6])

# accessing element of array
print("Access element is: ", a[0])


# In[126]:


# accessing element of array
print("Access element is: ", a[3])


# In[127]:


# array with float type
b = arr.array("d", [2.5, 3.2, 3.3])

# accessing element of array
print("Access element is: ", b[1])


# In[128]:


# accessing element of array
print("Access element is: ", b[2])


# #### Slicing an Array
# 
# - Slice operation is performed on array with the use of colon(`:`)
# - To print elements from beginning to a range use `[:Index]`
# - To print elements from end use `[:-Index]`, to print elements from specific Index till the end use `[Index:]`
# - To print elements within a range, use `[Start Index:End Index]`
# - To print whole List with the use of slicing operation, use `[:]`
# - To print whole array in reverse order, use `[::-1]`
# 
# ![](figs/slicing.png)
# 

# In[129]:


# Python program to demonstrate
# slicing of elements in a Array

# importing array module
import array as arr

# creating a list
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

a = arr.array("i", l)
print("Initial Array: ")
for i in a:
    print(i, end=" ")


# In[130]:


# Print elements of a range
# using Slice operation
Sliced_array = a[3:8]
print("Slicing elements in a range 3-8: ")
print(Sliced_array)


# In[131]:


# Print elements from a
# pre-defined point to end
Sliced_array = a[5:]
print("Elements sliced from 5th " "element till the end: ")
print(Sliced_array)


# In[132]:


# Printing elements from
# beginning till end
Sliced_array = a[:]
print("Printing all elements using slice operation: ")
print(Sliced_array)


# #### Searching element in a Array
# 
# - In order to search an element in the array we use a python in-built index() method
# - This function returns the index of the first occurrence of value mentioned in arguments
# 

# In[133]:


# Python code to demonstrate
# searching an element in array


# importing array module
import array

# initializing array with array values
# initializes array with signed integers
arr = array.array("i", [1, 2, 3, 1, 2, 5])

# printing original array
print("The new created array is : ", end="")
for i in range(0, 6):
    print(arr[i], end=" ")


# In[134]:


# using index() to print index of 1st occurrence of 2
print("The index of 1st occurrence of 2 is : ", end="")
print(arr.index(2))


# In[135]:


# using index() to print index of 1st occurrence of 1
print("The index of 1st occurrence of 1 is : ", end="")
print(arr.index(1))


# #### Updating Elements in a Array
# 
# - We can simply reassign a new value to the desired index we want to update
# 

# In[136]:


# Python code to demonstrate
# how to update an element in array

# importing array module
import array

# initializing array with array values
# initializes array with signed integers
arr = array.array("i", [1, 2, 3, 1, 2, 5])

# printing original array
print("Array before updating : ", end="")
for i in range(0, 6):
    print(arr[i], end=" ")


# In[137]:


# updating a element in a array
arr[2] = 6
print("Array after updating : ", end="")
for i in range(0, 6):
    print(arr[i], end=" ")


# In[138]:


# updating a element in a array
arr[4] = 8
print("Array after updating : ", end="")
for i in range(0, 6):
    print(arr[i], end=" ")

