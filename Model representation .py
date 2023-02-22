#!/usr/bin/env python
# coding: utf-8

# # Model Representation
# 
# # Problem Statement
# <img align="left" src="./images/C1_W1_L3_S1_trainingdata.png"    style=" width:380px; padding: 10px;  " /> 
# 
# As in the lecture, you will use the motivating example of housing price prediction.  
# This lab will use a simple data set with only two data points - a house with 1000 square feet(sqft) sold for \\$300,000 and a house with 2000 square feet sold for \\$500,000. These two points will constitute our *data or training set*. In this lab, the units of size are 1000 sqft and the units of price are 1000s of dollars.
# 
# | Size (1000 sqft)     | Price (1000s of dollars) |
# | -------------------| ------------------------ |
# | 1.0               | 300                      |
# | 2.0               | 500                      |
# 
# You would like to fit a linear regression model (shown above as the blue straight line) through these two points, so you can then predict price for other houses - say, a house with 1200 sqft.
# 

# import numpy as np
# import matplotlib.pyplot as plt
# 
# x_train = np.array([1.0, 2.0])
# y_train = np.array([300.0, 500.0])
# 
# print(f"x_train = {x_train}")
# print(f"y_train = {y_train}")

# In[ ]:





# In[3]:


# Number of training examples m

print(f"x_train.shape: {x_train.shape}")
m = x_train.shape
print(f"number of training is: {m}")


# In[6]:


# Also can use len() function for number of training

m = len(x_train)
print(f"number of training is :{m}")


# In[8]:


# training example of x_i, y_i
i = 1 

x_i = x_train[i]
y_i = y_train[i]

print(f"(x^{i}),y^({i})) = ({x_i},{y_i})")


# In[13]:


# plot the data point

plt.scatter(x_train, y_train, marker='*', c='blue') 
#marker means what do u want to graph or point
#squence of color for marker

#set the title

plt.title("Housing prices")

# set the y_axis and x_axis

plt.ylabel('price(in 1000s of doller)')
plt.xlabel('size(1000 sqrf)')
plt.show


# # Model Function 
# 
# <img align="left" src="./images/C1_W1_L3_S1_model.png"     style=" width:380px; padding: 10px; " > The model function for linear regression (which is a function that maps from `x` to `y`) is represented as 
# 
# $$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
# 

# In[25]:


#The formula represent straight line

w = 200
b = 100

print(f"w: {w}")
print(f"b: {b}")      


# In[22]:


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    print("i",f_wb)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


# In[26]:


tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
# plot gives straight line
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
# scatter gives us datapoints
plt.scatter(x_train, y_train, marker='*', c='pink',label='d Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (100 sqft)')
plt.legend() #This creates a box showing us all the naming and symbols 
plt.show()


# In[29]:


#Predicatoin

w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft :.0f} thousand dollars")


# In[ ]:




