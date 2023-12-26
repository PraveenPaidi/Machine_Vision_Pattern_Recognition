#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import cv2

path= r'C:\Users\prave\Downloads\HW 1\Prob4\Image4.jpg'
image= cv2.imread(path)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rows, cols = grey.shape
crow, ccol = rows // 2, cols // 2  # Center of the image


# Create a Gaussian filter
sigma = 30  # Adjust the standard deviation as needed

x =np.arange(cols) - ccol

y = np.arange(rows) - crow


# In[11]:



X, Y = np.meshgrid(x, y)

print(Y.shape)
mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
mask = mask / mask.max()  # Normalize to [0, 1]


# In[ ]:




