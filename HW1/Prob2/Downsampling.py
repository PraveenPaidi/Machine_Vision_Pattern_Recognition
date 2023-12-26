#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import ipdb
import numpy as np

################################
#display using plt.imshow
path= r'C:\Users\prave\Downloads\HW 1\Prob2\Image.jpg'
image= cv2.imread(path)
# plt.imshow(image)
plt.savefig('Imshowimage.png', bbox_inches='tight', pad_inches=0, dpi=300)


#################################
# display using cv2.colorbgr2rgb
lol=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(lol)
plt.savefig('Imshowimagecv2bgrRGB.png', bbox_inches='tight', pad_inches=0, dpi=300)
cv2.imwrite('popop.jpg', lol)

#display using cv2.colorbgr2rgb
l=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(l)
# plt.savefig('Imshowimagecv2bgrGREY.png', bbox_inches='tight', pad_inches=0, dpi=300)

######################################

path= r'C:\Users\prave\Downloads\HW 1\prob2\Image3.jpg'
image2= cv2.imread(path)
print(image2.shape)


width = 384
height = 216 # keep original height
dim = (width, height)
# resize image
downsampled = cv2.resize(image2, dim)
cv2.imwrite('Downsamples.jpg', downsampled)

#######################################

width = 3840
height = 2160 # keep original height
dim = (width, height)
nearest_neighbours = cv2.resize(downsampled, dim, interpolation = cv2.INTER_NEAREST)
cv2.imwrite('Nearest_neigh.jpg', nearest_neighbours)
cubic = cv2.resize(downsampled, dim, interpolation = cv2.INTER_CUBIC)
cv2.imwrite('Bicubic.jpg', cubic)

#######################################
Diff_Nearest= image2 - nearest_neighbours
Diff_Cubic = image2 - cubic
print(np.sum(Diff_Cubic))
print(np.sum(Diff_Nearest))
# cv2.imwrite('Nearestneigh_diff', Diff_Nearest)
# cv2.imwrite('Bicubic_diff', Diff_Cubic)


# In[3]:


cv2.imwrite('Diff nearest neighbor.jpg', Diff_Nearest)
cv2.imwrite('Diff Cubic.jpg', Diff_Cubic)


# In[ ]:




