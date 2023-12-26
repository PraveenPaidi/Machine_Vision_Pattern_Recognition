#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
from PIL import Image, ImageDraw

# Create a new image with a black background
img = Image.new('RGB', (300, 300), 'black')

# Generate random angles for a regular quadrilateral (in radians)
angles = np.sort(np.random.rand(4) * 2 * np.pi)

# Define the size of the quadrilateral
side_length = 50

# Calculate the coordinates for the corners
x_coords = np.round(150 + side_length * np.cos(angles))
y_coords = np.round(150 + side_length * np.sin(angles))

# Calculate the centroid of the quadrilateral
centroid_x = np.mean(x_coords)
centroid_y = np.mean(y_coords)

# Translate the coordinates so the centroid is at the center of the image
x_coords = x_coords - centroid_x + 150
y_coords = y_coords - centroid_y + 150

# Define the points for the quadrilateral
points = list(zip(x_coords, y_coords))

# Draw the quadrilateral
draw = ImageDraw.Draw(img)
draw.polygon(points, fill='white')

# # Save the image
# img.save('random_quadrilateral_centered1.png')


# In[50]:


import cv2
# Load the original image
original_image = Image.open("random_quadrilateral_centered1.png")

# Convert the image to a NumPy array
original_array = np.array(original_image)

# Define the translation matrix
translation_matrix = np.float32([[1, 0, 30], [0, 1, 100]])

# Apply the translation
translated_image = cv2.warpAffine(original_array, translation_matrix, (300, 300))

result_image = Image.fromarray(translated_image)

result_image.save('translated1.png')

# Define the rotation matrix
center = (150, 150)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)

# Apply the rotation
rotated_image = cv2.warpAffine(translated_image, rotation_matrix, (300, 300))

# Convert the result back to an Image object
result_image = Image.fromarray(rotated_image)

# result_image.save("Rotated1.png")


# In[35]:


T= np.float32([[1, 0, 30], [0, 1, 100],[0, 0, 1]])
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)


# In[36]:


rotation_matrix


# In[37]:


final =np.dot( rotation_matrix,T)
print(final)


# In[64]:



x, y  = 110,121

# Convert to homogeneous coordinates (add 1 for the third element)
original_point = np.array([x, y, 1])

# Apply the rotation matrix
rotated_point = np.dot(final, original_point)

# Extract the x and y coordinates of the rotated point
x_rotated, y_rotated = rotated_point

# Convert back to non-homogeneous coordinates
x_rotated, y_rotated = int(x_rotated), int(y_rotated)

print(f'Original Pixel: ({x}, {y})')
print(f'Rotated Pixel: ({x_rotated}, {y_rotated})')


# In[ ]:




