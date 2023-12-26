#!/usr/bin/env python
# coding: utf-8

# In[13]:


from PIL import Image, ImageDraw, ImageOps
import random
import os
import numpy as np

# Create a directory to save the images
os.makedirs('colored_squares_dataset', exist_ok=True)

for i in range(100000):
    
    # Create a black background
    image = Image.new('RGB',  (64, 64), 'black')
    
    # Generate random properties of the square
    size = random.randint(10, 40)  # Random size between 10 and 40 pixels
    x = random.randint(0, 64 - size)  # Adjusted for the image size and square size
    y = random.randint(0, 64 - size)  # Adjusted for the image size and square size
    a, b, c = np.random.randint(0, 255, 3)
    color = (a, b, c)
    angle = random.randint(0, 90)
    
    # Draw the square on the black background
    draw = ImageDraw.Draw(image)
    draw.rectangle([(x, y), (x + size, y + size)], fill=color)
    
    # Rotate the image
    rotated_image = image.rotate(angle, expand=True)
    
    # Save the image
    rotated_image.save(f'colored_squares_dataset/image_{i}.png')

print("Images generated and saved successfully.")


# In[10]:


from PIL import Image
import matplotlib.pyplot as plt
import os

# Path to the directory containing the generated images
image_dir = 'colored_squares_dataset'

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Display 20 images in a grid
num_images_to_display = 20
num_rows = 4
num_cols = 5

fig, axes = plt.subplots(num_rows, num_cols, figsize=(4, 4))

for i, ax in enumerate(axes.flat):
    if i < num_images_to_display:
        image_path = os.path.join(image_dir, image_files[i])
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




