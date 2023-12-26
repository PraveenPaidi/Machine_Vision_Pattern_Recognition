#!/usr/bin/env python
# coding: utf-8

# In[54]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

path = r'C:\Users\prave\Downloads\HW 1\New folder\IMG_2388.mov'
cap = cv2.VideoCapture(path)


# Initialize variables
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0

ret1, frame1 = cap.read()
(height, width, l) = frame1.shape

sum_of_frames = np.zeros((height, width, 3), dtype=np.float)

# Step 2: Loop through frames
frame1= np.zeros(frame1.shape)
count=0
framefiles=[]
frames=[]
intensities=[]

while True:
    
    ret, frame = cap.read()
    
    
    
    if not ret:
        break
        
    frames.append(frame)
    intensities.append(np.sum(frame))
    
    
    framefiles.append(frame)
    a = np.sum(np.subtract(frame, frame1))   
    count= count+1
    if a<201488593: 
        continue
    
    else:
        sum_of_frames += frame
        frame_count += 1
        frame1= frame
    
print(frame_count)
print(count)

highest_intensity_frame_index = intensities.index(max(intensities))
highest_intensity_frame = frames[highest_intensity_frame_index]

          
# Step 3 and 4: Calculate the average
average_frame = sum_of_frames / frame_count

# Clean up
cap.release()
cv2.destroyAllWindows()

# Save or display the average frame
cv2.imwrite('average_frame.png', highest_intensity_frame)

# average_frame= np.rint(average_frame)
# average_frame=average_frame.astype(np.uint8)


# In[56]:


import os
path = r'C:\Users\prave\Downloads\HW 1\New folder\path'
highest_intensity_frame_index


# In[57]:


highest_intensity_frame_index
plt.imshow(highest_intensity_frame)


# In[62]:


l=[]
for i in range(len(framefiles)):    #  400 to 500   
    if i>150 and i<400:
        l1=cv2.subtract(highest_intensity_frame, framefiles[i])*5
    # 10 is good  , 5 is better, 4 is good
        l.append(l1)
        cv2.imwrite(os.path.join(path , 'difference'+str(i)+'.jpg'),l1)
    


# In[ ]:





# In[ ]:





# In[53]:


cv2.imwrite('kokok.jpg', l[50])


# In[ ]:




