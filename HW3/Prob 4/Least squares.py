#!/usr/bin/env python
# coding: utf-8

# In[10]:


import re
import numpy as np

def parse_array(string):
    return [int(x) for x in re.findall(r'\d+', string)]

file_path = r"C:\Users\prave\Downloads\HW 3\Prob 4\final\final_matches.txt"

final_matches = []

with open(file_path, 'r') as file:
    for line in file:
        match_strings = line.strip().split('), ')
        match = (parse_array(match_strings[0]), parse_array(match_strings[1]))
        final_matches.append(match)


# In[12]:


Points1 = [(final_matches[i][0][0],final_matches[i][0][1]) for i in range(len(final_matches))]
Points2 = [(final_matches[i][1][0],final_matches[i][1][1]) for i in range(len(final_matches))]


# In[ ]:





# In[52]:


A = np.zeros((2 * len(Points1),6))
b = np.zeros(2 * len(Points2) )

j=1
for i in range(0,len(Points1)):
    A[2*i, :] = [Points1[i][0], Points1[i][1], 1, 0, 0, 0]
    A[j, :] = [0, 0, 0, Points1[i][0], Points1[i][1], 1]
    
    b[2*i] = Points2[i][0]
    
    b[j ] = Points2[i][1]
    
    j=j+2
      
   


# In[53]:


A = A[:80, :]
b = b[:80]


# In[54]:


np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)),A.T),b) 


# In[49]:


A


# In[27]:


b


# In[ ]:





# In[ ]:




