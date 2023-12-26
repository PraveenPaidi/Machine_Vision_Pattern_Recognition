#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np 
import time

start_time = time.time()
# Input image 
image= np.array([[1, 2, 3], [4, 5, 6], [7, 8 ,9]])
image_rows, image_columns = image.shape

# vectorize the image 
I = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).T

# kernel input 
kernel = np.array ([[1, 0 , -1], [1, 0, -1], [1, 0, -1]])
kernel_rows, kernel_columns = kernel.shape

# output dimensions input 
output_rows, output_columns= (5, 5)

# pad width calculation
pad_width = int((output_rows - image_rows + 2)/2)  # Number of rows/columns to pad on each side
constant_value = 0 

# Pad the matrix with zeros
image_padded = np.pad(image, pad_width, mode='constant', constant_values=constant_value)
padded_rows,padded_columns = np.shape(image_padded)

# H matrix initialization
H= np.zeros((25, 9), dtype=int)
print('Padded image is\n', image_padded)


# In[18]:


def create_position_mapping(matrix1, matrix2):
    # Get the dimensions of the matrices
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    # Calculate the padding (if any)
    row_padding = (rows2 - rows1) // 2
    col_padding = (cols2 - cols1) // 2

    # Initialize an empty dictionary to store the position mappings
    position_mappings = {}

    # Iterate through the positions in the smaller matrix (matrix1)
    for i in range(rows1):
        for j in range(cols1):
            position1 = (i, j)  # Position in matrix1
            position2 = (i + row_padding, j + col_padding)  # Position in matrix2
            position_mappings[position1] = position2

    return position_mappings

# Create the position mapping
position_mappings = create_position_mapping(image, image_padded)

# reversing the mapping 
reverse_position_mappings = {v: k for k, v in position_mappings.items()}


# In[19]:



count=0

for i in range(output_rows):
    
    for j in range(output_columns):
        
        # creating the sliding window kernel 
        temp_mat = np.array([[image_padded[i][j], image_padded[i][j+1], image_padded[i][j+2]],
                             [image_padded[i+1][j], image_padded[i+1][j+1], image_padded[i+1][j+2]],
                             [image_padded[i+2][j], image_padded[i+2][j+1], image_padded[i+2][j+2]]])
        
        # convolution matrix
        H_mat = np.zeros((3, 3))
        
        # for image coordniate location
        pos1= i
        
        for a in range(kernel_rows):
            
            #for image coordniate location
            pos2 = j
            
            for b in range(kernel_columns):
                 
                if temp_mat[a][b] !=0:
                    # to get the position of image coordinate                    
                    reverse_mapping = reverse_position_mappings.get((pos1,pos2))
                    
                    if reverse_mapping is not None:
                         H_mat[reverse_mapping[0]][reverse_mapping[1]]= kernel[a][b]
                pos2 = pos2+1
            pos1= pos1+1
                    
        
        # reshaping for the multiplication
        H[count,:]= np.resize(H_mat, (1,9))
        
        count+=1

print(H)


# In[20]:



def conv2dmatrix(I, H):
#     start_time = time.time()
    convolved= np.dot(H, I)
    end_time = time.time()
    
    timer = end_time - start_time
    
    return convolved, timer
    


# In[21]:


output, time = conv2dmatrix(I, H )
output=output.reshape(5,5)
print(time)


# In[25]:


print('Convolve image is\n', output)


# In[ ]:




