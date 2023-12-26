#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from matplotlib import pyplot as plt
import numpy as np

path= r'C:\Users\prave\Downloads\HW 1\Prob4\Image4.jpg'
image= cv2.imread(path)
                  
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filename = 'grayimage3.jpg'
cv2.imwrite(filename, grey)
                  
def fourier(image, i):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Appplying fourier transform function 
    f = np.fft.fft2(gray)

    # Dc is at the top left corner , to shift that dc from there to center
    fshift = np.fft.fftshift(f)
    
    # converting frquency transform to magnitude transform
    magnitude_spectrum = 20*np.log1p(np.abs(fshift))
    
    magnitude_spectrum = ((magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)

    # phase transform
    phase_spectrum = np.angle(fshift)
    
    cv2.imwrite("Magnitude" + str(i)+ ".jpg", magnitude_spectrum)
    
    return magnitude_spectrum, fshift
    
# mag1, f1= fourier(image, 1)


# In[2]:


# before LPF
mag2, f2= fourier(image, 2)


rows, cols = grey.shape
crow, ccol = rows // 2, cols // 2  # Center of the image

# Create a Gaussian filter
sigma = 30  # Adjust the standard deviation as needed
x = np.arange(cols) - ccol
y = np.arange(rows) - crow
X, Y = np.meshgrid(x, y)
mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
mask = mask / mask.max()  # Normalize to [0, 1]


f_transform_shifted = f2 * mask

magnitude_spectrum = 20*np.log1p(np.abs(f_transform_shifted))
    
magnitude_spectrum = ((magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)

cv2.imwrite("Magnitudelowpass" + ".jpg", magnitude_spectrum)

f_transform_shifted = np.fft.ifftshift(f_transform_shifted)
img_back = np.fft.ifft2(f_transform_shifted)

img_back = np.maximum(0, np.minimum(img_back, 255))

cv2.imwrite("Imagelowpassgreyscale" + ".jpg", img_back.astype(np .uint8))


# In[3]:


rows, cols = grey.shape
crow, ccol = rows // 2, cols // 2  # Center of the image

# Create a circular mask with high values in the center
mask = np.ones((rows, cols), np.uint8)
r = 30  # Radius of the circular mask
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0  # Low values towards the center


f_transform_shifted = f2 * mask

magnitude_spectrum = 20*np.log1p(np.abs(f_transform_shifted))
    
magnitude_spectrum = ((magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)

cv2.imwrite("Magnitudehighpass" + ".jpg", magnitude_spectrum)

f_transform_shifted = np.fft.ifftshift(f_transform_shifted)
img_back = np.fft.ifft2(f_transform_shifted)

img_back = np.maximum(0, np.minimum(img_back, 255))

cv2.imwrite("Imagehighpassgreyscale" + ".jpg", img_back.astype(np .uint8))


# In[4]:


rows, cols = grey.shape
crow, ccol = rows // 2, cols // 2  # Center of the image

# Create first Gaussian filter (high-pass)
sigma_high = 60
x = np.arange(cols) - ccol
y = np.arange(rows) - crow
X, Y = np.meshgrid(x, y)
high_pass_mask = np.exp(-(X**2 + Y**2) / (2 * sigma_high**2))
high_pass_mask = high_pass_mask / high_pass_mask.max()  # Normalize to [0, 1]

# Create second Gaussian filter (low-pass)
sigma_low = 50
low_pass_mask = np.exp(-(X**2 + Y**2) / (2 * sigma_low**2))
low_pass_mask = low_pass_mask / low_pass_mask.max()  # Normalize to [0, 1]

f_transform_high = f2 * high_pass_mask
f_transform_low = f2 * low_pass_mask

f_transform_dog = f_transform_high - f_transform_low

f_transform_shifted= f_transform_dog

magnitude_spectrum = 20*np.log1p(np.abs(f_transform_shifted))
    
magnitude_spectrum = ((magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)

cv2.imwrite("MagnitudeDOG" + ".jpg", magnitude_spectrum)


f_transform_inverse = np.fft.ifftshift(f_transform_dog)
img_back = np.fft.ifft2(f_transform_inverse)

img_back = np.maximum(0, np.minimum(img_back, 255))

cv2.imwrite("ImageDOGgreyscale" + ".jpg", img_back.astype(np .uint8))


# In[5]:


path = r'C:\Users\prave\Downloads\HW 1\Prob4\bug2.jpg'
path2 = r'C:\Users\prave\Downloads\HW 1\Prob4\bug3.jpg'
firstimage= cv2.imread(path)
secondimage=cv2.imread(path2)


# In[6]:



# Define the size of the kernel
kernel_size = 100

# Create a low-pass kernel (averaging filter)
low_pass_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

# Create a high-pass kernel by subtracting low-pass kernel from the identity kernel
identity_kernel = np.zeros((kernel_size, kernel_size), np.float32)
identity_kernel[kernel_size//2, kernel_size//2] = 1
high_pass_kernel = identity_kernel - low_pass_kernel

# Load the image
image = cv2.imread('input_image.jpg', cv2.IMREAD_COLOR)

# Apply the low-pass filter
low_pass_filtered = cv2.filter2D(firstimage, -1, low_pass_kernel)

# Apply the high-pass filter
high_pass_filtered = cv2.filter2D(secondimage, -1, high_pass_kernel)


# In[7]:


result= low_pass_filtered+high_pass_filtered


# In[8]:


cv2.imwrite('result.jpg', result)


# In[9]:


path = r'C:\Users\prave\Downloads\HW 1\Prob4\bug2.jpg'
path2 = r'C:\Users\prave\Downloads\HW 1\Prob4\bug3.jpg'
firstimage= cv2.imread(path)
secondimage=cv2.imread(path2)

mag10, f10=fourier(firstimage, 10)
mag11, f11=fourier(secondimage, 11)

grey = cv2.cvtColor(firstimage, cv2.COLOR_BGR2GRAY)


rows, cols = grey.shape
crow, ccol = rows // 2, cols // 2  # Center of the image

# Create a Gaussian filter
sigma = 20  # Adjust the standard deviation as needed
x = np.arange(cols) - ccol
y = np.arange(rows) - crow
X, Y = np.meshgrid(x, y)
mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
mask = mask / mask.max()  # Normalize to [0, 1]


f_transform_shifted = f10 * mask

# magnitude_spectrum = 20*np.log1p(np.abs(f_transform_shifted))


###########################################

mask2= 1- mask

f_transform_shifted1 = f11 * mask2

# magnitude_spectrum1 = 20*np.log1p(np.abs(f_transform_shifted))



lolo= f_transform_shifted+f_transform_shifted1


f_transform_inverse = np.fft.ifftshift(lolo)
img_back = np.fft.ifft2(f_transform_inverse)

img_back = np.maximum(0, np.minimum(img_back, 255))

cv2.imwrite("reeeeee" + ".jpg", img_back.astype(np .uint8))


# In[11]:


import cv2
from matplotlib import pyplot as plt
import numpy as np
path = r'C:\Users\prave\Downloads\HW 1\Prob4\1.jpg'
# path2 = r'C:\Users\prave\Downloads\HW 1\Prob4\12.jpg'
firstimage= cv2.imread(path)
# secondimage=cv2.imread(path2)

mag10, f10=fourier(firstimage, 15)
# mag11, f11=fourier(secondimage, 16)


# In[16]:


def create_diagonal_bandpass_filter(rows, cols, center, radius):
    # Generate a frequency grid
    freq_rows = np.fft.fftfreq(rows)
    freq_cols = np.fft.fftfreq(cols)
    freq_radius = np.sqrt(freq_rows[:, np.newaxis]**2 + freq_cols**2)
    
    # Create the band-pass filter
    bandpass_filter = np.logical_and(freq_radius >= center - radius, freq_radius <= center + radius).astype(float)
    
    return bandpass_filter

def apply_bandpass_filter(image, center, radius):
    # Apply FFT to the image
    f_transform = np.fft.fft2(image)
    
    # Create the band-pass filter
    rows, cols = image.shape
    bandpass_filter = create_diagonal_bandpass_filter(rows, cols, center, radius)
    
    # Apply the filter
    f_transform = f_transform * bandpass_filter
    
    
    magnitude_spectrum = 20*np.log1p(np.abs(f_transform))
    
    magnitude_spectrum = ((magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)

    cv2.imwrite("MagnitudeDiag" + ".jpg", magnitude_spectrum)

    
    # Inverse FFT
    result = np.fft.ifft2(f_transform).real
    
    return result

# Load an image
image = cv2.imread(r'C:\Users\prave\Downloads\HW 1\Prob4\Image4.jpg', cv2.IMREAD_GRAYSCALE)

# Define the center and radius of the band-pass filter
center = 0.25  # Adjust this value to control the band
radius = 0.25 # Adjust this value to control the width of the band

# Apply the band-pass filter
filtered_image = apply_bandpass_filter(image, center, radius)

cv2.imwrite('Diag Image.jpg', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


import cv2
import numpy as np

# Define the function to apply the band-pass filter
def apply_bandpass_filter(image, center, radius):
    rows, cols = image.shape
    crow, ccol = int(rows * center), int(cols * center)

    # Create a mask with high values in the band
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-int(rows*radius):crow+int(rows*radius), ccol-int(cols*radius):ccol+int(cols*radius)] = 0

    # Apply the Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Apply the mask
    fshift = fshift * mask
    
    magnitude_spectrum = 20*np.log1p(np.abs(fshift))
    
    magnitude_spectrum = ((magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)

    cv2.imwrite("MagnitudeDiag" + ".jpg", magnitude_spectrum)

    # Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back.astype(np.uint8)

# Load an image
image = cv2.imread(r'C:\Users\prave\Downloads\Image4.jpg', cv2.IMREAD_GRAYSCALE)

# Define the center and radius of the band-pass filter
center = 0.25  # Adjust this value to control the band
radius = 0.25  # Adjust this value to control the width of the band

# Apply the band-pass filter
filtered_image = apply_bandpass_filter(image, center, radius)

# Display the original and filtered images
# cv2.imshow('Original Image', image)
# cv2.imshow('Filtered Image', filtered_image)
cv2.imwrite('Diag Image.jpg', filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[17]:


import cv2
import numpy as np

def apply_annulus_filter(image, center, inner_radius, outer_radius):
    rows, cols = image.shape
    crow, ccol = int(rows * center), int(cols * center)

    # Apply the Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Create a mask with low values inside the annulus ring
    mask = np.ones((rows, cols), np.uint8)
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask_area = x*x + y*y >= (int(rows*inner_radius))**2
    mask_area &= x*x + y*y <= (int(rows*outer_radius))**2
    mask[mask_area] = 0

    # Apply the mask
    fshift = fshift * mask
    
    magnitude_spectrum = 20*np.log1p(np.abs(fshift))
    
    magnitude_spectrum = ((magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)

    cv2.imwrite("MagnitudeDiag" + ".jpg", magnitude_spectrum)

    # Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back.astype(np.uint8)

# Load an image
image = cv2.imread(r'C:\Users\prave\Downloads\Image4.jpg', cv2.IMREAD_GRAYSCALE)

# Define the center and radii of the annulus ring
center = 0.5  # Adjust this value to control the position of the ring
inner_radius = 0.2  # Adjust this value to control the inner radius of the ring
outer_radius = 0.4  # Adjust this value to control the outer radius of the ring

# Apply the annulus filter
filtered_image = apply_annulus_filter(image, center, inner_radius, outer_radius)

# # Display the original and filtered images
# cv2.imshow('Original Image', image)
cv2.imwrite("Filtered Image"+ ".jpg", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




