#!/usr/bin/env python
# coding: utf-8

# In[79]:


import cv2
import numpy as np
import copy
import sys

def harris_corner_detector(image, k=0.04, threshold=1500000000000000):   #10000000000000000
    
        
    ''' 1st step : providing image to calculate the gradients in x and y direction.
    passing sobel kernel of size 5 to calculate and cv2.CV_64F is the output type'''
    
    I_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    I_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    ''' 2nd step , taking the mean out '''
    mean_I_x = np.mean(I_x)
    I_x -= mean_I_x
    mean_I_y = np.mean(I_y)
    I_y -= mean_I_y

    ''' 3rd step , computing the covariance matrix '''
    # computing the covariance matrix
    I_x2 = I_x ** 2
    I_y2 = I_y ** 2
    I_xy = I_x * I_y

    
    ''' 4th step calculting eigen values and eigen vectors for M matrix '''
    height, width= image.shape
    window_size = 5

    offset = window_size // 2

    R = np.zeros_like(image, dtype=np.float32)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_I_x2 = I_x2[y - offset:y + offset + 1, x - offset:x + offset + 1]
            window_I_y2 = I_y2[y - offset:y + offset + 1, x - offset:x + offset + 1]
            window_I_xy = I_xy[y - offset:y + offset + 1, x - offset:x + offset + 1]

            M = np.array([[np.sum(window_I_x2), np.sum(window_I_xy)],
                          [np.sum(window_I_xy), np.sum(window_I_y2)]], dtype=np.float32)
            
            eigvals = np.linalg.eigvals(M)

            R[y, x] = min(eigvals) * max(eigvals) - k * ((min(eigvals) + max(eigvals))**2)

    '''5th step corners finding my applying the thresshold as 1000 '''
    corners = (R > threshold).nonzero()
    return corners, R


def ANMS(img, img_h, n_best, coords):
    # list length of corners
    ''' img being the original image, img_h being the liikelihood matrix of the harris corners'''
    num = len(coords)
    inf = sys.maxsize
    r = inf * np.ones((num,3))
    ED = 0
    
    for i in range(num):
        for j in range(num):
            # taking one corner at a time.
            x_i = coords[i][0]            
            y_i = coords[i][1]            
            
            # comparing with all other corners.
            neighbours_x = coords[j][0]  
            neighbours_y = coords[j][1]
            
            
            ''' taking the local maxima likelihood'''
            if img_h[y_i,x_i] > img_h[neighbours_y,neighbours_x]:
                ED = (neighbours_x - x_i)**2 + (neighbours_y - y_i)**2

            if ED < r[i,0]:
                r[i,0] = ED
                r[i,1] = x_i
                r[i,2] = y_i

    arr = r[:,0]
    #We get the index of biggest that is the reason of -ve sign(Descending order index)
    feature_sorting = np.argsort(-arr)  
    feature_cord = r[feature_sorting]
    #We also can is find min of(n_best, num_of_feature_cordinates we got)
    Nbest_corners = feature_cord[:n_best,:]   
    
    return Nbest_corners



def feature_descriptors(img, corners,n_best,patch_size):
    ''' sending the corner list and the image and the corner coordinates'''
    n_descriptors = []
    x = corners[:,1]
    y = corners[:,2]

    for i in range(n_best):
        y_i = x[i]         
        x_i = y[i]
        gray = copy.deepcopy(img)
        
        #pad the image by 40 on all sides
        gray = np.pad(img, ((patch_size,patch_size), (patch_size,patch_size)), mode='constant', constant_values=0)
        x_start = int(x_i + patch_size/2)
        y_start = int(y_i + patch_size/2)

        # creating feature descriptor 40x40 descriptor of one point
        descriptor = gray[x_start:x_start+patch_size, y_start:y_start+patch_size] 
        
        # Applying gaussian blur on the descriptor
        descriptor = cv2.GaussianBlur(descriptor, (7,7), cv2.BORDER_DEFAULT) 
               
        # Sub sampling to 8x8   
        sub =5
        descriptor = descriptor[::sub,::sub]  
               
        descriptor1 = descriptor.reshape((64,1))
                
        std = descriptor1.std()
        
        if std< 0.00000001:
            std = 0.000001

        # to remove illumination invariance.
        descriptor_standard = (descriptor1 - descriptor1.mean())/std
            
        n_descriptors.append(descriptor_standard)

    return n_descriptors



def feature_matching(Descriptors_image1, Descriptors_image2, corners1,corners2, match_ratio):
    f1 = Descriptors_image1
    f2 = Descriptors_image2
    
    matched_pairs = []
    for i in range(0, len(f1)):
        sqr_diff = []
        for j in range(0, len(f2)):
            # comparing each corner to every other corner in next image by taking the difference between the descriptors
            diff = np.sum((f1[i] - f2[j])**2)
            sqr_diff.append(diff)
        # converting  into array  
        sqr_diff = np.array(sqr_diff)
        diff_sort = np.argsort(sqr_diff)
        sqr_diff_sorted = sqr_diff[diff_sort]
        
        if (sqr_diff_sorted[1])==0:
            sqr_diff_sorted[1] = 0.00001
            
        # applying lowe's algorithm to check the matching 
        ratio = sqr_diff_sorted[0]/(sqr_diff_sorted[1])
        
        if ratio < match_ratio :
            matched_pairs.append((corners1[i,1:3], corners2[diff_sort[0],1:3]))	

    return matched_pairs

def dot_product(h_mat, keypoint):
	keypoint = np.expand_dims(keypoint, 1)
	keypoint = np.vstack([keypoint, 1])
	product = np.dot(h_mat, keypoint)
	if product[2]!=0:
		product = product/product[2]
	else:
		product = product/0.000001
	# print(product)
	return product[0:2,:]

def homography(point1, point2):
    h_matrix  = cv2.getPerspectiveTransform(np.float32(point1), np.float32(point2))
    return h_matrix

def ransac(matched_pairs, threshold):

    inliers = []   #to store ssd's and corresponding homography matrices
    COUNT = []
    for i in range(1000):    #Nmax iterations

        keypoints_1 = [x[0] for x in matched_pairs]
        keypoints_2 = [x[1] for x in matched_pairs]
        length = len(keypoints_1)

        randomlist = random.sample(range(0, length), 4)
        points_1 = [keypoints_1[idx] for idx in randomlist]
        points_2 = [keypoints_2[idx] for idx in randomlist]

        h_matrix = homography(points_1, points_2)
        # print(h_matrix)
        points = []
        count_inliers = 0
        for i in range(length):
            a = (np.array(keypoints_2[i]))
            # ssd = np.sum((np.expand_dims(np.array(keypoints_2[i]), 1) - dot_product(h_matrix, keypoints_1[i]))**2)
            ssd = np.linalg.norm(np.expand_dims(np.array(keypoints_2[i]), 1) - dot_product(h_matrix, keypoints_1[i]))
            # print("ssd",ssd)
            if ssd < threshold:
                count_inliers += 1
                points.append((keypoints_1[i], keypoints_2[i]))
        COUNT.append(-count_inliers)
        inliers.append((h_matrix, points))
    max_count_idx = np.argsort(COUNT)
    max_count_idx = max_count_idx[0]
    final_matched_pairs = inliers[max_count_idx][1]
    # print("Matched pairs", len(final_matched_pairs))

    pts_1 = [x[0] for x in final_matched_pairs]
    pts_2 = [x[1] for x in final_matched_pairs]
    h_final_matrix, status = cv2.findHomography(np.float32(pts_1),np.float32(pts_2))
    # print(h_final_matrix)
    return h_final_matrix, final_matched_pairs


# In[80]:


# Load an image

image1=cv2.imread(r'..\Prob 5\1.jpg',cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r'..\Prob 5\2.jpg', cv2.IMREAD_GRAYSCALE)

img1 = copy.deepcopy(image1)
img2 = copy.deepcopy(image2)


# In[81]:


# Detect corners using harris corners 
corners1,R1 = harris_corner_detector(img1)
corners2,R2 = harris_corner_detector(img2)

corner_list1=[]
corner_list2=[]

    
image1=cv2.imread(r'..\Prob 5\1.jpg')
image2 = cv2.imread(r'..\Prob 5\2.jpg')

# Display or save the result
cv2.imwrite('Harris Corners Orig.jpg', image1)
cv2.imwrite('Harris Corners Roatated.jpg', image2)

# Draw corners on the image
for x, y in zip(corners1[1], corners1[0]):
    cv2.circle(img1, (x, y), 2, 100, -1)
    corner_list1.append((x,y))
    
for x, y in zip(corners2[1], corners2[0]):
    cv2.circle(img2, (x, y), 2, 100, -1)
    corner_list2.append((x,y))


# In[82]:


len(corner_list1)


# In[ ]:





# In[ ]:





# In[83]:


image1=cv2.imread(r'..\Prob 5\1.jpg')
image2 = cv2.imread(r'..\Prob 5\2.jpg')

img1 = copy.deepcopy(image1)
img2 = copy.deepcopy(image2)

# Adaptive Non Max Supression
n_best =1000
Best_corners1 = ANMS(img1, R1, n_best, corner_list1 )

for i in range(len(Best_corners1)):
    cv2.circle(img1, (int(Best_corners1[i][1]),int(Best_corners1[i][2])), 3, 100, -1)
cv2.imwrite('Anms1.jpg', img1)

Best_corners2 = ANMS(img2, R2, n_best, corner_list2 )

for i in range(len(Best_corners2)):
    cv2.circle(img2, (int(Best_corners2[i][1]),int(Best_corners2[i][2])), 3, 100, -1)
cv2.imwrite("anms2.png",img2)


# In[ ]:





# In[84]:


image1=cv2.imread(r'..\Prob 5\1.jpg',cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r'..\Prob 5\2.jpg', cv2.IMREAD_GRAYSCALE)

img1 = copy.deepcopy(image1)
img2 = copy.deepcopy(image2)

# applying SIFT to get the feature descriptors 
patch_size= 40
Descriptors_image1= feature_descriptors(img1, Best_corners1, n_best, patch_size)
Descriptors_image2= feature_descriptors(img2, Best_corners2, n_best, patch_size)

matched_pairs= feature_matching(Descriptors_image1, Descriptors_image2, Best_corners1,Best_corners2, match_ratio=0.6)


# In[85]:


matched_pairs


# In[142]:


import random
final_h_mat, final_matched = ransac(matched_pairs, threshold=3)


# In[143]:


final_matched = [(a.astype(int), b.astype(int)) for a, b in final_matched]
final_matched
img1 = copy.deepcopy(image1)
img2 = copy.deepcopy(image2)
def keypoint(points):
	kp1 = []
	for i in range(len(points)):
		kp1.append(cv2.KeyPoint(int(points[i][0]), int(points[i][1]), 3))
	return kp1

def matches(points):
	m = []
	for i in range(len(points)):
		m.append(cv2.DMatch(int(points[i][0]), int(points[i][1]), 2))
	return m


# In[144]:


def draw_matches(img1,img2, matched_pairs):
	key_points_1 = [x[0] for x in matched_pairs]
	keypoints1 = keypoint(key_points_1)
	key_points_2 = [x[1] for x in matched_pairs]
	keypoints2 = keypoint(key_points_2)
	matched_pairs_idx = [(i,i) for i,j in enumerate(matched_pairs)]
	matches1to2 = matches(matched_pairs_idx)
	out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, None, flags =2)
	cv2.imwrite('out.jpg',out)


# In[145]:


draw_matches(img1,img2, final_matched)


# In[146]:


matched_pairs_idx = [(i,i) for i,j in enumerate(final_matched)]
matches1to2 = matches(matched_pairs_idx)
keypoints1 = [x[0] for x in final_matched]
keypoints2 = [x[1] for x in final_matched]

def keypoint(points):
	kp1 = []
	for i in range(len(points)):
		kp1.append(cv2.KeyPoint(int(points[i][0]), int(points[i][1]), 3))
	return kp1

keypoints1= keypoint(keypoints1)
keypoints2= keypoint(keypoints2)

out = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches1to2, None, flags =2)
cv2.imwrite('outliers.jpg',out)


# In[147]:


def warpTwoImages(img1, img2, H):

	img1 = img1
	img2 = img2
	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel())
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel())
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
	result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin), flags = cv2.INTER_LINEAR)
	result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2

	return result


# In[148]:


image1=cv2.imread(r'..\Prob 5\1.jpg')
image2 = cv2.imread(r'..\Prob 5\2.jpg')

img1 = copy.deepcopy(image1)
img2 = copy.deepcopy(image2)

warped = warpTwoImages(img1, img2, final_h_mat)


# In[149]:


cv2.imwrite('warp.jpg',warped)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


image1=cv2.imread(r'..\Prob 5\1.jpg')
image2 = cv2.imread(r'..\Prob 5\2.jpg')

re1= cv2.resize(image1, (506*2,672*2 ))

re2= cv2.resize(image2, (506*2,672*2 ))

cv2.imwrite('1.jpg', re1)
cv2.imwrite('2.jpg', re2)


# In[ ]:




