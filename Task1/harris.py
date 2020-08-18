"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Prateek Arora (u6742441)
"""

## Reference - https://stackoverflow.com/questions/3862225/implementing-a-harris-corner-detector #################################
###Some parts of the code are taken from the reference but whole code is referenced just to be on the safer side#########

import numpy as np
import cv2
import matplotlib.pyplot as plt

def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result

def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# Task: Compute the Harris Cornerness
def Harris_Corners(bw):
    
    # Parameters, add more if needed
    sigma = 2
    thresh = 0.01
    # Derivative masks
    dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    dy = dx.transpose()
    bw = np.array(bw * 255, dtype=int)
    # computer x and y derivatives of image
    Ix = conv2(bw, dx)
    Iy = conv2(bw, dy)
    
    g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma) ## fspecial function returns the gaussian lower pass filter  of size h which is returned the fspecial function with the standard deviation of sigma which is equal to 2.
    
    Iy2 = conv2(np.power(Iy, 2), g) # Over here, we convolute intensity matrix(Iy**2) with the gaussian filter formed
    Ix2 = conv2(np.power(Ix, 2), g) # Over here, we convolute intensity matrix(Ix**2) with the gaussian filter formed
    Ixy = conv2(Ix * Iy, g) # We convolute intensity matrix(Ix*Iy) with the gaussian filter formed
    
###############################################################################################################
    # Task: Compute Harris Corners
    height, width = bw.shape  ##Gets the shape of image
    result_image = np.zeros((height, width)) ## Assigning the zeros to an array of same size as image
    R = np.zeros((height, width)) ## Assigning the zeros to an array of same size as image
    max_value = 0
    

    ## Looping over the dimensions of image
    for i in range(height):
        for j in range(width):
            determinant = Ix2[i][j]*Iy2[i][j] - (Ixy[i][j]**2) ## Find determinant of using the intensity
            trace_matrix = Ix2[i][j] + Iy2[i][j] ## Finding the trace
            R[i][j] = determinant - 0.01 * (trace_matrix**2) ##Getting rhe corners for the matrix
            if R[i][j] > max_value:
                max_value = R[i][j] ## Saving the maximum value got
            
##################################################################################################################   
    
######################################################################
    # Task: Perform non-maximum suppression and
    #       thresholding, return the N corner points
    #       as an Nx2 matrix of x and y coordinates

    # Looping over the dimensions of the image
    for k in range(height - 1):
        for l in range(width - 1):
            if (R[k][l] > 0.01 * max_value and R[k][l] > R[k-1][l-1] and R[k][l] > R[k-1][l+1] and R[k][l] > R[k+1][l-1] and R[k][l] > R[k+1][l+1]):
                result_image[k][l] = 1 ##Assigning the values 1 if the above condition is satisfied
    return result_image
##################################################################

bw = plt.imread('Harris_2.jpg') # We use a grayscale image for the Harris corner
final_corner_matrix = Harris_Corners(bw) ## For 'Harris_2.jpg'
y_axis_point, x_axis_point = np.where(final_corner_matrix == 1) ## Finding the x and y axis of the point in the matrix where value is 1
plt.plot(x_axis_point, y_axis_point, 'r+') ## Plotting the points where the value is 1 in result_image matrix
plt.imshow(bw, cmap = 'gray') ##Plotting the image
plt.title("Harris_2")
plt.show() ## Displaying the image

img1 = plt.imread('Harris_1.jpg')
cvt = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
final_corner_matrix = Harris_Corners(cvt) ## For 'Harris_1.jpg'
y_axis_point1, x_axis_point1 = np.where(final_corner_matrix == 1) ## Finding the x and y axis of the point in the matrix where value is 1
plt.plot(x_axis_point1, y_axis_point1, 'r+') ## Plotting the points where the value is 1 in result_image matrix
plt.imshow(img1) ##Plotting the image
plt.title("Harris_1")
plt.show() ## Displaying the image

img2 = plt.imread('Harris_3.jpg')
cvt = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
final_corner_matrix = Harris_Corners(cvt) ## For 'Harris_3.jpg'
y_axis_point2, x_axis_point2 = np.where(final_corner_matrix == 1) ## Finding the x and y axis of the point in the matrix where value is 1
plt.plot(x_axis_point2, y_axis_point2, 'r+') ## Plotting the points where the value is 1 in result_image matrix
plt.imshow(img2) ##Plotting the image
plt.title("Harris_3")
plt.show() ## Displaying the image

img3 = plt.imread('Harris_4.jpg')
cvt = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
final_corner_matrix = Harris_Corners(cvt) ## For 'Harris_4.jpg'
y_axis_point3, x_axis_point3 = np.where(final_corner_matrix == 1) ## Finding the x and y axis of the point in the matrix where value is 1
plt.plot(x_axis_point3, y_axis_point3, 'r+') ## Plotting the points where the value is 1 in result_image matrix
plt.imshow(img3) ##Plotting the image
plt.title("Harris_4")
plt.show() ## Displaying the image

img_read = cv2.imread('Harris_3.jpg')
grayscale_image_inbuilt = cv2.cvtColor(img_read,cv2.COLOR_BGR2GRAY)

grayscale_image_inbuilt = np.float32(grayscale_image_inbuilt)
corners_distance = cv2.cornerHarris(grayscale_image_inbuilt,3,3,0.01)

distance = cv2.dilate(corners_distance,None)

img_read[distance>0.01*distance.max()]=[0,0,255]

fig_size = plt.figure(figsize=(14,7))
ax1 = fig_size.add_subplot(121)
ax2 = fig_size.add_subplot(122)

ax1.plot(x_axis_point2,y_axis_point2,'b+')
ax1.imshow(img2)
ax1.title.set_text("Result obtained from the function built from scratch")
ax2.imshow(img_read)
ax2.title.set_text("Result obtained from the inbuilt function")
plt.show()
############################################################################################3