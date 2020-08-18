import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

##### Q1
my_image_array = []

for a in glob.glob('Yale-FaceA/trainingset/*.png'):
    my_image = cv2.imread(a,0)  ## Reading the images from the training dataset
    my_image = np.asarray(my_image,dtype = 'uint8') ## Store the values in an array
    my_image_array.append(my_image) ## Append the values in a list
    
image_extracted = my_image_array[:][135:144] # These images are "my images" which are basically are not of the same size as other images

for i in range(9):
    image_extracted[i] = cv2.resize(image_extracted[i],(195,231)) ## Resizing the images
    
cv2.imwrite("Yale-FaceA/trainingset/subject16.confused.png",image_extracted[0])  ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.doubt.png",image_extracted[1])  ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.fine.png",image_extracted[2])   ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.happy.png",image_extracted[3])   ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.ignorance.png",image_extracted[4])   ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.laughing.png",image_extracted[5])  ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.shocked.png",image_extracted[6])  ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.smile.png",image_extracted[7])  ## Saving the images after resizing for the further usage.
cv2.imwrite("Yale-FaceA/trainingset/subject16.thinking.png",image_extracted[8])  ## Saving the images after resizing for the further usage.

##### Q2

### i)

image_array_formed = []
flattened = []

for a in glob.glob('Yale-FaceA/trainingset/*.png'):
    my_image = cv2.imread(a,0) # Reading the images from the training dataset
    my_image = np.asarray(my_image,dtype = 'uint8')
    image_array_formed.append(my_image) ## Append the values in the list
    
image_array_formed = image_array_formed[:][:135]    

for j in range(len(image_array_formed)):
    k = image_array_formed[j].flatten() ## Flattening of the image
    flattened.append(k) ## Appending the flattened images in a list
    
A_t_matrix = np.matrix(flattened) ## Converting the flattened image list to an matrix

### ii)

mean_matrix = np.mean(A_t_matrix.T, 1)  # Calculating mean of the matrix 
mean_array = np.asarray(mean_matrix, dtype = 'uint8')
mean_array = np.reshape(mean_array,(231,195)) # Reshapeing the image so as to plot the mean face
plt.title("Mean face")
plt.imshow(mean_array, cmap = "gray")
plt.show()

A_t_matrix = A_t_matrix.T - mean_matrix #Normalizing the matrix

Covariance_matrix = (A_t_matrix.T@A_t_matrix/(len(A_t_matrix) - 1))  ## Calculating the covariance of the matrix

u,v = np.linalg.eigh(Covariance_matrix) ## Calculating the eigenvector and eigenvalues

u = u.tolist()
u_index = sorted(range(len(u)), key=lambda k: u[k]) # Getting the index of the sorted array
u = sorted(u, reverse=True)
u = np.matrix(u)

v = v.tolist()
v.sort(key = lambda u_index: u_index) # Sorting the eigenvectors according to the index of the eigenvalues
v = np.asmatrix(v)
v = v[::-1]

eigen_faces = np.dot(A_t_matrix,v) # Dot product of the two matrix to make eigenface

eigen_faces = eigen_faces.T

### iii)

k = 10

### Plotting of the top k principal components

for i in range(k):
    new_im = eigen_faces[:][i]
    new_im = new_im.T
    new_im = np.reshape(new_im,(231,195))
    plt.imshow(new_im, cmap = "gray")
    plt.title(str(int(i)) + " " + "Principal Component")
    plt.show()

### iv) 

training_weights = A_t_matrix.T@eigen_faces[:10][:].T # Top 10 principal components is used for the training weights

testing_image_array_formed = []
testing_flattened = []

for i in glob.glob('Yale-FaceA/testset/*.png'):
    im = cv2.imread(i,0) # Reading the test image dataset
    im = np.asarray(im, dtype = 'uint8') ## Conversion of the matrix
    testing_image_array_formed.append(im) ## Appending the values in the list

## Conversion of test image to appropriate size#########    
my_test_image = testing_image_array_formed[:][10] ## Extracting "my image" from the dataset to change the size of the image

my_test_image = cv2.resize(my_test_image,(195,231)) ## Resizing the image

cv2.imwrite('Yale-FaceA/testset/subject16.normal.png',my_test_image) # Storing the image the test dataset
#################################################
testing_image_array_formed = testing_image_array_formed[:10] ## "My image" is not considered for testing because it is not required for this partof the question
    
for j in range(len(testing_image_array_formed)):
    k = testing_image_array_formed[j].flatten()  ## Flattening of the image
    testing_flattened.append(k) ## Appending the flattened image into a list
B_testing_matrix = np.matrix(testing_flattened)

Normalized_test = B_testing_matrix.T - mean_matrix # Normalization of the test image matrix

testing_weights = Normalized_test.T@eigen_faces[:10][:].T # Top 10 principal components is used for the testing weights

val = np.zeros((10,135))
diff_array = []

for i in range(10):
    for j in range(135):
        val[i][j] = np.linalg.norm(training_weights[:][j] - testing_weights[:][i]) # Calculating the norm for the image so as to the difference between the images and it is easier to see which one is closer
        
sort_index_array1 = np.zeros((10,3))

for b in range(10):
    sort_index_array = sorted(range(len(val[b][:])), key=lambda k: val[b][:][k]) # Sorting the array and extracting the index accordingly
    sort_index_array1[b][:] = sort_index_array[:3]

## Plotting of the images    
for l in range(10):
    for m in range(3):
        plt.title("Training image who is most similar with test image of index" + str(int(l)))
        plt.imshow(image_array_formed[int(sort_index_array1[:][l][m])], cmap = "gray")
        plt.show()
    plt.title("Testing Image of Index" + str(int(l)))
    plt.imshow(testing_image_array_formed[l], cmap = "gray")
    plt.show()

### v)

my_testing_flattened = []

my_testing_flattened = my_test_image.flatten() # Flattening fo the image
my_B_testing_matrix = np.matrix(my_testing_flattened) ## Storing the image in the the matrix

my_Normalized_test = my_B_testing_matrix.T - mean_matrix ## Normalizing the matri by subtracting the mean

my_image_testing_weights = my_Normalized_test.T@eigen_faces[:10][:].T # Getting testing weights using the images

val_my = np.zeros((135))

for j in range(135):
    val_my[j] = np.linalg.norm(training_weights[:][j] - my_image_testing_weights) # Calculating the norm for the image so as to the difference between the images and it is easier to see which one is closer
    
sort_index_array_for_my_image = sorted(range(len(val_my)), key=lambda k: val_my[k]) # Sorting the array and extracting the index accordingly

sort_index_array_for_my_image = sort_index_array_for_my_image[:3]

# Plotting the image closest to my image
for h in range(len(sort_index_array_for_my_image)):
    plt.imshow(image_array_formed[sort_index_array_for_my_image[h]],cmap = "gray")
    plt.title("Images similar to my images")
    plt.show()

plt.imshow(my_test_image, cmap = "gray")
plt.title("My Image")
plt.show()

### vi)

image_array_formed = []
flattened = []

for a in glob.glob('Yale-FaceA/trainingset/*.png'):
    my_image = cv2.imread(a,0) # Reading the images from the training dataset
    my_image = np.asarray(my_image,dtype = 'uint8')
    image_array_formed.append(my_image) ## Append the values in the list
    
for j in range(len(image_array_formed)):
    k = image_array_formed[j].flatten() ## Flattening of the image
    flattened.append(k) ## Appending the flattened images in a list
    
A_t_matrix = np.matrix(flattened) ## Converting the flattened image list to an matrix

A_t_matrix = A_t_matrix.T - mean_matrix #Normalizing the matrix

Covariance_matrix = (A_t_matrix.T@A_t_matrix/(len(A_t_matrix) - 1))  ## Calculating the covariance of the matrix

u,v = np.linalg.eigh(Covariance_matrix) ## Calculating the eigenvector and eigenvalues

u = u.tolist()
u_index = sorted(range(len(u)), key=lambda k: u[k]) # Getting the index of the sorted array
u = sorted(u, reverse=True)
u = np.matrix(u)

v = v.tolist()
v.sort(key = lambda u_index: u_index) # Sorting the eigenvectors according to the index of the eigenvalues
v = np.asmatrix(v)
v = v[::-1]

eigen_faces = np.dot(A_t_matrix,v) # Dot product of the two matrix to make eigenface

eigen_faces = eigen_faces.T

training_weights = A_t_matrix.T@eigen_faces[:10][:].T # Top 10 principal components is used for the training weights

testing_image_array_formed = []
testing_flattened = []

for i in glob.glob('Yale-FaceA/testset/*.png'):
    im = cv2.imread(i,0) # Reading the test image dataset
    im = np.asarray(im, dtype = 'uint8') ## Conversion of the matrix
    testing_image_array_formed.append(im) ## Appending the values in the list
    
for j in range(len(testing_image_array_formed)):
    k = testing_image_array_formed[j].flatten()  ## Flattening of the image
    testing_flattened.append(k) ## Appending the flattened image into a list
B_testing_matrix = np.matrix(testing_flattened)

Normalized_test = B_testing_matrix.T - mean_matrix # Normalization of the test image matrix

testing_weights = Normalized_test.T@eigen_faces[:10][:].T # Top 10 principal components is used for the testing weights

val = np.zeros((11,144))
diff_array = []

for i in range(11):
    for j in range(144):
        val[i][j] = np.linalg.norm(training_weights[:][j] - testing_weights[:][i]) # Calculating the norm for the image so as to the difference between the images and it is easier to see which one is closer
        
sort_index_array1 = np.zeros((11,3))

for b in range(11):
    sort_index_array = sorted(range(len(val[b][:])), key=lambda k: val[b][:][k])
    sort_index_array1[b][:] = sort_index_array[:3]
    
for m in range(3):
    plt.title("Training image with most similarity with test image of index" + str(int(10)))
    plt.imshow(image_array_formed[int(sort_index_array1[:][10][m])], cmap = "gray")
    plt.show()
plt.title("Testing Image of Index" + str(int(10)))
plt.imshow(testing_image_array_formed[10], cmap = "gray")
plt.show()