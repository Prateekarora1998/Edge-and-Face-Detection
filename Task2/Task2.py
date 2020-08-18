import numpy as np
import matplotlib.pyplot as plt
import cv2

############ Q1------------#############################
def my_kmeans(matrix,k,given_iterations):
    
    clusters = np.zeros(matrix.shape[0]) #Assign the clusters of each data point
    id1 = np.random.randint(matrix.shape[0], size=k) ## Selecting random index from the data points given
    
    distances_from_centroids = np.zeros((matrix.shape[0],k)) ## Array Initialization to store the distances from centroids
    new_centroid = matrix[id1,:] # Initialization of centroid
    max_iterations = 0 # Number of iterarions
    
    while max_iterations <= given_iterations:
        max_iterations += 1
        
        clusters = find_clusters(distances_from_centroids,matrix,new_centroid,k) # This functions helps in finding the clusters
            
        for j in range(k):
            new_centroid[j] = np.mean(matrix[clusters == j], axis=0) #Finding new centroids by taking mean of points in cluster
            
    return new_centroid , clusters

def find_clusters(dist,matrix,centroid,k):
    
    for i in range(k):
        
        dist[:,i] = np.linalg.norm(matrix - centroid[i],axis = 1) # Geting the distance from the centroid to the points given
        clusters = np.argmin(dist,axis = 1) # Extract the minimum distance and assigning the point to that particular cluster
        
    return clusters

new_array = 5 * np.random.randn(200,2) + 5 # Making an array elements which represents gaussian distribution
matrix = np.asmatrix(new_array)
k = 3 # Number of clusters

new_centroid,clusters = my_kmeans(matrix,k,given_iterations=1000) ## Running the function and getting the centroids and clusters
colors=['green', 'blue', 'yellow'] ## Choosing colors for the clusters formed 

for i in range(200):
    plt.scatter(matrix[i, 0], matrix[i,1], s=5, color = colors[int(clusters[i])]) ## Plotting cluster points with colors
plt.plot(new_centroid[:,0], new_centroid[:,1], 'r+') # Plotting final centroids using red cross sign.
plt.show()
##########################################################################

################### Q2 -------------####################################

### K means with x and y coordinates
image = cv2.imread('mandm.png',1)  # Reading the image
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Converting the image's color space from RGB to LAB
image_5d_array = []

m_mandm = lab.shape[0] 
n_mandm = lab.shape[1]

for i in range(m_mandm):
    for j in range(n_mandm):
        image_5d_array.append([lab[i,j,0],lab[i,j,1],lab[i,j,2],i,j]) ## Making of an 5d list
        
image_5d_array = np.asarray(image_5d_array) # Converting that list to the array
centers, labels = my_kmeans(image_5d_array,10,100)

image_1 = centers[labels]
image_1 = image_1.reshape(394,451,5)

plt.imshow(image_1[:,:,:3])
plt.title("Mandm image segmentation using K means with x and y coordinates")
plt.show()

image = cv2.imread('peppers.png',1) # Reading the image
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Converting the image's color space from RGB to LAB
image_5d_array = []

m_peppers = lab.shape[0]
n_peppers = lab.shape[1]

for i in range(m_peppers):
    for j in range(n_peppers):
        image_5d_array.append([lab[i,j,0],lab[i,j,1],lab[i,j,2],i,j]) ## Making of an 5d list

image_5d_array = np.asarray(image_5d_array) # Converting that list to the array
centers, labels = my_kmeans(image_5d_array,10,100)

image_1 = centers[labels]
image_1 = image_1.reshape(384,512,5)

plt.imshow(image_1[:,:,:3])
plt.title("Peppers image segmentation using K means with x and y coordinates")
plt.show()

### Kmeans without x and y coordinates

image = cv2.imread('mandm.png',1) # Reading the image
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Converting the image's color space from RGB to LAB
image_3d_array = []

m_mandm = lab.shape[0]
n_mandm = lab.shape[1]

for i in range(m_mandm):
    for j in range(n_mandm):
        image_3d_array.append([lab[i,j,0],lab[i,j,1],lab[i,j,2]]) ## Making of an 5d list
        
image_3d_array = np.asarray(image_3d_array) # Converting that list to the array
centers, labels = my_kmeans(image_3d_array,10,100)

image_1 = centers[labels]
image_1 = image_1.reshape(394,451,3)

plt.imshow(image_1)
plt.title("Mandm image segmentation using K means without using x and y coordinates")
plt.show()

image = cv2.imread('peppers.png',1) # Reading the image
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Converting the image's color space from RGB to LAB
image_3d_array = []

m_peppers = lab.shape[0]
n_peppers = lab.shape[1]

for i in range(m_peppers):
    for j in range(n_peppers):
        image_3d_array.append([lab[i,j,0],lab[i,j,1],lab[i,j,2]]) ## Making of an 5d list
        
image_3d_array = np.asarray(image_3d_array) # Converting that list to the array
centers, labels = my_kmeans(image_3d_array,10,100)

image_1 = centers[labels]
image_1 = image_1.reshape(384,512,3)

plt.imshow(image_1)
plt.title("Peppers image segmentation using K means without using x and y coordinates")
plt.show()

