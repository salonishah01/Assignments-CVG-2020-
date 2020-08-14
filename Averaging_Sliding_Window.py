#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Image averaging is a digital image processing technique that is often employed to enhance video images 
#that have been corrupted by random noise. The algorithm operates by computing an average or arithmetic mean
#of the intensity values for each pixel position in a set of captured images from the same scene or viewfield.


# In[2]:


#Import required libraries
import cv2
import numpy as np
    


# In[3]:


def averaging(img):
    
    patch = [[1,1,1],[1,1,1],[1,1,1]]  #create a patch of required size.( Here I considered 3*3)
    patch = np.array(patch)   # convert the patch to an array

    patch_size = [3,3]
    img_shape = img.shape

    output_matrix = np.zeros([img_shape[0]-2, img_shape[1]-2])  #If image is of size m*n and patch p*q the resultanat of convoluton is of size m-p+1 * n-q+1
    for i in range(img_shape[0]-2):
        for j in range(img_shape[1]-2):
            output = np.zeros(patch_size)
            for k in range(patch_size[0]):
                for m in range(patch_size[1]):
                    output[k,m]=img[i+k,j+m]
            temp = np.sum(patch*output)
            temp = temp/9       #Normalisation
            output_matrix[i,j]=temp
    return output_matrix


# In[4]:


#Read Image
I = cv2.imread('einstein.jfif',0)
 
# Call for Averaging function
output_matrix = averaging(I)


# In[5]:


output_matrix = np.uint8(output_matrix) #Convert the resultant to uint8
cv2.imwrite('Averaged_Image.png', output_matrix)  # Save the averaged image
    

