#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Histogram Equalization is a computer image processing technique used to improve contrast in images. 
##It accomplishes this by effectively spreading out the most frequent intensity values, 
##i.e. stretching out the intensity range of the image.


# In[2]:


#Import required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#Plot a histogram 
def plot_hist(I):
    hist=np.zeros(256) #create an array of size(256)
    s = I.shape   
    for i in range(s[0]):
        for j in range(s[1]):
            hist[I[i,j]] += 1
    return hist


# In[4]:


#Plotting Cumilative Histogram
def plot_cumilativeHist(hist):
    ch=np.zeros(256) #create an array of size(256)

    for i in range(len(hist)):
        if i==0:
            ch[i]=hist[i]
        else:
            ch[i]=hist[i]+ch[i-1]
            
    return ch


# In[5]:


#Construct A Look-Up Table
def lookUp_table(ch , m):
    lt=np.zeros(256)

    for k in range(m,len(ch)):
        lt[k]=(ch[k]-ch[m])/(ch[255]-ch[m])
    lt=lt*255
    return lt


# In[6]:


#Read Image
I = cv2.imread('lena_dark.png',0)


# In[7]:


hist = plot_hist(I)

plt.plot(hist)


# In[8]:


ch = plot_cumilativeHist(hist)

#Find the index of very first non-zero element
for i in range(len(ch)):
    if ch[i]!=0:
        m=i
        break


# In[9]:


lt = lookUp_table(ch , m)


# In[10]:


#Create a new image
s = I.shape
img1=np.uint8(np.zeros((s[0],s[1])))


for i in range(s[0]):
    for j in range(s[1]):
        img1[i,j]=(lt[I[i,j]])


# In[11]:


hist_new = plot_hist(img1)

plt.plot(hist_new)

