#!/usr/bin/python
import cv2
import numpy as np
import sys

# Load image
file_image = sys.argv[1]
file_template = sys.argv[2]
color_image = cv2.imread(file_image)
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# Get SURF
surf = cv2.SURF()
(kp, descriptors) = surf.detect(gray_image, None, useProvidedKeypoints = False)

# Setting up samples and responses for kNN
samples = np.array(descriptors)
responses = np.arange(len(kp), dtype = np.float32)

# kNN training
knn = cv2.KNearest()
print(len(samples), len(responses))
knn.train(samples,responses)

# Load Template
color_template = cv2.imread(file_template)
gray_template = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
keys, desc = surf.detect(gray_template, None, useProvidedKeypoints = False)

for h,des in enumerate(desc):
  des = np.array(des,np.float32).reshape((1,128))
  retval, results, neigh_resp, dists = knn.find_nearest(des,1)
  res,dist = int(results[0][0]),dists[0][0]
  
  # If matched mark in red
  if dist < 0.1:
    color = (0,0,255)
  # Else mark in blue
  else:
    print dist
    color = (255,0,0)

  # draw keypoints
  x,y = kp[res].pt
  center = (int(x),int(y))
  cv2.circle(color_image,center,2,color,-1)
  
  # Draw matched keypoints on template image
  x,y = keys[h].pt
  center = (int(x),int(y))
  cv2.circle(color_template,center,2,color,-1)

cv2.imshow('Image', color_image)
cv2.imshow('Template', color_template)
cv2.waitKey(0)
cv2.destroyAllWindows()

