#!/usr/bin/env python
# Finds location of Crab Grass in image for herbicide application
import cv2, cv
import numpy
import sys

# Load image
filename = sys.argv[1]
color_image = cv2.imread(filename)
cv2.imshow('Color Image', color_image)
cv2.waitKey(0)

# Convert to Gray
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)

# Inverse Threshold Image
MAX = 255
THRESHOLD = int(gray_image.mean())
(flag, threshold_image) = cv2.threshold(gray_image, THRESHOLD, MAX, cv2.THRESH_BINARY_INV) # Binary Inverted
cv2.imshow('Threshold Image', threshold_image)
cv2.waitKey(0)

# Dilate makes white areas bigger
DILATION = 5 # need a variable method for this parameter
threshold_matrix = cv.fromarray(threshold_image)
cv.Dilate(threshold_matrix, threshold_matrix, None, DILATION)
cv2.imshow('Threshold Image', threshold_image) 
cv2.waitKey(0)

# Erode makes black areas bigger
EROSION = 30 # need a variable method for this parameter
cv.Erode(threshold_matrix, threshold_matrix, None, EROSION)
cv2.imshow('Threshold Image', threshold_image)
cv2.waitKey(0)

# Display Weed coverage information
coverage = int((1 - threshold_image.sum()/float(640*480*255))*100)
print('Weed Coverage: %s%%' % coverage)
(x,y) = numpy.where(threshold_image == 0) # black regions are weeds
print('Weed X Locations: %s' % x)
print('Weed Y Locations: %s' % y)
