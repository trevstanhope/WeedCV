#!/usr/bin/env python
# Display Keypoints of image
import cv2
import sys
HESSIAN_THRESHOLD = 85
KEYPOINT_RADIUS = 2
KEYPOINT_COLOR = (0,0,255)

# Load image
color = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# Find keypoints
surf = cv2.SURF(HESSIAN_THRESHOLD)
keypoints, descriptors = surf.detect(gray, None, useProvidedKeypoints=False)
for target in keypoints:
  x = int(target.pt[0])
  y = int(target.pt[1])
  cv2.circle(color, (x,y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1) # draw solid circles
cv2.imshow('Key Features', color)
cv2.waitKey(0)
  
