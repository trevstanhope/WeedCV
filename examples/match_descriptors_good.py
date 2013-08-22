#!/usr/bin/env python
import cv2
import numpy
import sys
import time

# Load images
image = sys.argv[1]
template = sys.argv[2]
haystack_color = cv2.imread(image)
needle_color = cv2.imread(template)

# Make gray
haystack_gray = cv2.cvtColor(haystack_color, cv2.COLOR_BGR2GRAY)
needle_gray = cv2.cvtColor(needle_color, cv2.COLOR_BGR2GRAY)

# Build feature detector and descriptor extractor
hessian_threshold = 85
detector = cv2.SURF(hessian_threshold)
(haystack_keypoints, haystack_descriptors) = detector.detect(haystack_gray, None, useProvidedKeypoints=False)
(needle_keypoints, needle_descriptors) = detector.detect(needle_gray, None, useProvidedKeypoints=False)

# Extract vectors from raw arrays
rowsize = len(haystack_descriptors) / len(haystack_keypoints)
if rowsize > 1:
  haystack_rows = numpy.array(haystack_descriptors, dtype=numpy.float32).reshape((-1,rowsize))
  needle_rows = numpy.array(needle_descriptors, dtype=numpy.float32).reshape((-1,rowsize))
  print(haystack_rows.shape, needle_rows.shape)
else:
  haystack_rows = numpy.array(haystack_descriptors, dtype=numpy.float32)
  needle_rows = numpy.array(needle_descriptors, dtype=numpy.float32)
  rowsize = len(haystack_rows[0])

# kNN training
samples = haystack_rows
responses = numpy.arange(len(haystack_keypoints), dtype=numpy.float32)
print(len(samples), len(responses))
knn = cv2.KNearest()
knn.train(samples,responses)

# Input to knn must be a float matrix...
for (error, descriptor) in enumerate(needle_rows):
  print(descriptor)
  (error, results, neighor_response, distances) = knn.find_nearest(descriptor,1)
