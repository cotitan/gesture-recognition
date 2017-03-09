import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import math
import os
import gesture as gst
import heapq
from collections import Counter

def extract_feature(pic):
	ctr = gst.getCentroid(pic)
	contours = gst.getContours(pic)
	shape_signt = gst.centroid_dist(contours, ctr)
	fds = gst.FDFT(shape_signt)
	return fds

def loadDataSet():
	files = os.listdir('hands')[0:200]
	random.shuffle(files)
	label = []
	data = np.zeros((len(files), 120)) # 120 = fd_len - 8
	for i in range(0, 200):
		label.append(files[i][0])
		pic = mpimg.imread('hands/' + files[i])
		data[i] = extract_feature(pic).copy()
	return label[0:160], data[0:160], label[160:200], data[160:200]

def test(train_label, train_data, test_label, test_data):
	train_len = len(train_label)
	test_len = len(test_label)
	print('expected, output')
	error_count = 0; error_rate = 0.0
	for i in range(0, test_len):
		dist = []
		for j in range(0, train_len):
			dist.append( (j, ecli_dist(test_data[i], train_data[j])) )
		print(dist)
		top10 = heapq.nsmallest(10, dist, key = lambda x : x[1])
		print(top10)
		top10cls = [train_label[k[0]] for k in top10]
		pred = (Counter(top10cls).most_common(1))[0][0]
		if (pred != test_label[i]):
			print('   %s       %s' % (test_label[i], pred))
			error_count += 1
	error_rate = float(error_count) / test_len
	print('error_rate: ', error_rate)
	return error_rate

if __name__ == '__main__':
	train_l, train_data, test_l, test_data = loadDataSet()

def ecli_dist(vec1, vec2):
	return sum(vec1 * vec2)

