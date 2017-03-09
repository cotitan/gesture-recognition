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
	rows = 2080
	bound = int(rows * 0.8)
	files = os.listdir('hands')[0:rows]
	random.shuffle(files)
	label = np.zeros(len(files), dtype = 'int32')
	data = np.zeros((len(files), 120), dtype = 'float32') # 120 = fd_len - 8
	for i in range(0, rows):
		label[i] = ord(files[i][0])
		pic = mpimg.imread('hands/' + files[i])
		data[i] = extract_feature(pic)
	return label[0:bound], data[0:bound],\
			label[bound:rows], data[bound:rows]

def test(train_label, train_data, test_label, test_data):
	train_len = len(train_label)
	test_len = len(test_label)
	print('expected, output')
	error_count = 0; error_rate = 0.0
	for i in range(0, test_len):
		dist = []
		for j in range(0, train_len):
			dist.append( (j, ecli_dist(test_data[i], train_data[j])) )
		# print(dist)
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

def bayes(tr_lb, tr_dt, tst_lb, tst_dt):
	bayes_model = cv2.ml.NormalBayesClassifier_create()
	ret = bayes_model.train(tr_dt, cv2.ml.ROW_SAMPLE, tr_lb)
	retval, ans = bayes_model.predict(tst_dt)
	matches = ans[:,0] == tst_lb
	# success_rate = np.count_nonzero(matches) * 100.0 / ans.size
	success_rate = np.sum(np.in1d(ans, tst_lb)) / float(tst_lb.shape) * 100
	print('for bayes_model, success rate = ', success_rate )
	return retval, ans

def KNN(tr_lb, tr_dt, tst_lb, tst_dt):
	knn = cv2.ml.KNearest_create()
	knn.train(tr_dt, cv2.ml.ROW_SAMPLE, tr_lb)
	ret, result, neighbours, dist = knn.findNearest(tst_dt, k = 1)
	result = np.array(result, dtype = 'int32')[:,0]
	matches = result == tst_lb
	correct = np.count_nonzero(matches)
	accuracy = correct * 100.0 / result.size
	print('knn accuracy: %f%%' % accuracy)
	return result

def SVM(tr_lb, tr_dt, tst_lb, tst_dt):
	svm = cv2.ml.SVM_create()
	svm.setKernel(cv2.ml.SVM_LINEAR)
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.train(tr_dt, cv2.ml.ROW_SAMPLE, tr_lb)
	ret, res = svm.predict(tst_dt)
	mask = res[:,0] == tst_lb
	correct = np.count_nonzero(mask)
	print("svm accuracy: %f%%" %(correct * 100.0 / res.size))
	return res
if __name__ == '__main__':
	train_l, train_data, test_l, test_data = loadDataSet()

def ecli_dist(vec1, vec2):
	return sum(vec1 * vec2)

