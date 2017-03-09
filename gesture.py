import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as MPIMG
import os
import cv2
import math
import random

pic_dir = 'database1'
# pic_dir = 'a1'
fd_len = 128
pi = 3.1415926

def extract_hand(filename):
	src = MPIMG.imread(filename)
	# blur = cv2.medianBlur(src, 3)
	blur = cv2.GaussianBlur(src, (3, 3), 1.5)
	
	m, n = blur.shape
	for i in range(0, m):
		for j in range(0, n):
			if (blur[i][j] < 128):
				blur[i][j] = 0
	ctr_r, ctr_c = getCentroid(blur)
	hand = blur[ctr_r - 44 : ctr_r + 36, ctr_c - 35 : ctr_c + 45]
	# plt.imshow(hand, cmap = 'gray')
	# plt.show()
	return hand

def getCentroid(pic):
	(m, n) = pic.shape
	r = 0; c = 0
	count = 0
	for i in range(0, m):
		for j in range(0, n):
			if pic[i][j] > 0:
				r += i
				c += j
				count += 1
	return int(r / count), int(c / count)

def centroid_dist(boundarys, centroid):
	index = 0.0
	step = len(boundarys) / float(fd_len)
	dists = np.zeros(fd_len)
	for i in range(0, fd_len):
		j = int(index)
		dists[i] = eucld_metric(boundarys[j], centroid)
		index += step
	return dists

def eucld_metric(a, b):
	return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

def FDFT(shape_signt):
	fds = np.zeros(fd_len)
	for n in range(0, fd_len):
		for i in range(0, fd_len):
			fds[n] += shape_signt[i] * math.exp(-2 * pi * i * n / fd_len)
		fds[n] /= fd_len
	return fds[8:]

def getContours(pic):
	img, contours, hrc = cv2.findContours(pic, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_NONE)
	# shape of contours: (1, 136, 1, 2)
	# or (2,) and for 0: (136, 1, 2), for 1: (4, 1, 2)
	cts = np.array(contours[0])
	cts = cts.reshape(cts.shape[0], 2)
	return cts

def getBoundary(pic):
	edge = cv2.Canny(pic, 100, 250)
	points = []
	m, n = edge.shape
	for i in range(0, m):
		for j in range(0, n):
			if edge[i][j] != 0:
				points.append((i, j))
	return np.array(points)
	# thre2 = 250 is impressive
	"""
	thre1 = {25, 50, 75, 100}
	thre2 = {100, 150, 200, 250}
	for i in thre1:
		for j in thre2:
			edge = cv2.Canny(pic, i, j)
			plt.title('%d,%d' % (i, j))
			plt.imshow(edge, cmap = 'gray')
			plt.show()
	"""

# show pic from start to end
def showall(start, end):
	piclist = os.listdir(pic_dir)
	end = end if len(piclist) > end else len(piclist)
	for i in range(start, end):
		extract_hand(pic_dir + '/' + piclist[i])

def grayGraph(pic):
	grays = [0] * 306
	m, n = pic.shape
	for i in range(0, m):
		for j in range(0, n):
			grays[pic[i][j]] += 1
	for i in range(0, 306):
		if (grays[i] > 100):
			grays[i] = 100
	
	x = np.arange(len(grays))
	plt.plot(x, grays, 'r')
	plt.show()

def gen_dataFile():
	piclist = os.listdir(pic_dir)
	label = ['a'] * len(piclist)
	data = []
	for i in range(0, len(piclist)):
		label[i] = piclist[i][0]
		data.append((label[i], piclist[i]))
	random.shuffle(data)
	test = data[0:516]
	train = data[516:]
	return train, test

def extract_all_hand():
	piclist = os.listdir('database1')
	for each in piclist:
		print(each)
		hand = extract('database1/' + each)
		new_name = 'hands/' + each.split('.')[0] + '_small.tif'
		MPIMG.imsave(new_name, hand, cmap = 'gray')

def binary_seg(pic, threshold):
	m, n = pic.shape
	for i in range(0, m):
		for j in range(0, n):
			if pic[i][j] < threshold:
				pic[i][j] = 0


def hand_segmt(filename):
	pic = MPIMG.imread(filename)
	# blur = cv2.GaussianBlur(pic, (3, 3), 1.0)
	blur = cv2.medianBlur(pic, 3)
	copy = blur.copy()
	binary_seg(blur, 150)
	ctr = getCentroid(blur)
	hand = copy[ctr[0]-45 : ctr[0]+35, ctr[1]-40 : ctr[1]+40]
	m, n = hand.shape
	binary_seg(hand, 127)
	# plt.imshow(hand, cmap = 'gray')
	# plt.show()
	return hand
