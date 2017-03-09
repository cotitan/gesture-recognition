import os
import matplotlib.pyplot as plt
import cv2

def trans():
	files = os.listdir('hands')
	for each in files:
		pic = plt.imread('hands/' + each)
		cv2.imwrite('hands/' + each, pic[:,:,0])

