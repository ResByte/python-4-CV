#!/usr/bin/env python

import cv2
import numpy as np
from scipy.cluster.vq import vq, kmeans2, whiten




def surf_img(img):
	
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	surf = cv2.SURF()
	surf.extended = True
	surf.hessianThreshold = 1000
	kp,des = surf.detectAndCompute(imgray,None)
	features = np.asarray(des)		
	centroid,label = kmeans2(features,cluster_n,iter=10,thresh=1e-05, minit='random', missing='warn')
	return features, centroid, label,kp


if __name__ == '__main__':
	np.set_printoptions(threshold=np.nan)
	cluster_n = 50
	tmp_features=[]	
	cap = cv2.VideoCapture(0)
	cv2.namedWindow('plot',1)
	while True:
		ret,img = cap.read()
		
		features, centr,labels,kp1 = surf_img(img)
					
		img2 = cv2.drawKeypoints(img,kp1,None,(255,0,0),4)
		
		cv2.imshow('plot',img2)
		if cv2.waitKey(30)>0:
			break
