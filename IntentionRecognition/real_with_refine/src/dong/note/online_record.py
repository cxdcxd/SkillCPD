#!/usr/bin/env python
import sys
sys.path.append("/home/student6/ma-yang/real_with_refine/src/dong")

from pbdlib.gui import InteractiveDemos, MutliCsInteractiveDemos
import argparse
import pbdlib.gui

import os
# import time
import random
import pylab as pl
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
# import pbdlib as pbd


# KNN Classifier
# from sklearn import datasets  
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score
# import pandas as pd
# import seaborn as sns

# def DataProcessing(data):
#     data = np.array(data)
#     nsamples, nx, ny = data.shape
#     if nx > 200:                                                 # delete randomly datapoints
#         index = random.sample(range(1,nx), nx-200)
#         demo_data = np.delete(data, index, axis=1)
#         demo_data_dim2 = demo_data.reshape((nsamples,-1))
#         return demo_data_dim2
#     elif nx < 200:                                               # Interpolation
#         demo_data = []
#         for k in range(len(data)):
#             values = np.reshape(data[k],-1)
#             points = np.array([[i,j] for i in range(0, nx) for j in range(0, ny)])
#             grid_x, grid_y = np.mgrid[0:nx-1:200j, 0:ny-1:4j]
#             data_new = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
#             demo_data.append(data_new)
#         demo_data = np.array(demo_data)
#         demo_data_dim2 = demo_data.reshape((nsamples,-1))
#         return demo_data_dim2
#     else:
#         return data

# KNN Classifier training
# all_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# labels = []
# for letter in all_letters:
#     data = np.load('./DemoData/%s.npy'%(letter))
#     labels.extend(['%s'%(letter)]*len(data))
# labels = np.array(labels)
# training_data = np.load('./DemoData/A.npy')
# all_letters.pop(0)
# for letter in all_letters:
#     training_data2 = np.load('./DemoData/%s.npy'%(letter))
#     training_data = np.vstack((training_data,training_data2))

# nsamples, nx, ny = training_data.shape
# training_data_dim2 = training_data.reshape((nsamples,nx*ny))
# data_train,data_test,labels_train,labels_test = train_test_split(training_data_dim2,labels,test_size=0.1)
# knn = KNeighborsClassifier()
# knn.fit(data_train,labels_train)
# labels_pred = knn.predict(data_test)
# print("The accuracy of predition with test data is %.2f%%" %(accuracy_score(labels_test,labels_pred)*100))

# Start record the demostrations
arg_fmt = argparse.RawDescriptionHelpFormatter

parser = argparse.ArgumentParser(formatter_class=arg_fmt)

parser.add_argument(
	'-f', '--filename', dest='filename', type=str,
	default='test', help='filename for saving the demos'
)
parser.add_argument(
	'-p', '--path', dest='path', type=str,
	default='', help='path for saving the demos'
)

parser.add_argument(
	'-m', '--multi_cs', dest='multi_cs', action='store_true',
	default=False, help='record demos in multiple coordinate systems'
)
parser.add_argument(
	'-c', '--cs', dest='nb_cs', type=int,
	default=2, help='number of coordinate systems'
)

args = parser.parse_args()

if args.multi_cs:
	interactive_demo = MutliCsInteractiveDemos(
		filename=args.filename, path=args.path, nb_experts=args.nb_cs)
else:
	interactive_demo = InteractiveDemos(
		filename=args.filename, path=args.path)

# knn, accuracy = interactive_demo.knn_classifier()
# print("The accuracy of predition with test data is %.2f%%" %accuracy)

interactive_demo.start()

# # Take Data from record
# time_start = time.time()
# mydata = np.array(interactive_demo.translate_demo_data())
# mydata = mydata.tolist()
# mydata_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(mydata['x'], mydata['dx'])]

# # Online prediction
# data_dim2 = interactive_demo.DataProcessing(mydata_xdx)
# # time_start = time.time()
# demo_pred = knn.predict(data_dim2)
# time_end = time.time()
# print("The prediction delay is: %.5fs" %(time_end-time_start))
# print("The prediction of the given Data is: %s" %demo_pred[0])

