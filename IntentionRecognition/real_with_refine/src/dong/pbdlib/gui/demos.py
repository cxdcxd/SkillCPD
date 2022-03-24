#!/usr/bin/env python
import os
import time
import math
import random
import itertools
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from .interactive import Interactive
from termcolor import colored
import sys
if sys.version_info[0] == 3:
	import tkinter as tk
	from tkinter.filedialog import asksaveasfilename
else:
	import Tkinter as tk
	from tkFileDialog import asksaveasfilename

from matplotlib import gridspec
import pbdlib as pbd 

# KNN Classifier
from sklearn import datasets  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score
import pandas as pd
import seaborn as sns

# cpd registration
# from functools import partial
# import matplotlib.pyplot as plt
from pycpd import AffineRegistration
from pycpd import RigidRegistration

# ros package
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from joint_control.msg import HapticCommand 


class Robot(object):
	def __init__(self, T):
		self.x, self.dx, self.dt = np.array([0., 0.]), np.array([0., 0.]), 1. / T
		self.ddx, self.fsx, self.fx = np.array([0., 0.]), np.array([0.]), np.array([0., 0.])
		self.sensor_mode = 0

class InteractiveDemos(Interactive, Robot):
	"""
	GUI for recording demonstrations in 2D
	"""
	def	__init__(self, filename='test', path='', **kwargs):
		Interactive.__init__(self)

		self.path = os.path.dirname(pbd.__file__) + '/data/gui/' if path == '' else path
		
		self.fig = plt.figure(figsize=(15, 8), facecolor='white')
		self.bindings.update({
			'q': (self.save_demos, [], "save demos"),
			'c': (self.clear_demos, [], "clear demos"),
			'x': (self.clear_demos, [True], "clear last demos"),
			'i': ([self.incr_param, self.highlight_demos], [['current_demo'], []], "next demos"),
			'd': (self.clear_demos, [False, True], "clear selected demos"),
		})

		Robot.__init__(self, self.simulation_T)

# 		gs = gridspec.GridSpec(1, 2)

		gs = gridspec.GridSpec(1, 3)

		self.filename = filename

		self.ax_x = plt.subplot(gs[0])
		self.ax_dx = plt.subplot(gs[1])
		self.ax_pre = plt.subplot(gs[2])

		self.set_events()
		self.set_plots()

		self.is_demonstrating = False
		self.velocity_mode = False


		self.curr_demo, self.curr_demo_dx = [], []
		self.curr_demo_obj, self.curr_demo_obj_dx = [], []

		self._current_demo = {'x': [], 'dx': []}

		self.curr_mouse_pos = None
		self.robot_pos = np.zeros(2)

		self.nb_demos = 0
		self.stop_pred = 0
		self.demos = {'x': [], 'dx': []}
		self.preds = []
		self.mydemos = {'x': [], 'dx': []}
        
		self.demo_pred = []
		self.demo_data = np.array('0')
		self.xi = np.array('0')
		self.qos = 1
		# self.mouse_points = Pose()
		self.input_trigger = 0
		self.state_num = 0
		self.letters = ['A', 'G']
		self.letter_id_poss = [self.letters]
		self.stop_direct_tele = 0
		self.scaling_robot = 0.001*0.8
		self.scaling_haptic = 50
		self.res_finish_pub = 0
		self.haptic_pose = HapticCommand()
		self.haptic_x = 0
		self.haptic_directx = 0
		self.haptic_y = 0
		self.haptic_directy = 0
		self.sample_num = 0
		# self.demo_points = []

		self.params.update({'current_demo': [0, 0, self.nb_demos]})

		self.loaded = False
		
		self.names = locals()

		try:
			self.demos = np.load(self.path + filename + '.npy')[()]
			self.nb_demos = self.demos['x'].__len__(); self.params['current_demo'][2] = self.nb_demos - 1
			print(colored('Existing skill, demos loaded', 'green'))
			self.replot_demos()
			self.fig.canvas.draw()
			self.loaded = True
		except:
			self.demos = {'x': [], 'dx': []}
			print(colored('Not existing skill', 'red'))
		
		print('Training all HSMM-models in advance')
		for i in self.letters:
			self.names['model_' + i] = self.hsmm_training(i)

		rospy.init_node('dong', anonymous=True)
		print('The subscriber is set up')
		rospy.Subscriber('/movo/right_arm/control/haptic', HapticCommand, self.callback_haptic)
		rospy.Subscriber('QoS', Float64, self.callback_qos)
		# rospy.Subscriber('Mouse_input', Pose, self.callback_mouse_points)
		print('The publisher for demonstration is set up')
		# self.pub_demo = rospy.Publisher('right/nmpc_controller/in/goal', Pose, queue_size=1)
		self.pub_demo = rospy.Publisher('trajectory_points', Pose, queue_size=1)
		self.pub_demo_robot = rospy.Publisher('right/nmpc_controller/in/goal', Pose, queue_size=1)
		self.pub_finish = rospy.Publisher('finish_pub', Float64, queue_size=1)
		# self.pub_repro = rospy.Publisher('right/nmpc_controller/in/goal', Pose, queue_size=1)
		self.pub_repro = rospy.Publisher('trajectory_points', Pose, queue_size=1)
		rospy.sleep(1)
		self.init_robot()
		# rospy.spin()

	def init_robot(self):

		result = Pose()

		# In x-y coordinate
		# The height of robot is 321
		result.position.x = 0.237099394202
		result.position.y = 0.372370868921
		result.position.z = 0.57826089859
		result.orientation.x = 0.0498233951364
		result.orientation.y = -0.00260980955774
		result.orientation.z = -0.039337669017
		result.orientation.w = 0.997979642071

		self.pub_demo_robot.publish(result)
		print('The initial point is sent...')

	def callback_haptic(self, msg_haptic):
		self.sample_num += 1
		if self.sample_num % 5 == 0 and self.qos:
			self.haptic_pose = msg_haptic
			if self.haptic_pose.gripper_angle and not self.stop_direct_tele:
				self.haptic_directx = self.scaling_haptic * (-self.haptic_pose.position[1])
				self.haptic_directy = self.scaling_haptic * self.haptic_pose.position[0]
				if not self.input_trigger:
					self.input_trigger = 1
					self.start_system()
					result = Pose()

					# In x-z coordinate
					# result.position.x = self.scaling_robot * self.haptic_directx - 0.16
					# result.position.z = self.scaling_robot * self.haptic_directy + 0.64
					# result.position.y = 0.0564058
					# result.orientation.x = -0.500128
					# result.orientation.y = 0.510553
					# result.orientation.z = 0.491395
					# result.orientation.w = 0.497736	

					# In x-y coordinate
					result.position.x = self.scaling_robot * self.haptic_directx -0.143
					result.position.y = self.scaling_robot * (-self.haptic_directy) + 0.0670786			
					# result.position.z = 0.859739
					result.position.z = 0.89
					result.orientation.x = -0.00766462
					result.orientation.y = 0.0106087
					result.orientation.z = 0.701056
					result.orientation.w = 0.712987

					self.pub_demo_robot.publish(result)
				else:
					self.demo_event_haptic()
					result = Pose()

					# In x-z coordinate
					# result.position.x = self.scaling_robot * self.haptic_directx - 0.16
					# result.position.z = self.scaling_robot * self.haptic_directy + 0.64
					# result.position.y = 0.0564058
					# result.orientation.x = -0.500128
					# result.orientation.y = 0.510553
					# result.orientation.z = 0.491395
					# result.orientation.w = 0.497736

					# In x-y coordinate
					result.position.x = self.scaling_robot * self.haptic_directx -0.143
					result.position.y = self.scaling_robot * (-self.haptic_directy) + 0.0670786					
					# result.position.z = 0.859739
					result.position.z = 0.89
					result.orientation.x = -0.00766462
					result.orientation.y = 0.0106087
					result.orientation.z = 0.701056
					result.orientation.w = 0.712987

					# self.pub_demo.publish(result)
					self.pub_demo_robot.publish(result)

		elif self.sample_num % 5 == 0 and not self.qos and self.input_trigger:
			if not self.stop_direct_tele:
				print('QoS is too low, please wait for the response of the network...')

	def callback_qos(self, msg_qos):
		self.qos = msg_qos.data
		# print(self.qos)
	
	# def callback_mouse_points(self, msg_mouse):
	# 	if self.qos >= 50 and not self.stop_direct_tele:
	# 		self.mouse_points = msg_mouse
	# 		if not self.input_trigger:
	# 			self.input_trigger = 1
	# 			self.start_system()
	# 			result = Pose()
	# 			# result.position.x = self.scaling * self.mouse_points.position.x + 0.06
	# 			result.position.x = self.scaling_robot * self.mouse_points.position.x - 0.01
	# 			result.position.z = self.scaling_robot * self.mouse_points.position.y + 0.73
	# 			result.position.y = 0.0564058
	# 			result.orientation.x = -0.500128
	# 			result.orientation.y = 0.510553
	# 			result.orientation.z = 0.491395
	# 			result.orientation.w = 0.497736
	# 			self.pub_demo_robot.publish(result)
	# 		else:
	# 			self.demo_event()
	# 			result = Pose()
	# 			# rate = rospy.Rate(10)
	# 			# result.position.x = self.scaling * self.mouse_points.position.x + 0.06
	# 			result.position.x = self.scaling_robot * self.mouse_points.position.x - 0.01
	# 			result.position.z = self.scaling_robot * self.mouse_points.position.y + 0.73
	# 			result.position.y = 0.0564058
	# 			result.orientation.x = -0.500128
	# 			result.orientation.y = 0.510553
	# 			result.orientation.z = 0.491395
	# 			result.orientation.w = 0.497736
	# 			self.pub_demo.publish(result)
	# 		# rospy.sleep(10)
	# 		# rate.sleep()
	# 		# self.demo_points += [[self.mouse_points.position.x, self.mouse_points.position.y]]
	# 		# print([self.mouse_points.position.x, self.mouse_points.position.y])
	# 	elif self.qos < 50 and not self.stop_direct_tele:
	# 		print('QoS is too low, please wait for the response of the network...')
	# 	# else:
	# 	# 	print('The direct teleoperation is stopped..')

	# def trajectory_pub(self):
	# 	print('The publisher is set up')
	# 	pub = rospy.Publisher('trajectory_points', Pose, queue_size=1)
	# 	# pub = rospy.Publisher('right/nmpc_controller/in/goal', Pose, queue_size=1)
	# 	#rospy.init_node('dong', anonymous=True)
	# 	rate = rospy.Rate(5)
	# 	result = Pose()
	# 	for i in range(len(self.xi[:, 0])):
	# 		result.position.x = self.xi[i,0]
	# 		result.position.y = self.xi[i,1]
	# 		result.position.z = 0
	# 		# print('The reproduced trajectory point is publishing...')
	# 		pub.publish(result)
	# 		# rospy.sleep(0.2)
	# 		rate.sleep()
	# 	print('Publishing finished')

	# def trajectory_pub_repro(self):
	# 	print('The publisher for reproduction is set up')
	# 	pub = rospy.Publisher('right/nmpc_controller/in/goal', Pose, queue_size=1)
	# 	#rospy.init_node('dong', anonymous=True)
	# 	rate = rospy.Rate(5)
	# 	result = Pose()
	# 	for i in range(len(self.xi[:, 0])):
	# 		result.position.x = self.scaling * self.xi[i,0] + 0.218793
	# 		result.position.z = self.scaling * self.xi[i,1] + 0.743591
	# 		result.position.y = 0.0564058
	# 		result.orientation.x = -0.500128
	# 		result.orientation.y = 0.510553
	# 		result.orientation.z = 0.491395
	# 		result.orientation.w = 0.497736
	# 		pub.publish(result)
	# 		rate.sleep()
	# 	print('Publishing finished')		

	def DataProcessing(self, data):
		data = np.array(data)
		nsamples, nx, ny = data.shape
		if nx > 200:                                                # delete randomly data points
			index = random.sample(range(1,nx), nx-200)
			demo_data = np.delete(data, index, axis=1)
			demo_data_dim2 = demo_data.reshape((nsamples,-1))
			return demo_data, demo_data_dim2
		elif nx < 200:                                               # Interpolation
			demo_data = []
			for k in range(len(data)):
				values = np.reshape(data[k],-1)
				points = np.array([[i,j] for i in range(0, nx) for j in range(0, ny)])
				grid_x, grid_y = np.mgrid[0:nx-1:200j, 0:ny-1:4j]
				data_new = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
				demo_data.append(data_new)
			demo_data = np.array(demo_data)
			demo_data_dim2 = demo_data.reshape((nsamples,-1))
			return demo_data, demo_data_dim2
		else:
			demo_data_dim2 = data.reshape((nsamples,-1))
			return data, demo_data_dim2

	def create_hsmm_data(self, demo_pred):
		# print(demo_pred[0])
		letter = demo_pred[0]
		#files = ['demo%s'%letter, 'demo%s1'%letter, 'demo%s2'%letter, 'demo%s3'%letter, 'demo%s4'%letter]
		files = ['demo%s'%letter, 'demo%s1'%letter, 'demo%s2'%letter, 'demo%s3'%letter, 'demo%s4'%letter, 'demo%s5'%letter, 'demo%s6'%letter, 'demo%s7'%letter, 'demo%s8'%letter, 'demo%s9'%letter]
		files.pop(0)

		#train_data = np.load("./mydataset/demo%s.npy"%letter, allow_pickle=True)
		train_data = np.load("/home/student6/ma-yang/real_with_refine/src/dong/note/mydataset/demo%s.npy"%letter, allow_pickle=True)
		train_data = train_data.tolist()
		train_data_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(train_data['x'], train_data['dx'])]
		demo_data, _ = self.DataProcessing(train_data_xdx)
		for i in files:
			#mydata1 = np.load("./mydataset/%s.npy"%i, allow_pickle=True)
			mydata1 = np.load("/home/student6/ma-yang/real_with_refine/src/dong/note/mydataset/%s.npy"%i, allow_pickle=True)
			mydata1 = mydata1.tolist()
			mydata1_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(mydata1['x'], mydata1['dx'])]
			demo_data1, _ = self.DataProcessing(mydata1_xdx)
			demo_data = np.vstack((demo_data,demo_data1))
		return demo_data


	def hsmm_training(self, letter):

		demo_data = self.create_hsmm_data(letter)
		# demo_data_x, demo_data_dx = self.separate_xdx(self.demo_data)
		self.names['model_' + letter] = pbd.HSMM(nb_states=5, nb_dim=4)
		self.names['model_' + letter].init_hmm_kbins(demo_data)
		self.names['model_' + letter].em(demo_data, reg=1e-6)
		# print(letter)
		return self.names['model_' + letter]

	def cal_distance(self, p1, p2):
		return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

	def most_frequent(self, temp):
		res = {x:temp.count(x) for x in temp}
		if np.array(temp).shape[0] > 0:
			return [k for k,v in res.items() if v >= 10]
		else:
			return []

	# def likelihood_prediction(self):
	# 	time_start_pred = time.time()
	# 	mydata_xdx = self.translate_demo_data()
	# 	if np.array(mydata_xdx).size == 0:
	# 		print('Sorry, the prediction needs more data points')
	# 	else:
	# 		for i in self.letters:
	# 			self.names['alpha_hsmm_' + i], _, _, _, _ = self.names['model_' + i].compute_messages(mydata_xdx[np.array(mydata_xdx).shape[0]-1], marginal=slice(0, 4))
	# 		i = self.state_num - 1
	# 		temp = []
	# 		for j in range(40*i, self.names['alpha_hsmm_' + self.letter_id_poss[-1][0]].shape[1]):
	# 			alpha_letters = [self.names['alpha_hsmm_' + self.letter_id_poss[-1][0]][i][j]]
	# 			all_letters = self.letter_id_poss[-1][1:]
	# 			for k in all_letters:
	# 				alpha_letters += [self.names['alpha_hsmm_' + k][i][j]]						
	# 			alpha_letters = [value for value in alpha_letters if not math.isnan(value)]
	# 			for n in self.letter_id_poss[-1]:
	# 				if max(alpha_letters) == self.names['alpha_hsmm_' + n][i][j]:
	# 					temp.append(n)
	# 		if np.array(self.most_frequent(temp)).size != 0:
	# 			self.letter_id_poss += [self.most_frequent(temp)]
	# 		self.demo_pred = self.letter_id_poss[-1]
	# 		print('The predicted letter is: %s' %self.demo_pred)
	# 	time_end_pred = time.time()
	# 	print("The prediction delay is: %.5fs" %(time_end_pred - time_start_pred))

	# 	# Start to reproduce the trajectory
	# 	if len(self.demo_pred) == 1 and not self.stop_pred:
	# 		self.stop_pred = 1
	# 		# self.is_demonstrating = 0
	# 		while 1:
	# 			if self.qos < 50:
	# 				print('Qos is too low, reproduction based on intention recognition started...')
	# 				self.stop_direct_tele = 1
	# 				print('The direct teleoperation is stopped...')
	# 				# self.stop_pred = 1
	# 				self.letter_id_poss = [self.letters]
	# 				time_start_repro = time.time()
	# 				self.demo_data = self.create_hsmm_data(self.demo_pred)
	# 				self.xi = self.hsmm_lqr(self.demo_pred, self.demo_data)
	# 				time_end_repro = time.time()
	# 				print("The reproduction delay is: %.5fs" %(time_end_repro - time_start_repro))
	# 				# self.repro = self.cpd_repro_affine()
	# 				self.repro = self.cpd_repro_rigid()
	# 				#self.finish_demo()
	# 				#self.reproduction_plot()
	# 				self.finish_reproduction()
	# 				# try:
	# 				# 	self.trajectory_pub_repro()
	# 				# except rospy.ROSInterruptException:
	# 				# 	pass
	# 				# self.finish_reproduction()
	# 				break
	# 			else:
	# 				# print('QoS is good, keep direct teleoperation...')
	# 				# print(self.qos)
	# 				continue

	def likelihood_prediction(self):
		time_start_pred = time.time()
		mydata_xdx = self.translate_demo_data()
		if np.array(mydata_xdx).size == 0:
			print('Sorry, the prediction needs more data points')
		else:
			for i in self.letters:
				self.names['alpha_hsmm_' + i], _, _, _, _ = self.names['model_' + i].compute_messages(mydata_xdx[np.array(mydata_xdx).shape[0]-1], marginal=slice(0, 4))
			i = self.state_num - 1
			temp = []
			for j in range(40*i, self.names['alpha_hsmm_' + self.letter_id_poss[-1][0]].shape[1]):
				alpha_letters = [self.names['alpha_hsmm_' + self.letter_id_poss[-1][0]][i][j]]
				all_letters = self.letter_id_poss[-1][1:]
				for k in all_letters:
					alpha_letters += [self.names['alpha_hsmm_' + k][i][j]]						
				alpha_letters = [value for value in alpha_letters if not math.isnan(value)]
				for n in self.letter_id_poss[-1]:
					if max(alpha_letters) == self.names['alpha_hsmm_' + n][i][j]:
						temp.append(n)
			if np.array(self.most_frequent(temp)).size != 0:
				self.letter_id_poss += [self.most_frequent(temp)]
			self.demo_pred = self.letter_id_poss[-1]
			print('The predicted letter is: %s' %self.demo_pred)
		time_end_pred = time.time()
		print("The prediction delay is: %.5fs" %(time_end_pred - time_start_pred))

		# Start to reproduce the trajectory
		if len(self.demo_pred) == 1 and not self.stop_pred:
			self.stop_pred = 1
			while 1:
				if not self.qos:
					print('Qos is too low, reproduction based on intention recognition started...')
					self.stop_direct_tele = 1
					print('The direct teleoperation is stopped...')
					self.letter_id_poss = [self.letters]
					time_start_repro = time.time()
					self.demo_data = self.create_hsmm_data(self.demo_pred)
					self.xi = self.hsmm_lqr(self.demo_pred, self.demo_data)
					time_end_repro = time.time()
					print("The reproduction delay is: %.5fs" %(time_end_repro - time_start_repro))
					# self.repro = self.cpd_repro_affine()
					self.repro = self.cpd_repro_rigid()
					self.finish_reproduction()
					break
				else:
					print('QoS is good, keep direct teleoperation...')
					# self.demo_event_haptic()
					continue

	def separate_xdx(self, mydata_xdx):
		mydata_x = []
		mydata_dx = []
		if np.array(mydata_xdx).ndim == 3:
			for i in range(len(mydata_xdx)):
				temp_x = []
				temp_dx = []
				for j in range(len(mydata_xdx[i])):
					temp_x.append([mydata_xdx[i][j][0], mydata_xdx[i][j][1]])
					temp_dx.append([mydata_xdx[i][j][2], mydata_xdx[i][j][3]])
				temp_x1 = np.array(temp_x)
				temp_dx1 = np.array(temp_dx)
				mydata_x.append(temp_x1)
				mydata_dx.append(temp_dx1)
			return mydata_x, mydata_dx
		elif np.array(mydata_xdx).ndim == 2:
			temp_x = []
			temp_dx = []
			for i in range(len(mydata_xdx)):
				temp_x.append([mydata_xdx[i][0], mydata_xdx[i][1]])
				temp_dx.append([mydata_xdx[i][2], mydata_xdx[i][3]])
			return temp_x, temp_dx
		else:
			print("Error! The dimension of data is neither 2 or 3")
    
	def hsmm_lqr(self, demo_pred, demo_data):

		demo_data_x, demo_data_dx = self.separate_xdx(self.demo_data)

		alpha_hsmm, _, _, _, _ = self.names['model_' + demo_pred[0]].compute_messages(marginal=[], sample_size=demo_data_x[0].shape[0])
		# alpha_hsmm, _, _, _, _ = names['model_' + demo_pred].compute_messages(marginal=[], sample_size=200)

		sq = np.argmax(alpha_hsmm, axis=0)

		A, b = pbd.utils.get_canonical(2, 2, 0.1)

		lqr = pbd.LQR(A, b, horizon=demo_data[0].shape[0])
		# lqr = pbd.LQR(A, b, horizon=200)
		lqr.gmm_xi = self.names['model_' + demo_pred[0]], sq
		lqr.gmm_u = -4.
		lqr.ricatti()

		xi, _ = lqr.get_seq(demo_data[0][0])

#      find the nearest point of xi to mydata_xdx[0][-1]
		# temp = 99999
		# mydata_xdx = self.translate_demo_data_repro()
		# mydata_half_x, _ = np.array(self.separate_xdx(mydata_xdx))
		# self.preds = mydata_half_x.tolist()
		# for i in range(len(xi)):
		# 	#diff = self.cal_distance(xi[i][0:2], mydata_xdx[0][-1][0:2])
		# 	diff = self.cal_distance(xi[i][0:2], mydata_xdx[-1][0:2])
		# 	if diff <= temp:
		# 		point_id = i
		# 		temp = diff
		# xi = xi[point_id:]
		return xi

	def getClosestId(self, point, dataset):
		temp = 99999
		for i in range(len(dataset)):
			diff = self.cal_distance(dataset[i][0:2], point[0:2])
			if diff <= temp:
				point_id = i
				temp = diff
		return point_id

	def cpd_repro_affine(self):
		repro_x, repro_dx = np.array(self.separate_xdx(self.xi))
		data_half = self.translate_demo_data_repro()
		# print(repro_x)
		#data_half = self._current_demo
		data_half_x, data_half_dx = np.array(self.separate_xdx(data_half))
		# print(data_half_x)

		#X = data_half_x[0]
		X = data_half_x
		# X = np.unique(X.tolist(),axis=0)
		# print(X)
		self.preds = data_half_x.tolist()
		point_id = self.getClosestId(X[-1][0:2], self.xi)
		Y = repro_x[:point_id]
		# print(X[-1])
		# print(point_id)
		
		reg = AffineRegistration(**{'X': X, 'Y': Y})
		reg.register()
		res_B, res_t = reg.get_registration_parameters()
		res = np.dot(repro_x, res_B) + np.tile(res_t, (repro_x.shape[0], 1))
		point_id = self.getClosestId(X[-1][0:2], res)
		res = res[point_id:]
		# res = res[point_id-1:]

		return res	

	def cpd_repro_rigid(self):
		repro_x, repro_dx = np.array(self.separate_xdx(self.xi))
		data_half = self.translate_demo_data_repro()
		# print(repro_x)
		#data_half = self._current_demo
		data_half_x, data_half_dx = np.array(self.separate_xdx(data_half))
		# print(data_half_x)

		#X = data_half_x[0]
		X = data_half_x
		# X = np.unique(X.tolist(),axis=0)
		# print(X)
		self.preds = data_half_x.tolist()
		point_id = self.getClosestId(X[-1][0:2], self.xi)
		Y = repro_x[:point_id]
		# print(X[-1])
		# print(point_id)
		
		reg = RigidRegistration(**{'X': X, 'Y': Y})
		reg.register()
		res_s, res_R, res_t = reg.get_registration_parameters()
		res = res_s * np.dot(repro_x, res_R) + res_t
		point_id = self.getClosestId(X[-1][0:2], res)
		res = res[point_id:]
		# res = res[point_id-1:]

		translation_x = X[-1][0] - res[0][0]
		translation_y = X[-1][1] - res[0][1]
		for i in range(len(res)):
			res[i][0] = res[i][0] + translation_x
			res[i][1] = res[i][1] + translation_y

		return res

	# def incremental_refine(self):
		
	def highlight_demos(self):
		data = self.demos['x'][self.params['current_demo'][0]]
		self.plots['current_demo'].set_data(data[:, 0], data[:, 1] )
		self.fig.canvas.draw()

	def plot_sensor_value(self, s, scale=1.):
		data = np.vstack([self.x + np.array([0., 1.]) * s * scale, self.x])
		data -= 5. * np.array([0., 1.])[None]

		self.plots['sensor_value'].set_data(data[:, 0], data[:, 1])
# 		self.plots['prediction_value'].set_data(data[:, 0], data[:, 1])

	def set_plots(self):
		self.plots.update({
			'robot_plot':self.ax_x.plot([], [], 'o-', mew=4, mec='orangered', ms=10, mfc='w')[0],
			'sensor_value':self.ax_x.plot([], [],ls=(0,(1,1)), lw=10)[0],
			'attractor_plot':self.ax_x.plot([], [], 'o-', mew=4, mec='teal', ms=10, mfc='w')[0],
			'obj_plot':self.ax_x.plot([], [], 'o-', mew=4, mec='steelblue', ms=10, mfc='w')[0],
			'current_demo': self.ax_x.plot([], [], lw=3, ls='--', color='orangered')[0],
			'current_demo_dx': self.ax_dx.plot([], [], lw=3, ls='--', color='orangered')[0],
            
			'prediction_plot': self.ax_pre.plot([], [], 'o-', mew=4, mec='orangered', ms=10, mfc='w')[0],
# 			'prediction_value':self.ax_pre.plot([], [],ls=(0,(1,1)), lw=10)[0],
			'prediction_demo': self.ax_pre.plot([], [], lw=3, ls='--', color='orangered')[0]
		})

		for ax, lim in zip([self.ax_x, self.ax_dx, self.ax_pre], [150, 25, 150]):
			ax.set_xlim([-lim, lim])
			ax.set_ylim([-lim, lim])

	# def sim_dynamics(self, ffx, n_steps=10):
	# 	if not self.velocity_mode:
	# 		m = 1.0

	# 		ddx = ffx/m
	# 		self.x += self.dt / n_steps * self.dx + 0.5 * self.ddx * (self.dt / n_steps) ** 2
	# 		self.dx += self.dt / n_steps * 0.5 * (self.ddx + ddx)
	# 		self.dxx = np.copy(ddx)
	# 	else:
	# 		kp = 0.;
	# 		kv = kp ** 0.5 * 2
	# 		for i in range(50):
	# 			ddx = kp * (self.curr_mouse_pos - self.dx)
	# 			self.dx += self.dt * ddx
	# 			self.x += self.dt * self.dx + (self.dt ** 2) / 2. * ddx


	# def timer_event(self, event):
	# 	if self.is_demonstrating and not self.stop_pred: 
	# 		# if self.curr_mouse_pos is None: self.pretty_print('Outside'); return

	# 		# kp = 400.;
	# 		# kv = kp ** 0.5 * 2

	# 		# n_steps = 10
	# 		# for i in range(n_steps):
	# 		# 	ffx = kp * (self.curr_mouse_pos - self.x) - kv * self.dx
	# 		# 	self.sim_dynamics(ffx)

	# 		# self.curr_demo += [np.copy(self.x)]; self.curr_demo_dx += [np.copy(self.dx)]

	# 		# self._current_demo['x'] += [np.copy(self.x)]
	# 		# self._current_demo['dx'] += [np.copy(self.dx)]

	# 		# if not self.first_event:
	# 		# 	self._current_demo['x'] += [np.copy([self.mouse_points.position.x, self.mouse_points.position.y])]
	# 		# 	self._current_demo['dx'] += [np.copy(self.dx)]
	# 		# 	self.mouse_points_x_old = self.mouse_points.position.x
	# 		# 	self.mouse_points_y_old = self.mouse_points.position.y
	# 		# else:
	# 		# 	if self.mouse_points_x_old != self.mouse_points.position.x or self.mouse_points_y_old != self.mouse_points.position.y:
	# 		# 		self._current_demo['x'] += [np.copy([self.mouse_points.position.x, self.mouse_points.position.y])]
	# 		# 		self.mouse_points_x_old = self.mouse_points.position.x
	# 		# 		self.mouse_points_y_old = self.mouse_points.position.y
	# 		# 		self._current_demo['dx'] += [np.copy(self.dx)]

	# 		# self.first_event = 1

	# 		# print(len(self._current_demo['x']))

	# 		self._current_demo['x'] += [np.copy([self.mouse_points.position.x, self.mouse_points.position.y])]
	# 		self._current_demo['dx'] += [np.copy(self.dx)]
			
	# 		if np.array(self._current_demo['x']).shape[0] % 20 == 0 and np.array(self._current_demo['x']).shape[0] > 0 and not self.stop_pred:
	# 			# print(self._current_demo['x'])
	# 			self.state_num = np.array(self._current_demo['x']).shape[0] / 20
	# 			thread = threading.Thread(target=self.likelihood_prediction)
	# 			thread.start()

	# def demo_event(self):

	# 	self._current_demo['x'] += [np.copy([self.mouse_points.position.x, self.mouse_points.position.y])]
	# 	self._current_demo['dx'] += [np.copy(self.dx)]
	# 	# print(len(self._current_demo['x']))
		
	# 	if np.array(self._current_demo['x']).shape[0] % 40 == 0 and np.array(self._current_demo['x']).shape[0] > 0 and not self.stop_pred:
	# 		# print(self._current_demo['x'])
	# 		self.state_num = np.array(self._current_demo['x']).shape[0] / 40
	# 		thread = threading.Thread(target=self.likelihood_prediction)
	# 		thread.start()

	def demo_event_haptic(self):

		self.haptic_x = self.scaling_haptic * (-self.haptic_pose.position[1])
		self.haptic_y = self.scaling_haptic * self.haptic_pose.position[0]
		self._current_demo['x'] += [np.copy([self.haptic_x, self.haptic_y])]
		self._current_demo['dx'] += [np.copy(self.dx)]
		# print(len(self._current_demo['x']))
		
		if np.array(self._current_demo['x']).shape[0] % 40 == 0 and np.array(self._current_demo['x']).shape[0] > 0 and not self.stop_pred:
			# print(self._current_demo['x'])
			self.state_num = np.array(self._current_demo['x']).shape[0] / 40
			thread = threading.Thread(target=self.likelihood_prediction)
			thread.start()

	def move_event(self, event):
		self.curr_mouse_pos = None if None in [event.xdata, event.ydata] else np.array([event.xdata, event.ydata])

		if event.key == 'shift' or self.is_demonstrating:
			self.robot_pos = np.copy(self.curr_mouse_pos)

			if not self.is_demonstrating:
				self.plots['robot_plot'].set_data(self.robot_pos[0], self.robot_pos[1])
                
				self.plots['prediction_plot'].set_data(self.robot_pos[0], self.robot_pos[1])
				self.fig.canvas.draw()

	def plot_timer_event(self, event):
		# self.robot_pos = self.curr_mouse_pos if self.robot_pos is None else self.robot_pos
		# self.plots['attractor_plot'].set_data(self.robot_pos[0], self.robot_pos[1])
		# self.plots['robot_plot'].set_data(self.x[0], self.x[1])
        
		# self.plots['prediction_plot'].set_data(self.x[0], self.x[1])

		if self.is_demonstrating:
			curr_demo_arr = np.array(self._current_demo['x'])
			curr_demo_dx_arr = np.array(self._current_demo['dx'])

			# print(curr_demo_arr)

			self.plots['current_demo'].set_data(curr_demo_arr[:, 0],
												curr_demo_arr[:, 1])
			if not self.stop_pred:
            
				self.plots['prediction_demo'].set_data(curr_demo_arr[:, 0],
												curr_demo_arr[:, 1])

			self.plots['current_demo_dx'].set_data(curr_demo_dx_arr[:, 0],
												curr_demo_dx_arr[:, 1])
			self.fig.canvas.draw()

	# def click_event(self, event):
	# 	if event.key is None:
	# 		self.pretty_print('Demonstration started')
	# 		self.stop_pred = 0
	# 		self.is_demonstrating = True
	# 		self.velocity_mode = event.inaxes == self.ax_dx
	# 		if not self.velocity_mode:
	# 			self.x = self.curr_mouse_pos
	# 		else:
	# 			self.x = self.demos['x'][-1][0] if self.nb_demos > 0 else np.array([0., 0.])
	# 			self.dx = self.curr_mouse_pos

	# 		[t.start() for t in [self.timer, self.plot_timer]]
	# 		# [t.start() for t in [self.timer, self.plot_timer, self.prediction_timer]]
            
	def start_system(self):
		print('The system is started...')
		self.is_demonstrating = True
		self.stop_pred = 0
		# self.first_event = 0
		# [t.start() for t in [self.timer, self.plot_timer]]
		[t.start() for t in [self.plot_timer]]
		# self.demo_event()
		self.demo_event_haptic()

	#def end_system(self):



	# def release_event(self, event):
	# 	if event.key is None:
	# 		self.pretty_print('Demonstration finished')
	# 		self.is_demonstrating = False
	# 		self.finish_demo()

	# 		[t.stop() for t in [self.timer, self.plot_timer]]
	# 		# [t.stop() for t in [self.timer, self.plot_timer, self.prediction_timer]]           

	def replot_demos(self):
		for i in range(self.nb_demos):
			data = self.demos['x'][i]
			self.plots['demo_%d' % i] = \
			self.ax_x.plot(data[:, 0], data[:, 1], lw=2, ls='--')[0]
            
			data = self.demos['x'][i]
			self.plots['demo_prediction_%d' % i] = \
			self.ax_pre.plot(data[:, 0], data[:, 1], lw=2, ls='--')[0]
            
			data = self.demos['dx'][i]
			self.plots['demo_dx_%d' % i] = \
			self.ax_dx.plot(data[:, 0], data[:, 1], lw=2, ls='--')[0]

	def clear_demos(self, last=False, selected=False):
		"""
		:param last: 	 [bool]
			Delete only last one
		"""
		if last or selected:
			idx = -1 if last else self.params['current_demo'][0]

			for s in self.demos:
				self.demos[s].pop(idx)

			for i in range(self.nb_demos):
				self.plots['demo_%d' % (i)].remove()
				self.plots['demo_prediction_%d' % (i)].remove()
				self.plots['demo_dx_%d' % (i)].remove()

			self.nb_demos = len(self.demos['x']); self.params['current_demo'][2] = self.nb_demos - 1

			self.replot_demos()

			if selected:
				self.plots['current_demo'].set_data([], [])
				self.plots['prediction_demo'].set_data([], [])

			self.fig.canvas.draw()
		else:
			for i in range(self.nb_demos):
				self.plots['demo_%d' % i].remove()
				self.plots['demo_prediction_%d' % i].remove()
				self.plots['demo_dx_%d' % i].remove()
			for i in range(len(self.xi[:, 0])):
				#self.plots['demo_prediction_step_%d' % i].remove()
				self.plots['demo_reproduction_step_%d' %i].remove()	

			#self.plots['demo_reproduction'].remove()
			self.fig.canvas.draw()

			for s in self.demos:
				self.demos[s] = []
			self.nb_demos = 0; self.params['current_demo'][2] = self.nb_demos - 1

	def finish_demo(self):
		"""
		Called when finishing a demonstration to store the data
		:return:
		"""
		curr_demo_arr = np.array(self._current_demo['x'])
		curr_demo_dx_arr = np.array(self._current_demo['dx'])

		# self.demos['x'] += [curr_demo_arr]; self.demos['dx'] += [curr_demo_dx_arr]
		# self.curr_demo = []; self.curr_demo_dx = []

		for s in self._current_demo:
			self.demos[s] += [np.array(self._current_demo[s])]
			self._current_demo[s] = []

		self.plots['current_demo'].set_data([], []); self.plots['current_demo_dx'].set_data([], [])
		self.plots['demo_%d' % self.nb_demos] = self.ax_x.plot(curr_demo_arr[:, 0], curr_demo_arr[:, 1], lw=2, ls='--')[0]
		self.plots['demo_dx_%d' % self.nb_demos] = self.ax_dx.plot(curr_demo_dx_arr[:, 0], curr_demo_dx_arr[:, 1], lw=2, ls='--')[0]
        
		#self.plots['prediction_demo'].set_data([], [])
		#self.plots['demo_prediction_%d' % self.nb_demos] = self.ax_pre.plot(curr_demo_arr[:, 0], curr_demo_arr[:, 1], lw=2, ls='--')[0]

		self.nb_demos += 1; self.params['current_demo'][2] = self.nb_demos - 1

		self.fig.canvas.draw()

	def translate_demo_data(self):
		for s in self._current_demo:
			# self.mydemos[s] += self._current_demo[s]
			self.mydemos[s] += [np.array(self._current_demo[s])]
			# self._current_demo[s] = []

		data = np.array(self.mydemos).tolist()
		data_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(data['x'], data['dx'])]
		# print(data_xdx)
		return data_xdx

	def translate_demo_data_repro(self):

		data = np.array(self._current_demo).tolist()
		data_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(data['x'], data['dx'])]
		return data_xdx

	def reproduction_plot(self):
#		xi_pre, _ = np.array(self.separate_xdx(self.xi))
		self.plots['prediction_demo'].set_data(self.repro[:, 0], self.repro[:, 1])
		self.fig.canvas.draw()

	def finish_reproduction(self):

		curr_demo_arr = np.array(self._current_demo['x'])
		curr_demo_dx_arr = np.array(self._current_demo['dx'])
		self.plots['prediction_demo'].set_data([], [])
		self.plots['demo_prediction_%d' % self.nb_demos] = self.ax_pre.plot(curr_demo_arr[:, 0], curr_demo_arr[:, 1], lw=2, ls='--')[0]
		
		# for i in range(len(self.xi[:, 0])):
		# 	#self.plots['prediction_demo'].set_data(self.repro[i, 0], self.repro[i, 1])
		# 	self.plots['demo_reproduction_step_%d' %i] = self.ax_pre.plot(self.xi[i, 0], self.xi[i, 1], 'ro')[0]
		# 	self.fig.canvas.draw()
		# 	#time.sleep(0.001)
		# print('Reproduction finished!')

		for i in range(len(self.repro[:, 0])):
			#self.plots['prediction_demo'].set_data(self.repro[i, 0], self.repro[i, 1])
			# rate = rospy.Rate(10)
			result = Pose()

			# In x-z coordinate
			# # result.position.x = self.scaling * self.repro[i, 0] + 0.06
			# result.position.x = self.scaling_robot * self.repro[i, 0] - 0.16
			# # result.position.z = self.scaling * self.repro[i, 1] + 0.7
			# result.position.z = self.scaling_robot * self.repro[i, 1] + 0.64
			# result.position.y = 0.0564058
			# result.orientation.x = -0.500128
			# result.orientation.y = 0.510553
			# result.orientation.z = 0.491395
			# result.orientation.w = 0.497736

			# In x-y coordinate
			result.position.x = self.scaling_robot * self.repro[i, 0] -0.143
			result.position.y = self.scaling_robot * (-self.repro[i, 1]) + 0.0670786					
			# result.position.z = 0.859739
			result.position.z = 0.89
			result.orientation.x = -0.00766462
			result.orientation.y = 0.0106087
			result.orientation.z = 0.701056
			result.orientation.w = 0.712987		

			# self.pub_repro.publish(result)
			self.pub_demo_robot.publish(result)
			# rate.sleep()
			self.plots['demo_reproduction_step_%d' %i] = self.ax_pre.plot(self.repro[i, 0], self.repro[i, 1], 'ro')[0]
			self.fig.canvas.draw()
		self.res_finish_pub = 1
		self.pub_finish.publish(self.res_finish_pub)
		print('Reproduction finished!')
   
	def save_demos(self):
		"""
		Saving demonstrations with filename prompt
		:return:
		"""
		root = tk.Tk(); root.withdraw()

		file_path = asksaveasfilename(initialdir=self.path, initialfile=self.filename + '.npy')

#		res = self.demos['x'][0].tolist()
		for i in range(len(self.xi[:, 0])):
			self.preds.append([self.xi[i, 0], self.xi[i, 1]])

		self.pretty_print("Demonstrations saved as\n "+ file_path)

		#np.save(file_path, self.demos)

		np.save(file_path, self.preds)

		pass
