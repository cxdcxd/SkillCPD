import numpy as np
import os
import time
import random
import threading
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

		gs = gridspec.GridSpec(1, 2)

# 		gs = gridspec.GridSpec(1, 3)

		self.filename = filename

		self.ax_x = plt.subplot(gs[0])
		self.ax_dx = plt.subplot(gs[1])

		self.set_events()
		self.set_plots()

		global is_demonstrating
		self.is_demonstrating = False
		self.velocity_mode = False
# 		self.is_predicting = False

		self.curr_demo, self.curr_demo_dx = [], []
		self.curr_demo_obj, self.curr_demo_obj_dx = [], []

		self._current_demo = {'x': [], 'dx': []}

		self.curr_mouse_pos = None
		self.robot_pos = np.zeros(2)

		self.nb_demos = 0
		self.demos = {'x': [], 'dx': []}

		self.params.update({'current_demo': [0, 0, self.nb_demos]})

		self.loaded = False
		#
		# win = plt.gcf().canvas.manager.window
		#
		# win.lift()
		# win.attributes("-topmost", True)
		# win.attributes("-alpha", 0.4)


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

	def DataProcessing(self, data):
		data = np.array(data)
		print(data)
		print(data.shape)
		nsamples, nx, ny = data.shape
		if nx > 200:                                                # delete randomly datapoints
			index = random.sample(range(1,nx), nx-200)
			demo_data = np.delete(data, index, axis=1)
			demo_data_dim2 = demo_data.reshape((nsamples,-1))
			return demo_data_dim2
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
			return demo_data_dim2
		else:
			return data
        
	def knn_classifier(self):
		all_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
		labels = []
		for letter in all_letters:
			data = np.load('./DemoData/%s.npy'%(letter))
			labels.extend(['%s'%(letter)]*len(data))
		labels = np.array(labels)
		training_data = np.load('./DemoData/A.npy')
		all_letters.pop(0)
		for letter in all_letters:
			training_data2 = np.load('./DemoData/%s.npy'%(letter))
			training_data = np.vstack((training_data,training_data2))

		nsamples, nx, ny = training_data.shape
		training_data_dim2 = training_data.reshape((nsamples,nx*ny))
		data_train,data_test,labels_train,labels_test = train_test_split(training_data_dim2,labels,test_size=0.1)
		knn = KNeighborsClassifier()
		knn.fit(data_train,labels_train)
		labels_pred = knn.predict(data_test)
		return knn, (accuracy_score(labels_test,labels_pred)*100)
   
	def online_prediction(self):
		# Take Data from record
		time_start = time.time()
		mydata = np.array(self.translate_demo_data())
# 		mydata = np.array(self.demos)
		mydata = mydata.tolist()
		mydata_xdx = [np.hstack([_x, _dx]) for _x ,_dx in zip(mydata['x'], mydata['dx'])]

		# Online prediction
		data_dim2 = self.DataProcessing(mydata_xdx)
		knn, accuracy = self.knn_classifier()
		print("The accuracy of predition with test data is %.2f%%" %accuracy)
		demo_pred = knn.predict(data_dim2)
		time_end = time.time()
		print("The prediction delay is: %.5fs" %(time_end-time_start))
		print("The prediction of the given Data is: %s" %demo_pred[0])
        
        
	def highlight_demos(self):
		data = self.demos['x'][self.params['current_demo'][0]]
		self.plots['current_demo'].set_data(data[:, 0], data[:, 1] )
		self.fig.canvas.draw()

	def plot_sensor_value(self, s, scale=1.):
		data = np.vstack([self.x + np.array([0., 1.]) * s * scale, self.x])
		data -= 5. * np.array([0., 1.])[None]

		self.plots['sensor_value'].set_data(data[:, 0], data[:, 1])

	def set_plots(self):
		self.plots.update({
			'robot_plot':self.ax_x.plot([], [], 'o-', mew=4, mec='orangered', ms=10, mfc='w')[0],
			'sensor_value':self.ax_x.plot([], [],ls=(0,(1,1)), lw=10)[0],
			'attractor_plot':self.ax_x.plot([], [], 'o-', mew=4, mec='teal', ms=10, mfc='w')[0],
			'obj_plot':self.ax_x.plot([], [], 'o-', mew=4, mec='steelblue', ms=10, mfc='w')[0],
			'current_demo': self.ax_x.plot([], [], lw=3, ls='--', color='orangered')[0],
			'current_demo_dx': self.ax_dx.plot([], [], lw=3, ls='--', color='orangered')[0]
		})

		for ax, lim in zip([self.ax_x, self.ax_dx], [100, 25]):
			ax.set_xlim([-lim, lim])
			ax.set_ylim([-lim, lim])

	def sim_dynamics(self, ffx, n_steps=10):
		if not self.velocity_mode:
			m = 1.0

			ddx = ffx/m
			self.x += self.dt / n_steps * self.dx + 0.5 * self.ddx * (
																	 self.dt / n_steps) ** 2
			self.dx += self.dt / n_steps * 0.5 * (self.ddx + ddx)
			self.dxx = np.copy(ddx)
		else:
			kp = 0.;
			kv = kp ** 0.5 * 2
			for i in range(50):
				ddx = kp * (self.curr_mouse_pos - self.dx)
				self.dx += self.dt * ddx
				self.x += self.dt * self.dx + (self.dt ** 2) / 2. * ddx


	def timer_event(self, event):
		if self.is_demonstrating:
			if self.curr_mouse_pos is None: self.pretty_print('Outside'); return

			# print self.x, self.dx
			kp = 400.;
			kv = kp ** 0.5 * 2

			n_steps = 10
			for i in range(n_steps):
				ffx = kp * (self.curr_mouse_pos - self.x) - kv * self.dx
				self.sim_dynamics(ffx)


			# self.curr_demo += [np.copy(self.x)]; self.curr_demo_dx += [np.copy(self.dx)]

			self._current_demo['x'] += [np.copy(self.x)]
			self._current_demo['dx'] += [np.copy(self.dx)]

	def move_event(self, event):
		self.curr_mouse_pos = None if None in [event.xdata, event.ydata] else np.array([event.xdata, event.ydata])

		if event.key == 'shift' or self.is_demonstrating:
			self.robot_pos = np.copy(self.curr_mouse_pos)

			if not self.is_demonstrating:
				self.plots['robot_plot'].set_data(self.robot_pos[0], self.robot_pos[1])
				self.fig.canvas.draw()

	def plot_timer_event(self, event):
		self.robot_pos = self.curr_mouse_pos if self.robot_pos is None else self.robot_pos
		self.plots['attractor_plot'].set_data(self.robot_pos[0], self.robot_pos[1])
		self.plots['robot_plot'].set_data(self.x[0], self.x[1])

		if self.is_demonstrating:
			curr_demo_arr = np.array(self._current_demo['x'])
			curr_demo_dx_arr = np.array(self._current_demo['dx'])

			self.plots['current_demo'].set_data(curr_demo_arr[:, 0],
												curr_demo_arr[:, 1])

			self.plots['current_demo_dx'].set_data(curr_demo_dx_arr[:, 0],
												curr_demo_dx_arr[:, 1])
			self.fig.canvas.draw()

	def click_event(self, event):
		if event.key is None:
			self.pretty_print('Demonstration started')
			self.velocity_mode = event.inaxes == self.ax_dx
			self.is_demonstrating = True
			if not self.velocity_mode:
				self.x = self.curr_mouse_pos
			else:

				self.x = self.demos['x'][-1][0] if self.nb_demos > 0 else np.array([0., 0.])
				self.dx = self.curr_mouse_pos


			[t.start() for t in [self.timer, self.plot_timer]]
            
# 		self.start_predicting()
		second_thread = threading.Thread(target = self.start_predicting)
		second_thread.start()
            
	def start_predicting(self):
		global is_demonstrating
		while self.is_demonstrating:
			time.sleep(5)
			self.online_prediction()
        
# 	def stop_predicting(self, event):
# 		time.sleep(1)
# # 		print('hello world!')
# 		self.online_prediction() 

	def release_event(self, event):
		if event.key is None:
			self.pretty_print('Demonstration finished')
			self.is_demonstrating = False
			self.finish_demo()

			[t.stop() for t in [self.timer, self.plot_timer]]
# 		self.online_prediction()
# 		self.stop_predicting()

	def replot_demos(self):
		for i in range(self.nb_demos):
			data = self.demos['x'][i]
			self.plots['demo_%d' % i] = \
			self.ax_x.plot(data[:, 0], data[:, 1], lw=2, ls='--')[0]
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
				self.plots['demo_dx_%d' % (i)].remove()

			self.nb_demos = len(self.demos['x']); self.params['current_demo'][2] = self.nb_demos - 1

			self.replot_demos()

			if selected:
				self.plots['current_demo'].set_data([], [])

			self.fig.canvas.draw()
		else:
			for i in range(self.nb_demos):
				self.plots['demo_%d' % i].remove()
				self.plots['demo_dx_%d' % i].remove()

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

		self.nb_demos += 1; self.params['current_demo'][2] = self.nb_demos - 1

		self.fig.canvas.draw()


	def translate_demo_data(self):
		return self.demos
    

	def save_demos(self):
		"""
		Saving demonstrations with filename prompt
		:return:
		"""
		root = tk.Tk(); root.withdraw()

		file_path = asksaveasfilename(initialdir=self.path, initialfile=self.filename + '.npy')

		self.pretty_print("Demonstrations saved as\n "+ file_path)

		np.save(file_path, self.demos)

		pass