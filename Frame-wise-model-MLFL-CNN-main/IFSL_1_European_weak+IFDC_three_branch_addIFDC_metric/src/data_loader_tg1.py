import os
import numpy as np
import random
import configparser
import pandas as pd

def read_lst(lst):
		""""
		Read multiple lines from a file and convert them to a list.
		(a general tool)
		Args:
			lst: list
				the path of a file to read
				Return:
			files: list
				multiple file ids from the file
			f_len: integer
				the length of the list to return
				"""
		with open(lst) as f:
			files = f.readlines()
		files = [f.rstrip() for f in files]
		f_len = len(files)
		return files, f_len


class data_loader(object):
	def __init__(self, conf_dir):
		""""
		Load the data stream for training.
		Count relevant parameters on the data set for DF and GL.
		Args:
			conf_dir: string
				the path of configuration dir   
		Attributes:
			conf_dir
			feature_dir
			label_dir
			train_lst
			vali_lst
			test_lst
			vali_csv
			win_len_csv
			LEN
			DIM
			CLASS
			batch_size
			events
			dinsentangle_n
			dinsentangle_m
			ratio_for_win_len
			ep_per_epochs
		Interface:
			init_data_conf: Initialize most of attribute values from data configuration file.
			get_train: Get the file list of the training set.
			get_vali: Get the file list and the groundtruths of the validation set.
			get_test: Get the file list and the groundtruths of the test set.
			count_disentangle: Count the coefficient for the DF dimension per class.
			count_win_len_per_class: Count a group of adaptive window sizes for the median filters.
			generator_train: Generate a generator for training.
			generator_vali: Generate data from vali_lst.
			generator_test: Generate data from test_lst.
			generator_weak: Generate data from weak_lst.
			generator_all: To generate feature data list and label data list for test_lst, vali_lst or weak_lst.
			generator: Generate a generator for prediction.

		"""
		self.conf_dir = conf_dir
		self.init_data_conf()

	def init_data_conf(self):
		""""
		Initialize most of attribute values from data configuration file.
		Args:
		Return:

		"""
		conf_dir = self.conf_dir
		data_cfg_path = os.path.join(conf_dir, 'data.cfg')
		assert os.path.exists(data_cfg_path)
		config = configparser.ConfigParser()
		config.read(data_cfg_path)

		assert 'path' in config.sections()
		path_cfg = config['path']

		
		self.feature_dir = path_cfg['feature_dir']
		self.label_dir = path_cfg['label_dir']
		self.train_lst = path_cfg['train_lst']
		self.unlabel_lst = path_cfg['unlabel_lst']
		self.syn_lst = path_cfg['syn_lst']
		self.vali_lst = path_cfg['vali_lst']
		self.vali_syn_lst = path_cfg['vali_syn_lst']
		self.test_lst = path_cfg['test_lst']
		self.vali_csv = path_cfg['vali_csv']
		self.detail_weak_unlabel_dir = path_cfg['detail_weak_unlabel_dir']
		self.test_csv = path_cfg['test_csv']
		self.vali_dur_csv = path_cfg['vali_dur_csv']
		self.test_dur_csv = path_cfg['test_dur_csv']
		self.win_len_csv = path_cfg['win_len_csv']

		files = [self.feature_dir, 
			self.label_dir, 
			self.detail_weak_unlabel_dir,
			self.train_lst, 
			self.unlabel_lst,
			self.syn_lst, 
			self.test_lst, 
			self.vali_lst, 
			self.vali_syn_lst, 
			self.test_csv, 
			self.vali_csv, 
			self.test_dur_csv,
			self.vali_dur_csv,
			self.win_len_csv]
	
		#ensure that all the paths are valid
		for  f in files:
			assert os.path.exists(f)

		assert 'parameter' in config.sections()
		parameter_cfg = config['parameter']
		self.LEN = int(parameter_cfg['LEN'])
		self.DIM = int(parameter_cfg['DIM'])
		self.batch_size = int(parameter_cfg['batch_size'])
		self.dinsentangle_n = int(parameter_cfg['dinsentangle_n'])
		self.dinsentangle_m = float(parameter_cfg['dinsentangle_m'])
		self.ratio_for_win_len = float(parameter_cfg['ratio_for_win_len'])
		self.ep_per_epochs = float(parameter_cfg['ep_per_epochs'])
		self.exponent = float(parameter_cfg['exponent'])
		self.start_epoch = int(parameter_cfg['start_epoch'])

		assert'events' in config.sections()
		event_cfg = config['events']
		
		self.events = event_cfg['events'].split(',')
		self.CLASS = len(self.events)

	
	def get_train(self):
		""""
		Get the file list of the training set.
		Args:
		Return:
			lst: list
				multiple file ids from the train_lst
		"""
		lst, _ = read_lst(self.train_lst)
		return lst

	def get_unlabel(self):
		""""
		Get the file list of the training set.
		Args:
		Return:
			lst: list
				multiple file ids from the train_lst
		"""
		lst, _ = read_lst(self.unlabel_lst)
		return lst

	def get_vali(self):
		""""
		Get the file list and the groundtruths of the validation set.
		Args:
		Return:
			lst: list
				multiple file ids from the vali_lst
			csv: list
				multiple strong groundtruths from the vali_csv
		"""
		print("get_vali")
		lst, _ = read_lst(self.vali_lst)
		csv, _ = read_lst(self.vali_csv)
		dur_csv = pd.read_csv(self.vali_dur_csv, sep='\t')
		return lst, csv, dur_csv

	def get_test(self):
		""""
		Get the file list and the groundtruths of the test set.
		Args:
		Return:
			lst: list
				multiple file ids from the test_lst
			csv: list
				multiple strong groundtruths from the test_csv
		"""
		lst, _ = read_lst(self.test_lst)
		csv, _ = read_lst(self.test_csv)
		dur_csv = pd.read_csv(self.test_dur_csv, sep='\t')
		return lst, csv, dur_csv


	def count_disentangle(self):
		""""
		Count the coefficient for the DF dimension per class.
		coefficient x hidden feature dimension = DF dimension
		Args:
		Return:
			disentangle: list
				a group of coefficients.
		"""
		n = self.dinsentangle_n
		m = self.dinsentangle_m
		CLASS = self.CLASS

		#get the file list of the training set
		lst = self.get_train()
		label_dir = self.label_dir
		detail_weak_unlabel_dir = self.detail_weak_unlabel_dir
		disentangle = np.zeros([CLASS])
		co_occurence = np.zeros([CLASS, CLASS + 1])

		for f in lst:
			path = os.path.join(label_dir, f + '.npy')
			#ignore the unlabeled training data
			if os.path.exists(path):
				label = np.load(path)
				#count the number of the clips containing n event classes in the training set
				co_occ = int(np.sum(label))
				if co_occ > n:
					continue
				co_occurence[:, co_occ] += label
		
		weights = np.zeros([CLASS + 1, 1])
		for i in range(CLASS):
			weights[i + 1, 0] = 1 / (i + 1)

		disentangle = np.matmul(co_occurence, weights)
		disentangle = np.reshape(disentangle, [CLASS])	
		#nomalization
		disentangle = disentangle / np.max(disentangle)

		#prevent too-small DF coefficient
		disentangle = disentangle * (1-m) + m
		return disentangle	

	def count_win_len_per_class(self, top_len):
		""""
		Count a group of adaptive window sizes for the median filters.
		Args:
			top_len: integer
				the sequence length (frames) of the final output of the model
		Return:
			out: list
				the adaptive window sizes of the median filters
				"""
		path = self.win_len_csv
		ratio_for_win_len = self.ratio_for_win_len
		print ("ratio_for_win_len", ratio_for_win_len)

		#get strong label (timestamps) from win_len_csv
		csv, clen = read_lst(path)
		label_cnt = {}
		for event in self.events:
			label_cnt[event] = {'num':0, 'frame':0}

		#get the number of frames per second
		frames_per_second = top_len / 10.0

		#count the total number of frames and total number of occurrences per event class in win_len_csv
		for c in csv:
			cs = c.split('\t')
			if len(cs) < 4:
				continue
			label = cs[3]
			label_cnt[label]['num'] += 1
			label_cnt[label]['frame'] += (
				(float(cs[2])-float(cs[1])) * frames_per_second)

		#count the number of frames per occurrence per event class
		for label in label_cnt:
			print (int(label_cnt[label]['frame']))
			print (int(label_cnt[label]['num']))
			label_cnt[label]['win_len'] = int(label_cnt[label]['frame'] / label_cnt[label]['num'])

		#get adaptive window sizes by multiplying by ratio_for_win_len
		out = []
		for label in label_cnt:
			out += [int(label_cnt[label]['win_len'] * ratio_for_win_len)]
			if out[-1] == 0:
				out[-1] = 1
		return out
		
	
	def generator_train(self):
		""""
		Generate a generator for training.
		Args:
				Return: 
			generator: function 
				that can generate a generator for training
			steps: integer
				steps (the number of batches) per epoch 

				"""
		train_lst = self.train_lst
		unlabel_lst = self.unlabel_lst
		batch_size = self.batch_size
		unlabeled_batch_size = 4
		labeled_batch_size = batch_size - unlabeled_batch_size
		feature_dir = self.feature_dir
		label_dir = self.label_dir
		detail_weak_unlabel_dir = self.detail_weak_unlabel_dir
		CLASS = self.CLASS
		LEN = self.LEN
		DIM = self.DIM

		N = 1 + 4
		start_epoch = self.start_epoch
		exponent = self.exponent
		ep_per_epochs = self.ep_per_epochs

		#get file list from train_lst
		weak_files, weak_f_len = read_lst(train_lst)
		unlabel_files, unlabel_f_len = read_lst(unlabel_lst)
		syn_files, syn_f_len = read_lst(self.syn_lst)

		steps = (weak_f_len * ep_per_epochs + labeled_batch_size // N - 1) // (labeled_batch_size // N)


		#shuffle train_lst
		random.shuffle(weak_files)
		random.shuffle(unlabel_files)
		random.shuffle(syn_files)
			

		def generator():
			#index of file list
			i = 0
			i_unlabel = 0
			i_syn = 0

			#index of a batch
			cur = 0
			#current epoch
			epoch = 0
			#current step in a epoch
			step = 0

			mask_w = 0
			while True:
				#get the ith file of the file list
				f = weak_files[i]
				i = (i + 1)%weak_f_len
				if i == 0:
					random.shuffle(weak_files)
				
				data_f = os.path.join(feature_dir, f + '.npy')

				assert os.path.exists(data_f)

				data = np.load(data_f)
				label_f = os.path.join(label_dir, f + '.npy')
				label = np.load(label_f)

				dwlabel_f = os.path.join(detail_weak_unlabel_dir, f + '.npy')
				dwlabel = np.load(dwlabel_f)

				#else:
					#unlabeled data
				#	label = np.zeros([CLASS])
				#	mask = np.zeros([CLASS])

				#batch begin
				if cur == 0:
					labels = np.zeros([batch_size, CLASS + 1])
#					labels_sep = np.zeros([batch_size, CLASS])
#					labels_not_sep = np.zeros([batch_size, CLASS])
					train_data = np.zeros([batch_size, LEN * 2, DIM])
					detail_labels = np.zeros([batch_size, LEN, CLASS])


				train_data[cur * N] = data
				labels[cur * N, :CLASS] = label
				detail_labels[cur * N, 0] = label
				detail_labels[cur * N] = dwlabel

				for j in range(0, N - 1):
					uf = syn_files[i_syn]
					i_syn = (i_syn + 1) % syn_f_len
					if (i_syn == 0):
						random.shuffle(syn_files)
					udata_f = os.path.join(feature_dir, uf + '.npy')
					detail_label_dir = 'data/detail_label'
					udata = np.load(udata_f)
					ulabel_f = os.path.join(label_dir, uf + '.npy')
					dlabel_f = os.path.join(detail_label_dir, uf + '.npy')
					ulabel = np.load(ulabel_f)
					dlabel = np.load(dlabel_f)
					train_data[cur * N + j + 1] = udata
					#labels[cur * 2 + 1, : CLASS] = ulabel
					labels[cur * N + j + 1, :CLASS] = ulabel
					detail_labels[cur * N + j + 1] = dlabel

				# add unlabeled data
				f = unlabel_files[i_unlabel]
				un_label_f = os.path.join(label_dir, f + '.npy')
				un_label = np.load(un_label_f)
				un_dwlabel_f = os.path.join(detail_weak_unlabel_dir, f + '.npy')
				un_dwlabel = np.load(un_dwlabel_f)
				
				if (i_unlabel == 0):
					random.shuffle(unlabel_files)
				unlabel_data_f = os.path.join(feature_dir, f + '.npy')
				unlabel_data = np.load(unlabel_data_f)
				train_data[labeled_batch_size + cur] = unlabel_data
				labels[labeled_batch_size + cur, :CLASS] = un_label
				detail_labels[labeled_batch_size + cur] = un_dwlabel
				i_unlabel = (i_unlabel + 1) % unlabel_f_len
			
				cur += 1

				#batch end
				if cur == labeled_batch_size // N:
					cur = 0
					#count the weight of unsupervised loss for the PT-model
					if epoch > start_epoch:
						a = 1-np.power(exponent, epoch-start_epoch)
					else:
						a = 0
					#labels_sep = labels_not_sep = labels
					yield train_data, [labels, labels, labels, detail_labels, detail_labels, detail_labels, labels, labels]

					#count current step and epoch
					step += 1
					if step%steps == 0:
						epoch += 1
						if epoch >= 10:
							mask_w = epoch / 100
						step = 0

		return generator, steps

	def generator_vali(self):
		""""
		Generate data from vali_lst.
		Args:
				Return:
						 / : tuple
				feature list and label list of vali_lst

				"""
		return self.generator_all('vali')

	def generator_test(self):
		""""
		Generate data from test_lst.
		Args:
		Return:
			 / : tuple
				feature list and label list of test_lst

				"""
		return self.generator_all('test')

	def generator_weak(self):
		""""
		Generate data from weak_lst.
		Args:
		Return:
			 / : tuple
				feature list and label list of weak_lst

		"""
		return self.generator_all('weak')


	def generator_all(self, mode):
		""""
		To generate feature data list and label data list for test_lst, vali_lst or weak_lst.
		Args:
			mode: string in ['vali','test','weak']
				prediction mode
		Return:
			data: list
				feature data
			labels: list
				label data

		"""
		gt, steps = self.generator(mode)
		gt = gt()
		data = []
		labels = []
		detail_labels = []
		for cnt, (X, Y) in enumerate(gt):
			data += [X]
			if mode == 'vali':
				detail_labels += [Y[3]]    
				labels += [Y[0]] 
			else:
				labels += [Y]
		data = np.concatenate(data)
		if mode == 'vali':
			dlabel = np.concatenate(detail_labels)
			label = np.concatenate(labels)
			labels = [label, label, label, dlabel, dlabel, dlabel, label, label]
		else:
			labels = [np.concatenate(labels)]
		return data, labels
		

	def generator(self, mode):
		""""
				Generate a generator for prediction
		Args:
		Return:
			generator: function
				that can generate a generator for prediction
			steps: integer
				the number of total batches

				"""	

		#set file list to solve
		if mode == 'vali':
			gen_lst = self.vali_lst
		elif mode == 'test':
			gen_lst = self.test_lst
		elif mode == 'weak':
			gen_lst = self.train_lst

		batch_size = self.batch_size
		feature_dir = self.feature_dir
		label_dir = self.label_dir
		detail_weak_unlabel_dir = self.detail_weak_unlabel_dir
		CLASS = self.CLASS
		LEN = self.LEN
		DIM = self.DIM
		files, f_len = read_lst(gen_lst)

		def generator():
			#the index of the file list
			cur = 0
			
			for i in range(f_len):

				#batch begin
				if i % batch_size == 0:
					train_data = np.zeros([batch_size, LEN * 2, DIM])
					#[label, mask, weight] be consistent with the generator of training, but for prediction, we just use labels[:, :CLASS]
					tclass = CLASS + 1 
					labels = np.ones([batch_size, tclass])
#					labels_sep = np.ones([batch_size, tclass])
#					labels_not_sep = np.ones([batch_size, tclass])
					
				f = files[i]
				data_f = os.path.join(feature_dir, f + '.npy')
				assert os.path.exists(data_f)
				data = np.load(data_f)
				label_f = os.path.join(label_dir, f + '.npy')

				#we can predict for unlabeled data, but can not calculate correct score without label files
				if os.path.exists(label_f):
					label = np.load(label_f)
				else:
					label = np.zeros([CLASS])

				train_data[cur] = data
				labels[cur, :CLASS] = label
				cur += 1

				#batch end
				if cur == (batch_size):
					if mode == 'vali':
						yield train_data, [labels, labels, labels, np.zeros([batch_size, LEN, CLASS]), np.zeros([batch_size, LEN, CLASS]), np.zeros([batch_size, LEN, CLASS]), labels, labels]
					else:
						yield train_data, labels
					cur = 0
			#yield the last batch
			if not f_len % (batch_size) == 0:
				if mode == 'vali':
					yield train_data, [labels, labels, labels, np.zeros([batch_size, LEN, CLASS]), np.zeros([batch_size, LEN, CLASS]), np.zeros([batch_size, LEN, CLASS]), labels, labels]
				else:
					yield train_data, labels

		steps = (f_len + batch_size-1 ) // (batch_size)
		return generator, steps
