import configparser
from keras import backend as K
import keras
from keras import objectives
import tensorflow as tf
from keras.models import load_model, Model
import os
import numpy as np
from src import utils
from src.Logger import LOG
import numpy as np
from src import utils

class metricCallback(keras.callbacks.Callback):
	def __init__(self, conf_dir, train_mode = 'semi'):
		""""
		MetricCallback for training.
		Args:
			conf_dir: string
				the path of configuration dir
			train_mode: string in ['semi','supervised']
				semi-supervised learning or weakly-supervised learning
		Attributes:
			conf_dir
			train_mode
			learning_rate
			decay_rate
			epoch_of_decay
			early_stop
			metric
			ave
			f1_utils
			best_model_path
			batch_size
			CLASS
			best_f1
			best_epoch
			wait
		Interface:
			set_extra_attributes: Set several required attributes.
			init_attributes: Set default values to some attributes.
			check_attributes: Check whether some required attributes have been set.
			init_train_conf: Initialize most of attribute values from the configuration file.
			get_at: Count audio tagging performance (F1).
			get_opt: Optimizer with specified learning rate.
			get_loss: Loss function for semi-supervised learning and weakly-supervised learning.
			on_train_begin
			on_epoch_end
			on_train_end

                """	
		self.train_mode = train_mode
		self.utils = utils
		assert train_mode == 'semi' or train_mode == 'supervised'
		self.conf_dir = conf_dir
		self.init_train_conf()
		self.init_attributes()
		super(metricCallback, self).__init__()

		
	def set_extra_attributes(self, f1_utils, best_model_path, data_loader, batch_size, CLASS):
		""""
		Set several required attributes.
		Args:
			f1_utils: src.utils
				a tool to calculate F-meansure
			best_model_path: string
				the path to save the best performance model 
			batch_size: integer
				the size of a batch
			CLASS: integer
				the number of event categories
		Return:
		"""
		self.f1_utils = f1_utils
		self.best_model_path = best_model_path
		self.data_loader = data_loader
		self.batch_size = batch_size
		self.CLASS = CLASS

	def init_attributes(self):
		""""
		Set default values to some attributes.
		Args:
		Return:
		"""
		self.best_f1 = -1
		self.best_epoch = -1
		self.wait = 0

	def check_attributes(self):
		""""
		Check whether some required attributes have been set.
		If not, assert.
		Args:
		Return:
		"""
		attributes = [self.f1_utils, 
			self.best_model_path, 
			self.batch_size, 
			self.CLASS]

		for attribute in attributes:
			assert attribute is not None

	def init_train_conf(self):
		""""
		Initialize most of attribute values from the configuration file.
		Args:
		Return:

		"""	
		conf_dir = self.conf_dir
		train_cfg_path = os.path.join(conf_dir, 'train.cfg')
		assert os.path.exists(train_cfg_path)
		config = configparser.ConfigParser()
		config.read(train_cfg_path)

		assert 'metricCallback' in config.sections()
		train_conf = config['metricCallback']
		self.learning_rate = float(train_conf['learning_rate'])
		self.decay_rate = float(train_conf['decay_rate'])
		self.epoch_of_decay = int(train_conf['epoch_of_decay'])
		self.early_stop = int(train_conf['early_stop'])
		assert 'validate' in config.sections()
		vali_conf = config['validate']
		self.metric = vali_conf['metric']
		self.ave = vali_conf['ave']

	def get_at(self, preds, labels):
		""""
		Count audio tagging performance (F1).
		Args:
			preds: numpy.array
				shape: [number_of_files_( + padding), CLASS]
					prediction of the model
			labels: numpy.array
				shape: [number_of_files_( + padding), CLASS]
					labels loaded from files
		Return:	
			f1: float
			the audio tagging performance (F1)
		"""
		f1_utils = self.f1_utils
		f1, _, _, _, _, _ = f1_utils.get_f1(preds, labels, mode = 'at')
		return f1

	def get_opt(self, lr):
		""""
		Optimizer with specified learning rate.
		Args:
			lr: float
				learning rate
		Return:
			opt: keras.optimizers
				Adam optimizer
		"""
		opt = keras.optimizers.Adam(lr = lr, beta_1 = 0.9, 
			beta_2 = 0.999, epsilon = 1e-8, decay = 1e-8)
		return opt

	def get_distance_encoder_loss(self):
		def loss_distance_encoder(y_true, y_pred):
			return 0.01 * K.mean(y_pred, axis = -1)
		return loss_distance_encoder

	def get_no_loss(self):
		def loss(y_true, y_pred):
			return 0.0 * K.mean(K.mean(y_pred, axis = -1), axis = -1)
		return loss

	def get_loss(self, weak, sep_str, epoch):
		""""
		Loss function for semi-supervised learning and weakly-supervised learning.
		Args:
		Return:
			loss (if train_mode is 'supervised'): function
				loss function for weakly-supervised learning
			semi_loss (if train_mode is 'semi'): function
				loss function for semi-supervised learning
		"""

		CLASS = self.CLASS
		train_mode = self.train_mode
		LEN = 500

		def add_win(probs, win_len):
			half_win_len = win_len // 2
			pad_probs = K.concatenate([probs[:, :1]] * half_win_len + [probs] + [probs[:, LEN -1:]] * (half_win_len + 1), axis = -1)
			add_probs = []
			for i in range(win_len):
				add_probs += [K.expand_dims(pad_probs[:, i : i + LEN], axis = -1)]
			add_prob = K.concatenate(add_probs, axis = -1)
			add_prob = tf.nn.top_k(add_prob, win_len // 2 + 1).values[:, :, -1]	
			return add_prob
			#print(add_prob.shape)

		def smooth(probs, dt):#
			win_lens = [25, 88, 13, 8, 17, 123, 155, 93, 25, 143]
#			win_lens = [50] * CLASS
			smooth_probs = []
			for i in range(CLASS):
				prob = probs[:, :, i]
				prob = pred(prob, dt)
				prob = add_win(prob, win_lens[i])
				prob = K.expand_dims(prob, axis = -1)
				smooth_probs += [prob]
			smooth_prob = K.concatenate(smooth_probs, axis = -1)
			return smooth_prob


		def pred(X, dt):
			return tf.where(tf.greater(X, dt), tf.ones_like(X), tf.zeros_like(X))

		def calcu_gr(A, B):
			A = K.expand_dims(A, axis = 1)
			B = K.expand_dims(B, axis = 0)
			C = K.sum(K.abs(A - B), axis = -1)
			D = K.sum(A + B, axis = -1)
			E = D - C
			C = 1 - tf.where(tf.greater(C, 1), tf.ones_like(C), C)
			E = tf.where(tf.greater(E, 1), tf.ones_like(E), E) - 1
			F = C
			return - F

		"""
		def get_mask(gr1, gr2, w_gr1, w_gr2):			
			print ("gr1", gr1)
			print ("gr2", gr2)
			print ("w_gr1", w_gr1)
			print ("w_gr2", w_gr2)
			ep_gr1 = K.expand_dims(gr1, axis = 1)
			ep_gr2 = K.expand_dims(gr2, axis = 0)
			ep_w_gr1 = K.expand_dims(w_gr1, axis = 1)
			ep_w_gr2 = K.expand_dims(w_gr2, axis = 0)	
					
			is_same_w_gr_tmp = K.sum(K.abs(ep_w_gr1 - ep_w_gr2), axis = -1)	
			is_same_gr_tmp = K.sum(K.abs(ep_gr1 - ep_gr2), axis = -1)
			
			is_diff_w_gr_tmp = K.max(K.abs(ep_w_gr1 + ep_w_gr2), axis = -1)
			is_diff_gr_tmp = K.max(K.abs(ep_gr1 + ep_gr2), axis = -1)

			is_same_w_gr = 1 - tf.where(tf.greater(is_same_w_gr_tmp, 0), tf.ones_like(is_same_w_gr_tmp), tf.zeros_like(is_same_w_gr_tmp))
			is_same_gr = 1 - tf.where(tf.greater(is_same_gr_tmp, 0), tf.ones_like(is_same_gr_tmp), tf.zeros_like(is_same_gr_tmp))

			is_diff_w_gr = tf.where(tf.greater(is_diff_w_gr_tmp, 1.5), tf.ones_like(is_diff_w_gr_tmp), tf.zeros_like(is_diff_w_gr_tmp))
			is_diff_gr = tf.where(tf.greater(is_diff_gr_tmp, 1.5), tf.ones_like(is_diff_gr_tmp), tf.zeros_like(is_diff_gr_tmp))			
			
			is_not_both_slience_gr = 1 - tf.where(tf.equal(is_diff_gr_tmp, 0), tf.ones_like(is_diff_gr_tmp), tf.zeros_like(is_diff_gr_tmp))

			pos_mask = is_same_w_gr * is_same_gr
			neg_mask = is_diff_w_gr * is_diff_gr * is_not_both_slience_gr
			
			return pos_mask, neg_mask


		"""
		def get_mask(A, B, C, D):					 #get_mask(gr1, gr2, wgr1, wgr2)
			C = K.expand_dims(C, axis = 1)
			D = K.expand_dims(D, axis = 0)
			A = K.expand_dims(A, axis = 1)
			B = K.expand_dims(B, axis = 0)			

			E = K.sum(K.abs(C - D), axis = -1)          # weak
			F = K.sum(K.abs(C + D), axis = -1)          # weak
			G = K.sum(K.abs(A - B), axis = -1)          # strong
			H = K.sum(K.abs(A + B), axis = -1)          # strong

			I = E                                       # weak pos
			J = K.abs(E - F)                            # weak neg
			Ki = G
			L = K.abs(G - H)
			M = K.abs(G + H)

			I = 1 - tf.where(tf.greater(I, 0), tf.ones_like(I), I)
			J = 1 - tf.where(tf.greater(J, 0), tf.ones_like(J), J)
			Ki = 1 - tf.where(tf.greater(Ki, 0), tf.ones_like(Ki), Ki)
			L = 1 - tf.where(tf.greater(L, 0), tf.ones_like(L), L)
			M = 1 - tf.where(tf.greater(M, 0), tf.ones_like(M), M)

			pos_mask = Ki * I                     # * I
			neg_mask = L * (1 - M) * J             # * J

			return pos_mask, neg_mask
	
	

		def loss(y_true, y_pred):
			""""
			Loss function for weakly-supervised learning.

		         """

			chunksize = 4
			num_syn = 4
			a = 1 / 2 / num_syn
			b = 1 / num_syn
			N = 1
			M = num_syn + 1
			labeled_batch = int(M * chunksize)

			if weak:
				y_is_unlabel = y_true[:,CLASS:CLASS+1]
				y_true = y_true[:,:CLASS]
				if(sep_str == '_' or sep_str == 'sep' or sep_str == 'not_sep'):
					y_pred_PT = y_pred[:,:CLASS]
					y_pred_PS = y_pred[:,CLASS:2*CLASS]
					y_pesudo_PT = pred(y_pred_PT, 0.5)
					y_pesudo_PS = pred(y_pred_PS, 0.5)	
					losses = []
					for i in range(2):
						st = i * CLASS
						ed = (i + 1) * CLASS
						closs = K.mean(K.binary_crossentropy(y_true[:labeled_batch][0::M], y_pred[:labeled_batch][0::M, st:ed]), axis = -1)
						uloss = []
						for j in range(1, M):
							uloss += [a * K.mean(K.binary_crossentropy(y_true[:labeled_batch][j::M], y_pred[:labeled_batch][j::M, st:ed]), axis = -1)]
						
						losses += [K.concatenate([closs] + uloss + [tf.zeros(chunksize)], axis = 0)]
					culoss = losses[0]

					for i in range(1, 2):
						culoss += losses[i]
					guided_loss_PT =  (epoch + 1) / 100 *K.mean(K.binary_crossentropy(y_pesudo_PT * y_is_unlabel, y_pred_PS * y_is_unlabel), axis = -1)
					guided_loss_PS =  (epoch + 1) / 100 * K.mean(K.binary_crossentropy(y_pesudo_PS * y_is_unlabel, y_pred_PT * y_is_unlabel), axis = -1)
					guided_loss = guided_loss_PT + guided_loss_PS
					return culoss + guided_loss
				else:
					raise AssertionError("You should indicate the branch type")


			else:
				y_true = y_true[:labeled_batch]
				y_pred = y_pred[:labeled_batch]
				disilarity_1 = y_pred[:, CLASS * 2 : labeled_batch + CLASS * 2]

				y_pred = y_pred[:, :CLASS * 2]
				y_pred = K.permute_dimensions(y_pred, (0, 2, 1))

				syn_w = np.load('mask_weights/syn-1234_weights.npy')
				weak_w = np.load('mask_weights/weak_weights.npy')
				w = syn_w

				uloss = []

				mask = y_true[0::M, 0]
				mask_2 = tf.where(tf.greater(mask, 0), tf.ones_like(mask), tf.ones_like(mask) * w)
				pred_X = y_pred[0::M, :, CLASS : 2 * CLASS]
				mask_w = y_true[0::M, 1, 0]
				smooth_gr = smooth(pred_X * y_true[0::M, 0 : 1], 0.5)
				closs = tf.zeros_like(mask_w)
				sloss = tf.zeros_like(mask_w)
				masks =[]
				grs = []

				eyes = K.expand_dims(K.eye(chunksize), axis = -1)

				for j in range(1, M):
					mask = K.sum(y_true[j::M], axis = 1)
					masks += [tf.where(tf.greater(mask, 0), tf.ones_like(mask), tf.ones_like(mask) * w)]
					gr = tf.where(tf.greater(mask, 0), tf.ones_like(mask), tf.zeros_like(mask))
					grs += [K.expand_dims(gr, axis = 1)]					

				dis_losses = []
				connect_loss = []

				base = 0.5
				margin = base + 0.1
				pos_margin = base - 0.1


				for j in range(1, M):
					mask_1 = masks[j - 1]
					uloss += [ b * K.mean(mask_1 * K.mean(K.binary_crossentropy(y_true[j::M], y_pred[j::M, :, : CLASS]), 
																									axis = 1), axis = -1)]

					pos_mask, neg_mask = get_mask(y_true[j::M], smooth_gr, grs[j - 1], y_true[0::M, 0 : 1])
					s = disilarity_1[j::M, 0::M]
					pos_tmp = tf.where(tf.less(s - pos_margin, 0.0), tf.zeros_like(s - pos_margin), s - pos_margin)
					unit_loss_pos = tf.where(tf.equal(pos_mask, 1), pos_tmp, tf.zeros_like(pos_tmp))
					neg_tmp = tf.where(tf.less(margin - s, 0.0), tf.zeros_like(s), margin - s)
					unit_loss_neg = tf.where(tf.equal(neg_mask, 1.0), neg_tmp, tf.zeros_like(s))
					unit_loss = unit_loss_pos + unit_loss_neg
#					unit_loss = s * (neg_mask - pos_mask) + margin
#					unit_loss = tf.where(tf.less(unit_loss, 0), tf.zeros_like(unit_loss), unit_loss)
#					unit_loss = tf.where(tf.less(unit_loss, 0), tf.zeros_like(unit_loss), unit_loss)
					dis_losses += [K.mean(K.mean(unit_loss, axis = -1), axis = -1)]

				for i in range(2):
					if i == 0:
						dis = disilarity_1
						p = 1
						gr1 = smooth_gr
						gr2 = smooth_gr
						wgr1 = y_true[0::M, 0 : 1]
						wgr2 = y_true[0::M, 0 : 1]
					else:
						dis = disilarity_1
						p = 0.5
						gr1 = y_true[i::M]
						gr2 = y_true[i::M]
						wgr1 = grs[i - 1]
						wgr2 = grs[i - 1]


					pos_mask, neg_mask = get_mask(gr1, gr2, wgr1, wgr2)

					s = dis[i::M, i::M]

					pos_tmp = tf.where(tf.less(s - pos_margin, 0), tf.zeros_like(s - pos_margin), s - pos_margin)
					unit_loss_pos = tf.where(tf.equal(pos_mask, 1), pos_tmp, tf.zeros_like(pos_tmp))
					neg_tmp = tf.where(tf.less(-s + margin, 0), tf.zeros_like(s), -s + margin)
					unit_loss_neg = tf.where(tf.equal(neg_mask, 1), neg_tmp, tf.zeros_like(s))
					unit_loss = unit_loss_pos + unit_loss_neg
			

					closs = closs + p * K.mean(K.mean(unit_loss, axis = -1))


				dis_loss = K.concatenate([closs] + dis_losses, axis = 0)
				culoss_1 = 1.0 * K.concatenate([sloss] + uloss, axis = 0) + 1.0 * dis_loss #detection branch loss + domain adaption loss
				culoss_1 = K.concatenate([culoss_1, tf.zeros(chunksize)], axis = 0)
#				culoss_1 = K.concatenate([sloss], axis = 0) + dis_loss
				return culoss_1 


		if train_mode == 'supervised' or train_mode == 'semi':
			return loss
		assert True


	
	def compile_again(self, epoch):
		#atten_dens = self.atten_dens
		#for layer in atten_dens:
		#	self.model.get_layer(layer).trainable = trainable
		opt = self.get_opt(self.learning_rate)
		loss = [self.get_loss(True, '_', epoch), self.get_loss(True, 'sep', epoch), self.get_loss(True, 'not_sep', epoch), self.get_loss(False, '_', epoch), self.get_loss(False, 'sep', epoch), self.get_loss(False, 'not_sep', epoch),self.get_distance_encoder_loss(), self.get_no_loss()]
		self.model.compile(optimizer = opt, loss = loss)

	def change_GRL_config(self, a):
		for i, layer in enumerate(self.model.layers):
			if layer.name == 'GRL':
				index = i
				break
		config = layer.get_config()
		config.update({'grl': a})
		layer = layer.from_config(config)
		self.model.layers[index] = layer

	def on_train_begin(self, logs = {}):
		""""
		(overwrite)
		The beginning of training.

		"""
		#check extra required attributes
		self.check_attributes()
		LOG.info('init training...')
		LOG.info('metrics : %s %s'%(self.metric, self.ave))

		self.compile_again(epoch = 0)
		#print(self.model.get_layer('GRL').get_config())
		#compile the model with specific loss function

	#def on_batch_end(self, epoch, logs = {}):
	#	self.reweight()	
	def on_epoch_end(self, epoch, logs = {}):
		""""
		(overwrite)
		The end of a training epoch.

		"""
		best_f1 = self.best_f1
		f1_utils = self.f1_utils
		CLASS = self.CLASS
		train_mode = self.train_mode
		early_stop = self.early_stop

		#self.reweight()

		#get the features of the validation data
		vali_data = self.validation_data
		#get the labels of the validation data
		labels = vali_data[1][:, :CLASS]

		#get audio tagging predictions of the model
		preds = self.model.predict(vali_data[0], batch_size = self.batch_size)
		frame_preds = preds[-1]
		preds = preds[0]
		
		if train_mode == 'semi':
			#get the predictions of the PT-model
			preds_PT = preds[:, :CLASS]
			#get the predictions of the PS-model
			preds_PS = preds[:, CLASS:]
			#count F1 score on the validation set for the PT-model
			pt_f1 = self.get_at(preds_PT, labels)
			#count F1 score on the validation set for the PS-model
			ps_f1 = self.get_at(preds_PS, labels)
			# load the file list and the groundtruths
			# psds_utils = f1_utils
			data_loader = self.data_loader
			mode = 'vali'
			lst, csv, dur_csv = data_loader.get_vali()
	
			# prepare the file list and the groundtruths for counting scores
			# psds_utils.set_vali_csv(lst, csv, dur_csv)

			# # get psds
			# psds_1, psds_2 = psds_utils.get_psds(preds, frame_preds, mode=mode)
			# print("psds_1", psds_1)
			# print("psds_2", psds_2)

		else:
			#count F1 score on the validation set for the PS-model
			ps_f1 = self.get_at(preds[:, :CLASS], labels)

		#the final performance depends on the PS-model
		logs['f1_val'] = ps_f1

		is_best = 'not_best'
		#self.model.save_weights(self.best_model_path + '_' + str(epoch))
		#preserve the best model during training
		if logs['f1_val'] >= self.best_f1:
			self.best_f1 = logs['f1_val']
			self.best_epoch = epoch
			self.model.save_weights(self.best_model_path)
			is_best = 'best'
			self.wait = 0

		#the PS-model has not been improved after [wait] epochs
		self.wait += 1


		if train_mode == 'semi':
			LOG.info('[ epoch %d , sed f1 : %f , at f1 : %f ] %s'
				%(epoch, logs['f1_val'], pt_f1, is_best))
		else:
			LOG.info('[ epoch %d, f1 : %f ] %s'
				%(epoch, logs['f1_val'], is_best))

		if self.wait > early_stop:
			self.stopped_epoch = epoch
			self.model.stop_training = True
			return

		#learning rate decays every epoch_of_decay epochs
		if epoch > 0 and epoch%self.epoch_of_decay == 0:
			self.learning_rate *= self.decay_rate
			LOG.info('[ epoch %d , learning rate decay to %f ]'%(
					epoch, self.learning_rate))
			#recompile the model with decreased learning rate
		#self.model.compile(optimizer = opt, loss = loss)
		#a = 2 / (1 + np.exp(-(epoch + 1) * 0.5)) - 1
		#print('-----%d---grl----%f--------' %(epoch, a))
		#self.change_GRL_config(a)	
			self.compile_again(epoch)
		
	def on_train_end(self, logs = {}):
		""""
		(overwrite)
		The end of training.

		"""
		best_epoch = self.best_epoch
		best_f1 = self.best_f1
		#report the best performance of the PS-model
		LOG.info('[ best vali f1 : %f at epoch %d ]'%(best_f1, best_epoch))

	
