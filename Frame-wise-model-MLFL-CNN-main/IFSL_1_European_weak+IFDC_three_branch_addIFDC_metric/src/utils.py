from numpy.core.fromnumeric import size
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import dcase_util
import os
import numpy as np
import configparser
import scipy
from src.evaluation import sound_event_eval
from src.evaluation import scene_eval
import src.evaluation.TaskAEvaluate as taskAEvaluate
import copy
from scipy.ndimage.filters import median_filter, uniform_filter1d
from src.evaluation.evaluation_measures import compute_psds_from_operating_points, compute_psds_from_operating_points_tune
import pandas as pd
from sed_scores_eval.base_modules.io import write_sed_scores, write_score_transform, read_score_transform
from sed_scores_eval.utils.scores import create_score_dataframe
import pdb

#duration of a single audio file (second)
DURATION=10.0

class utils(object):
	def __init__(self, conf_dir, exp_dir, label_lst,  win_conf):
		""""
		Tools to calculate performace.
		Args:
			conf_dir: string
				the path of configuration dir
			exp_dir: string
				the path of experimental dir
			label_lst: list
				the event list
		Attributes:
			conf_dir
			label_lst
			evaluation_path
			evaluation_ests
			evaluation_refs
			preds_path
			CLASS
			win_lens
			metric
			ave
			
		Interface:	
		
		"""
		self.conf_dir = conf_dir
		self.label_lst = label_lst
		self.win_conf = win_conf
		self.exp_dir = exp_dir
		self.init_utils_conf()

		self.evaluation_path=os.path.join(exp_dir,'evaluation')	
		self.init_dirs(self.evaluation_path)
		self.evaluation_ests=os.path.join(self.evaluation_path,'ests')
		self.init_dirs(self.evaluation_ests)
		self.evaluation_refs=os.path.join(self.evaluation_path,'refs')
		self.init_dirs(self.evaluation_refs)
		self.preds_path=os.path.join(self.evaluation_path,'preds.csv')
		self.evaluation_psds = os.path.join(self.evaluation_path, 'psds')
		self.init_dirs(self.evaluation_psds)
		self.evaluation_psds_test=os.path.join(self.evaluation_psds,'test')
		self.init_dirs(self.evaluation_psds_test)
		self.evaluation_psds_vali = os.path.join(self.evaluation_psds, 'vali')
		self.init_dirs(self.evaluation_psds_vali)
		self.score_test = os.path.join(self.evaluation_psds, 'score_test')
		self.init_dirs(self.score_test)

	def init_dirs(self,path):
		""""
		Create new dir.
		Args:
			path: string
				the path of the dir to create
		Return:

		"""
		if not os.path.exists(path):
			os.mkdir(path)

	def set_win_lens(self,win_lens):
		""""
		Set adaptive sizes of median windows.
		Args: list
			adaptive sizes of median windows
		Return

		"""
		if not len(self.win_lens) == self.CLASS:
			self.win_lens = win_lens
		print("win len:", self.win_lens)

	def tune_win_lens(self, win_len):
		""""
		Set adaptive sizes of median windows.
		Args: list
			adaptive sizes of median windows
		Return

		"""
		self.win_lens=[win_len] * self.CLASS

	def init_utils_conf(self):
		"""""
		Initialize most of attribute values from the configuration file.
		Args:
		Return:

		"""
		CLASS = len(self.label_lst)
		self.CLASS = CLASS
		conf_dir=self.conf_dir
		utils_cfg_path=os.path.join(conf_dir,'train.cfg')

		assert os.path.exists(utils_cfg_path)
		config = configparser.ConfigParser()
		config.read(utils_cfg_path)
		conf = config['validate']
		win_len = conf['win_len']
		win_len_max = int(conf['win_len_max'])
		improve_psds = conf['improve_psds']

		try:
			self.psds_1_win_dir = os.path.join(self.exp_dir, 'result', 'psds_1_win.csv')
			self.psds_2_win_dir = os.path.join(self.exp_dir, 'result', 'psds_2_win.csv')
			psds_1_win = pd.read_csv(self.psds_1_win_dir, index_col=0)
			psds_2_win = pd.read_csv(self.psds_2_win_dir, index_col=0)
			assert psds_1_win.shape == psds_2_win.shape
			self.win_lens = list()
			psds_1 = np.array(psds_1_win.iloc[:, :-1])  # psds_1 is max
			psds_2 = np.array(psds_2_win.iloc[:, :-1])  # psds_2 is max
			psds_1_2 = psds_1 + psds_2  # psds_1 and psds_2's mean is max
		except:
			print("Tune-Window is not available.")
		try:
			if 'nst_' in self.win_conf :
				try:
					self.psds_1_win_dir = os.path.join(self.exp_dir, 'result', 'psds_1_win_cnn_nst.csv')
					self.psds_2_win_dir = os.path.join(self.exp_dir, 'result', 'psds_2_win_cnn_.csv')
					psds_1_win = pd.read_csv(self.psds_1_win_dir, index_col=0)
					psds_2_win = pd.read_csv(self.psds_2_win_dir, index_col=0)
					assert psds_1_win.shape == psds_2_win.shape
					self.win_lens = list()
					psds_1 = np.array(psds_1_win.iloc[:, :-1])  # psds_1 is max
					psds_2 = np.array(psds_2_win.iloc[:, :-1])  # psds_2 is max
					psds_1_2 = psds_1 + psds_2  # psds_1 and psds_2's mean is max
				except:
					print("Tune-Window is not available.")
				if '_12' in self.win_conf:
					print('improve psds12')
					psds_num = psds_1_2  # choose from psds_1 psds_2 psds_1_2
				elif '_1' in self.win_conf  :
					print('improve psds1')
					psds_num = psds_1 # choose from psds_1 psds_2 psds_1_2
				elif '_2' in self.win_conf:
					print('improve psds2')
					psds_num = psds_2  # choose from psds_1 psds_2 psds_1_2
				psds_num = psds_num[: win_len_max]
				for i in range(CLASS):
					self.win_lens.append(np.where(psds_num[:, i] == np.max(psds_num[:, i]))[-1].item() + 1)
				self.win_lens = np.array(self.win_lens)
			elif 'st_' in self.win_conf :
				try:
					self.psds_1_win_dir = os.path.join(self.exp_dir, 'result', 'psds_1_win_cnn.csv')
					self.psds_2_win_dir = os.path.join(self.exp_dir, 'result', 'psds_2_win_cnn.csv')
					psds_1_win = pd.read_csv(self.psds_1_win_dir, index_col=0)
					psds_2_win = pd.read_csv(self.psds_2_win_dir, index_col=0)
					assert psds_1_win.shape == psds_2_win.shape
					self.win_lens = list()
					psds_1 = np.array(psds_1_win.iloc[:, :-1])  # psds_1 is max
					psds_2 = np.array(psds_2_win.iloc[:, :-1])  # psds_2 is max
					psds_1_2 = psds_1 + psds_2  # psds_1 and psds_2's mean is max
				except:
					print("Tune-Window is not available.")
				if '_12' in self.win_conf:
					print('improve psds12')
					psds_num = psds_1_2  # choose from psds_1 psds_2 psds_1_2
				elif '_1' in self.win_conf  :
					print('improve psds1')
					psds_num = psds_1 # choose from psds_1 psds_2 psds_1_2
				elif '_2' in self.win_conf:
					print('improve psds2')
					psds_num = psds_2  # choose from psds_1 psds_2 psds_1_2
				psds_num = psds_num[: win_len_max]
				for i in range(CLASS):
					self.win_lens.append(np.where(psds_num[:, i] == np.max(psds_num[:, i]))[-1].item() + 1)
				self.win_lens = np.array(self.win_lens)
		except:
			if self.win_conf=='auto':
				self.win_lens = []
			elif type(self.win_conf) == int:
				self.win_lens = np.array([self.win_conf] * CLASS)
			elif type(self.win_conf) == str:
				if self.win_conf == '1':
					print('improve psds1')
					psds_num = psds_1 # choose from psds_1 psds_2 psds_1_2
				elif self.win_conf == '2':
					print('improve psds2')
					psds_num = psds_2  # choose from psds_1 psds_2 psds_1_2
				elif self.win_conf == '12':
					print('improve psds12')
					psds_num = psds_1_2  # choose from psds_1 psds_2 psds_1_2
				psds_num = psds_num[: win_len_max]
				for i in range(CLASS):
					self.win_lens.append(np.where(psds_num[:, i] == np.max(psds_num[:, i]))[-1].item() + 1)
				self.win_lens = np.array(self.win_lens)
			elif win_len=='auto':
				self.win_lens = []
			elif win_len=='tune':
				if improve_psds == '1':
					print('improve psds1')
					psds_num = psds_1 # choose from psds_1 psds_2 psds_1_2
				elif improve_psds == '2':
					print('improve psds2')
					psds_num = psds_2  # choose from psds_1 psds_2 psds_1_2
				elif improve_psds == '12':
					print('improve psds12')
					psds_num = psds_1_2  # choose from psds_1 psds_2 psds_1_2
				psds_num = psds_num[: win_len_max]
				for i in range(CLASS):
					self.win_lens.append(np.where(psds_num[:, i] == np.max(psds_num[:, i]))[-1].item() + 1)
				self.win_lens = np.array(self.win_lens)
			else:
				self.win_lens = np.array([int(conf['win_len'])] * CLASS)

		self.metric=conf['metric']
		self.ave=conf['ave']
		self.score_transform = conf['score_transform']

	def get_vali_lst(self):
		""""
		Get current file list and groundtruths using for calculating 
		performance.
		Args:
		Return:
			lst: list
				the path list of files
			csv: list
				the groundtruth list of files

		"""
		lst=self.lst
		csv=self.csv
		dur_csv=self.dur_csv
		return lst, csv, dur_csv

	def set_vali_csv(self,lst,csv,dur_csv):
		""""
		Set current file list and groundtruths for calculating performance.
		Args:
			lst: list
				the path list of files
			csv: list
				the groundtruth list of files
		Return:

		"""
		self.lst=lst
		self.csv=csv
		self.dur_csv=dur_csv
			
	def init_csv(self,csvs,flag=True):
		""""
		Format groundtruths from a csv file to several single files.
		All the delimiters should be '\t'.
		Eg.
		original file:
			file_ori:
				A.wav	0.00	1.00	Cat
				A.wav	1.00	2.00	Dog	
				B.wav	0.00	1.00	Dog
		target files:
			file1: A.txt
				0.00    1.00    Cat
				1.00    2.00    Dog
			file2: B.txt
				0.00    1.00    Dog
		Args:
			csvs: list
				the groundtruth list to format
			flag: bool
				If flag is true, save result files into 
				evaluation_refs dir.
				Otherwise, save result files into
				evaluation_ests dir.
			
		"""
		if flag:
			root=self.evaluation_refs
		else:
			root=self.evaluation_ests
		#get formatted results
		result=self.format_lst(csvs)
		#save formatted results
		self.format_csv(result,root)



	def format_lst(self,csvs):
		""""
		Format the groundtruths.
		Eg.
		ori list:
			['A.wav   0.00    1.00    Cat',
			 'A.wav   1.00    2.00    Dog',
			 'B.wav   0.00    1.00    Dog']
		obj dict:
			{'A':[['0.00','1.00','Cat'],
			      ['1.00','2.00','Dog']],
			 'B':[['0.00','1.00','Dog']]}
		Args:
			csv: list
				the groundtruth list to format
		Return:
			result: dict
				formatted results
				
		
		"""
		tests=[t.rstrip().split('\t') for t in csvs]
		result={}
		cur=0
		for i,t in enumerate(tests):
			f=str.replace(t[0],'.wav','')
			if f not in result:
				result[f]=[]
			if len(t)>1:
				result[f]+=[[t[1],t[2],t[3]]]
			else:
				cur+=1
		return result
	
	def format_csv(self,tests,root):
		""""
		Save formatted results to several files.
		Eg.
		ori dict:
			{'A':[['0.00','1.00','Cat'],
                              ['1.00','2.00','Dog']],
			 'B':[['0.00','1.00','Dog']]}
		obj files:
			file1: A.txt
				0.00    1.00    Cat
				1.00    2.00    Dog
			file2: B.txt
				0.00    1.00    Dog
		Args:
			tests: list
				formatted results
			root: string
				the path of dir to save files
		Return:
		
		"""
		for t in tests:
			#get the path of a single file
			fname=os.path.join(root,t)
			with open(fname+'.txt','w') as f:
				result=[]
				for k in tests[t]:
					if len(k)>1:
						result+=['%s\t%s\t%s'%(k[0],k[1],k[2])]
				result='\n'.join(result)
				f.writelines(result)

	def get_f1(self,preds,labels,mode='at'):
		""""
		Calculate perfomance.
		Args:
			preds: numpy.array
				clip-level predicton (posibilities)
			labels: numpy.array
				weakly labels (or frame-level predicton)
			mode: string in ['at','sed']
				get audio tagging perfomance in mode 'at' and 
				get sound event perfomance in mode 'sed'
		Return:
			if mode=='at':
				F1: numpy.array
					clip-level F1 score
				precision: numpy.array
					clip-level precision
				recall: numpy.array
					clip-level recall
			if mode=='sed':
				segment_based_metrics: sed_eval.sound_event.SegmentBasedMetrics
					segment based result
				event_based_metrics: sed_eval.sound_event.event_based_metrics
					event based result

		"""
		#get current file list
		lst,_,_=self.get_vali_lst()
		preds=preds[:len(lst)]
		#using threshold of 0.5 to get clip-level decision
		preds[preds>=0.5]=1
		preds[preds<0.5]=0

		evaluation_path=self.evaluation_path
		
		#get audio tagging performance
		if mode=='at':
			labels=labels[:len(lst)]	
			ave=self.ave
			CLASS=labels.shape[-1]
			TP=(labels+preds==2).sum(axis=0)
			FP=(labels-preds==-1).sum(axis=0)
			FN=(labels-preds==1).sum(axis=0)
			if ave=='class_wise_F1':
				TFP=TP+FP
				TFP[TFP==0]=1
				precision=TP/TFP
				TFN=TP+FN
				TFN[TFN==0]=1
				recall=TP/TFN
				pr=precision + recall
				pr[pr==0]=1
			elif ave=='overall_F1':
				TP=np.sum(TP)
				FP=np.sum(FP)
				FN=np.sum(FN)

				TFP=TP+FP
				if TFP==0:
					TFP=1
				precision=TP/TFP
				TFN=TP+FN
				if TFN==0:
					TFN=1
				recall=TP/TFN
				pr=precision + recall
				if pr==0:
					pr=1

			F1=2*precision*recall/pr

			if ave=='class_wise_F1':
				class_wise_f1=F1
				class_wise_pre=precision
				class_wise_recall=recall
				F1=np.mean(F1)
				precision=np.mean(precision)
				recall=np.mean(recall)

			return F1,precision,recall,class_wise_f1,class_wise_pre,class_wise_recall

		#get event detection performance
		elif mode=='sed':
			segment_based_metrics,event_based_metrics=self.get_sed_result(preds,labels)
			return segment_based_metrics,event_based_metrics
		assert False



	def get_predict_csv(self,results, path=None):
		""""
		Format all the results into a file.
		Eg.
                ori dict:
			{'A':[['0.00','1.00','Cat'],
                              ['1.00','2.00','Dog']],
			 'B':[['0.00','1.00','Dog']],
			 'C':[]}
		obj file content:
			A.wav   0.00    1.00    Cat
			A.wav   1.00    2.00    Dog
			B.wav   0.00    1.00    Dog
			C.wav			
		
		Args:
			results: dict
				original dict to format
		Return:
			outs: list
				content of the file to save
				
		"""
		outs=[]
		for re in results:
			flag=True
			for line in results[re]:
				outs+=['%s.wav\t%s\t%s\t%s'%(
					re,line[0],line[1],line[2])]
				flag=False
			if flag:
				outs+=['%s.wav\t\t\t'%re]
		if path is None:
			with open(self.preds_path,'w') as f:
				f.writelines('\n'.join(outs))
		else:
			with open(path,'w') as f:
				f.writelines('\n'.join(outs))
		return outs
	
	def get_sed_result(self, preds, frame_preds):
		""""
		Calculate event detection performance.
		Args:
			preds: numpy.array
				clip-level decision
			frame_preds: numpy.array
				frame-level prediction
		Return:
			segment_based_metrics: sed_eval.sound_event.SegmentBasedMetrics
				segment based result
			event_based_metrics: sed_eval.sound_event.EventBasedMetrics
				event based result
			
		"""

		#get current file list
		lst,csv,_=self.get_vali_lst()
		label_lst=self.label_lst
		win_lens=self.win_lens
		CLASS=self.CLASS
		#get the number of frames of the frame-level predicion
		top_LEN=frame_preds.shape[1]
		#duration (second) per frame
		hop_len=DURATION/top_LEN

		frame_preds=frame_preds[:len(lst)]

		decision_encoder=dcase_util.data.DecisionEncoder(
			label_list=label_lst)

		shows=[]
		result={}
		file_lst=[]

		for i in range(len(lst)):
			pred=preds[i]
			frame_pred=frame_preds[i]
			for j in range(CLASS):
				#If there is not any event for class j
				if pred[j]==0:
					frame_pred[:,j]*=0
				else:
					#using median_filter on prediction for the first post-processing
					frame_pred[:,j]=median_filter(
                                            frame_pred[:,j],(win_lens[j]))	
			#making frame-level decision
			frame_decisions=dcase_util.data.ProbabilityEncoder()\
				.binarization(
					probabilities=frame_pred,
					binarization_type='global_threshold',
					time_axis=0)

			# using median_filter on decision for the second post-processing
			for j in range(CLASS):
				frame_decisions[:,j]=median_filter(
					frame_decisions[:,j], (win_lens[j]))
			
			#generate reference-estimated pairs
			if lst[i] not in result:
				result[lst[i]]=[]
				file_lst+=[{'reference_file':'refs/%s.txt'%lst[i],
					'estimated_file':'ests/%s.txt'%lst[i]}]

			#encode discrete decisions to continuous decisions 
			for j in range(CLASS):
				estimated_events=decision_encoder\
						.find_contiguous_regions(
					activity_array=frame_decisions[:,j])
				
				for [onset, offset] in estimated_events:
					result[lst[i]]+=[[str(onset*hop_len),
							str(offset*hop_len),
							label_lst[j]]]
		#save continuous decisions to a file
		self.get_predict_csv(result)
		#save continuous decisions to multiple files for evaluation
		self.format_csv(result,self.evaluation_ests)

		#get performance using dcase_util
		segment_based_metrics,event_based_metrics=sound_event_eval.main(
			self.evaluation_path,file_lst)

		return segment_based_metrics,event_based_metrics
	
	def get_psds(self, preds, frame_preds, test_n_thresholds = 50, mode='test'):
		""""
		Calculate event detection performance.
		Args:
			preds: numpy.array
				clip-level predicton (posibilities)
			frame_preds: numpy.array
				frame-level prediction
			test_n_thresholds: int
				number of frame-level probability thresholds
		Return:

		"""
		# get current file list
		lst, csv, dur_csv=self.get_vali_lst()
		preds=preds[:len(lst)]

		label_lst = self.label_lst
		win_lens = self.win_lens
		CLASS = self.CLASS
		# get the number of frames of the frame-level predicion
		top_LEN = frame_preds.shape[1]
		# duration (second) per frame
		hop_len = DURATION / top_LEN
		timestamps = np.linspace(0, DURATION, top_LEN + 1)

		frame_preds = frame_preds[:len(lst)]

		decision_encoder = dcase_util.data.DecisionEncoder(
			label_list=label_lst)

		# shows = []
		# file_lst = []
		test_thresholds = np.arange(
			1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
		)
		pred_dfs = {}
		score_test = os.path.join(self.score_test, str(np.mean(win_lens)))
		if os.path.exists(score_test):
			os.system("rm -rf %s" %(score_test))
		os.mkdir(score_test)
		gt_df = pd.DataFrame([s.split() for s in csv],columns=["filename","onset", "offset","event_label"])
		gt_df['onset'] = gt_df['onset'].astype(float)
		gt_df['offset'] = gt_df['offset'].astype(float)
		# gt_df = pd.read_csv("/data/dcase2021/preprocessed_dataset/real/metadata/validation/validation.tsv", sep='\t')
		gt_df.columns = ["filename", "onset", "offset", "event_label"]

		if self.score_transform == 'True':
			print('Score Transform')
			#pdb.set_trace()
			for i in range(len(lst)):
				frame_pred = frame_preds[i].copy()
				"""
				for j in range(CLASS):
					if preds[i][j] == 0:
						frame_pred[:, j] *= 0
					else:
						frame_pred[:, j] = uniform_filter1d(
								frame_pred[:, j], size=int(2.5 * win_lens[j]))
				frame_pred = np.maximum(frame_pred, 0)
				"""
				write_sed_scores(
					frame_pred, os.path.join(score_test, lst[i]),
					timestamps=timestamps, event_classes=label_lst
				)
			write_score_transform(
				scores=score_test,
				ground_truth=gt_df,
				filepath=os.path.join(self.evaluation_psds, 'score_transform.tsv'),
				num_breakpoints=800
			)
			score_transform = read_score_transform(
				os.path.join(self.evaluation_psds, 'score_transform.tsv')
			)
			for i in range(len(lst)):
				frame_preds[i] = np.array(
					score_transform(
						create_score_dataframe(
							frame_preds[i],
							timestamps=timestamps,
							event_classes=label_lst
						)
					).iloc[:, 2:]
				)

		for th in test_thresholds:
			# using threshold of th to get clip-level decision
			preds_th = np.zeros_like(preds)
			preds_th[preds >= th] = 1
			result = {}
			for i in range(len(lst)):
				pred = preds_th[i]
				frame_pred = frame_preds[i].copy()
				for j in range(CLASS):
					# If there is not any event for class j
					if pred[j] == 0:
						frame_pred[:, j] *= 0
					else:
						# using median_filter on prediction for the first post-processing
						# frame_pred[:, j] = median_filter(
						# 	frame_pred[:, j], (win_lens[j]))
						frame_pred[:, j] = uniform_filter1d(
							frame_pred[:, j], size=int(2.5*win_lens[j]))
				# making frame-level decision
				frame_decisions = dcase_util.data.ProbabilityEncoder() \
					.binarization(
					probabilities=frame_pred,
					binarization_type='global_threshold',
					threshold=th,
					time_axis=0)

				# using median_filter on decision for the second post-processing
				for j in range(CLASS):
					frame_decisions[:, j] = median_filter(
						frame_decisions[:, j], (win_lens[j]))
				#pdb.set_trace()
				# encode discrete decisions to continuous decisions
				for j in range(CLASS):
					estimated_events = decision_encoder \
						.find_contiguous_regions(
						activity_array=frame_decisions[:, j])

					if lst[i] not in result:
						result[lst[i]] = []

					for [onset, offset] in estimated_events:
						result[lst[i]] += [[str(onset * hop_len),
											str(offset * hop_len),
											label_lst[j]]]
			# save continuous decisions to a file
			path = os.path.join(self.evaluation_psds, mode, f'{round(th,2)}.tsv')
			# path = f"/data/yzr/predictions_dtc0.7_gtc0.7_cttc0.3/predictions_th_{round(th,2)}.tsv"
			# pred_dfs[th] = pd.read_csv(path, sep='\t')
			self.get_predict_csv(result, path=path)
			pred_dfs[th] = pd.read_csv(path, sep='\t',header=None)
			pred_dfs[th].columns=["filename","onset", "offset","event_label"]

		img_save_dir = os.path.join(self.evaluation_psds,mode,"img")
		os.makedirs(img_save_dir, exist_ok=True)

		psds_score_scenario1 = compute_psds_from_operating_points(
			pred_dfs,
			gt_df,
			dur_csv,
			dtc_threshold=0.7,
			gtc_threshold=0.7,
			alpha_ct=0,
			alpha_st=1,
			save_dir=img_save_dir
		)

		psds_score_scenario2 = compute_psds_from_operating_points(
			pred_dfs,
			gt_df,
			dur_csv,
			dtc_threshold=0.1,
			gtc_threshold=0.1,
			cttc_threshold=0.3,
			alpha_ct=0.5,
			alpha_st=1,
			save_dir=img_save_dir
		)

		return psds_score_scenario1, psds_score_scenario2

	def get_psds_tune(self, preds, frame_preds, test_n_thresholds=50, mode='test'):
		""""
		Calculate event detection performance.
		Args:
			preds: numpy.array
				clip-level predicton (posibilities)
			frame_preds: numpy.array
				frame-level prediction
			test_n_thresholds: int
				number of frame-level probability thresholds
		Return:

		"""
		# get current file list
		lst, csv, dur_csv = self.get_vali_lst()
		preds = preds[:len(lst)]

		label_lst = self.label_lst
		win_lens = self.win_lens
		CLASS = self.CLASS
		# get the number of frames of the frame-level predicion
		top_LEN = frame_preds.shape[1]
		# duration (second) per frame
		hop_len = DURATION / top_LEN

		frame_preds = frame_preds[:len(lst)]

		decision_encoder = dcase_util.data.DecisionEncoder(
			label_list=label_lst)

		# shows = []
		# file_lst = []
		test_thresholds = np.arange(
			1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
		)
		pred_dfs = {}

		# Score Transform
		tune_path = os.path.join(self.evaluation_psds, 'tune')
		self.init_dirs(tune_path)
		score_test = os.path.join(self.score_test, str(np.mean(win_lens)))
		if os.path.exists(score_test):
			os.system("rm -rf %s" % (score_test))
		os.mkdir(score_test)
		timestamps = np.linspace(0, DURATION, top_LEN + 1)
		gt_df = pd.DataFrame([s.split() for s in csv], columns=["filename", "onset", "offset", "event_label"])
		gt_df['onset'] = gt_df['onset'].astype(float)
		gt_df['offset'] = gt_df['offset'].astype(float)
		gt_df.columns = ["filename", "onset", "offset", "event_label"]
		if self.score_transform == 'True':
			print('Score Transform')
			for i in range(len(lst)):
				write_sed_scores(
					frame_preds[i], os.path.join(score_test, lst[i]),
					timestamps=timestamps, event_classes=label_lst
				)
			write_score_transform(
				scores=score_test,
				ground_truth=gt_df,
				filepath=os.path.join(tune_path, str(np.mean(win_lens)) + '_score_transform.tsv'),
				num_breakpoints=800
			)
			score_transform = read_score_transform(
				os.path.join(tune_path, str(np.mean(win_lens)) + '_score_transform.tsv')
			)
			for i in range(len(lst)):
				frame_preds[i] = np.array(
					score_transform(
						create_score_dataframe(
							frame_preds[i],
							timestamps=timestamps,
							event_classes=label_lst
						)
					).iloc[:, 2:]
				)

		for th in test_thresholds:
			# using threshold of th to get clip-level decision
			preds_th = np.zeros_like(preds)
			preds_th[preds >= th] = 1
			result = {}
			for i in range(len(lst)):
				pred = preds_th[i]
				frame_pred = frame_preds[i].copy()
				for j in range(CLASS):
					# If there is not any event for class j
					if pred[j] == 0:
						frame_pred[:, j] *= 0
					else:
						# using median_filter on prediction for the first post-processing
						# frame_pred[:, j] = median_filter(
						# 	frame_pred[:, j], (win_lens[j]))
						frame_pred[:, j] = uniform_filter1d(
							frame_pred[:, j], size=int(2.5 * win_lens[j]))
				# making frame-level decision
				frame_decisions = dcase_util.data.ProbabilityEncoder() \
					.binarization(
					probabilities=frame_pred,
					binarization_type='global_threshold',
					threshold=th,
					time_axis=0)

				# using median_filter on decision for the second post-processing
				for j in range(CLASS):
					frame_decisions[:, j] = median_filter(
						frame_decisions[:, j], (win_lens[j]))

				# encode discrete decisions to continuous decisions
				for j in range(CLASS):
					estimated_events = decision_encoder \
						.find_contiguous_regions(
						activity_array=frame_decisions[:, j])

					if lst[i] not in result:
						result[lst[i]] = []

					for [onset, offset] in estimated_events:
						result[lst[i]] += [[str(onset * hop_len),
											str(offset * hop_len),
											label_lst[j]]]
			# save continuous decisions to a file
			path = os.path.join(tune_path, str(np.mean(win_lens)))
			self.init_dirs(path)
			path = os.path.join(path, f'{round(th, 2)}.tsv')
			self.get_predict_csv(result, path=path)
			pred_dfs[th] = pd.read_csv(path, sep='\t', header=None)
			pred_dfs[th].columns = ["filename", "onset", "offset", "event_label"]

		img_save_dir = os.path.join(self.evaluation_psds, mode, "img")
		os.makedirs(img_save_dir, exist_ok=True)

		psds_score_scenario1, psds_score_per_class_1 = compute_psds_from_operating_points_tune(
			pred_dfs,
			gt_df,
			dur_csv,
			dtc_threshold=0.7,
			gtc_threshold=0.7,
			alpha_ct=0,
			alpha_st=1,
			save_dir=img_save_dir
		)

		psds_score_scenario2, psds_score_per_class_2 = compute_psds_from_operating_points_tune(
			pred_dfs,
			gt_df,
			dur_csv,
			dtc_threshold=0.1,
			gtc_threshold=0.1,
			cttc_threshold=0.3,
			alpha_ct=0.5,
			alpha_st=1,
			save_dir=img_save_dir
		)
		return psds_score_scenario1, psds_score_per_class_1.tolist(), psds_score_scenario2, psds_score_per_class_2.tolist()

	def psds_sedt(self, lst, preds, frame_preds, gt_df, pred_dfs, at_m, test_n_thresholds=50, dfs=None, vote=None, mode='test'):
		if preds is None:
			preds = np.ones((dfs[1].shape[0], dfs[1].shape[-1]))
			frame_preds = np.zeros(dfs[1].shape)
		else:
			preds = np.array(preds)
			frame_preds = np.array(frame_preds)
		#pdb.set_trace()
		top_LEN = frame_preds.shape[1]
		label_lst = self.label_lst
		win_lens = self.win_lens
		CLASS = self.CLASS
		# get the number of frames of the frame-level predicion
		top_LEN = frame_preds.shape[1]
		# duration (second) per frame
		hop_len = DURATION / top_LEN
		timestamps = np.linspace(0, DURATION, top_LEN + 1)

		decision_encoder = dcase_util.data.DecisionEncoder(
			label_list=label_lst)

		test_thresholds = np.arange(
			1 / (test_n_thresholds * 2), 1, 1 / test_n_thresholds
		)
		score_test = os.path.join(self.score_test, str(np.mean(win_lens)))
		if os.path.exists(score_test):
			os.system("rm -rf %s" %(score_test))
		os.mkdir(score_test)

		if dfs is not None:
			psds_prediction = dfs[0][at_m]
			SEDT_preds = list()
			SEDT_pred = np.zeros((top_LEN, CLASS))
			for num in range(len(psds_prediction)):
				if num != 0 and psds_prediction.iloc[num]['filename'] != filename:
					SEDT_preds.append(SEDT_pred)
					SEDT_pred = np.zeros((top_LEN, CLASS))
				if not isinstance(psds_prediction.iloc[num]['event_label'], str):
					filename = psds_prediction.iloc[num]['filename']
					continue
				onset = int(psds_prediction.iloc[num]['onset'] / DURATION * top_LEN)
				offset = int(psds_prediction.iloc[num]['offset'] / DURATION * top_LEN)
				score = psds_prediction.iloc[num]['score']
				event_label = label_lst.index(psds_prediction.iloc[num]['event_label'])
				filename = psds_prediction.iloc[num]['filename']
				SEDT_pred[onset:offset + 1, event_label] = score
			SEDT_preds.append(SEDT_pred)
			SEDT_preds = np.array(SEDT_preds)

			SEDP_cnn_preds = dfs[1]
			"""
			for i in range(len(lst)):
				for j in range(CLASS):
					# If there is not any event for class j
					if preds[i][j] == 0:
						SEDP_cnn_preds[:, j] *= 0
					else:
						SEDP_cnn_preds[:, j] = uniform_filter1d(
							SEDP_cnn_preds[:, j], size=int(2.5*win_lens[j]))
			"""
			vote_SEDT = vote[0]
			vote_SEDP_CNN = vote[1]
			assert (vote_SEDT + vote_SEDP_CNN).all() == 1
			print("vote:", vote)
			for num in range(CLASS):
				frame_preds[:, :, num] = vote_SEDT[num] * SEDT_preds[:, :, num] + \
										vote_SEDP_CNN[num] * SEDP_cnn_preds[:, :, num]
			np.save('frame_preds.npy', frame_preds)

		# using threshold of th to get clip-level decision
		th = 0.5
		preds_th = np.zeros_like(preds)
		preds_th[preds >= th] = 1
		result = {}
		for i in range(len(lst)):
			pred = preds_th[i]
			frame_pred = frame_preds[i].copy()

			for j in range(CLASS):
				# If there is not any event for class j
				if pred[j] == 0:
					frame_pred[:, j] *= 0

			# making frame-level decision
			frame_decisions = dcase_util.data.ProbabilityEncoder() \
				.binarization(
				probabilities=frame_pred,
				binarization_type='global_threshold',
				threshold=th,
				time_axis=0)

			# encode discrete decisions to continuous decisions
			for j in range(CLASS):
				estimated_events = decision_encoder \
					.find_contiguous_regions(
					activity_array=frame_decisions[:, j])

				if lst[i] not in result:
					result[lst[i]] = []

				for [onset, offset] in estimated_events:
					result[lst[i]] += [[str(onset * hop_len),
										str(offset * hop_len),
										label_lst[j]]]
		# save continuous decisions to a file
		path = os.path.join(self.evaluation_psds, f'F1.tsv')
		self.get_predict_csv(result, path=path)
		F1_dfs = pd.read_csv(path, sep='\t', header=None)
		F1_dfs.columns = ["filename", "onset", "offset", "event_label"]

		frame_preds[frame_preds > 1] = 1
		frame_preds[frame_preds < 0] = 0
		if self.score_transform == 'True':
			print('Score Transform')
			for i in range(len(lst)):
				frame_pred = frame_preds[i].copy()
				write_sed_scores(
					frame_pred, os.path.join(score_test, lst[i]),
					timestamps=timestamps, event_classes=label_lst
				)
			try:
				write_score_transform(
					scores=score_test,
					ground_truth=gt_df,
					filepath=os.path.join(self.evaluation_psds, 'score_transform.tsv'),
					num_breakpoints=800
				)
			except:
				write_score_transform(
					scores=score_test,
					ground_truth=gt_df,
					filepath=os.path.join(self.evaluation_psds, 'score_transform.tsv'),
					num_breakpoints=360
				)

			score_transform = read_score_transform(
				os.path.join(self.evaluation_psds, 'score_transform.tsv')
			)
			for i in range(len(lst)):
				frame_preds[i] = np.array(
					score_transform(
						create_score_dataframe(
							frame_preds[i],
							timestamps=timestamps,
							event_classes=label_lst
						)
					).iloc[:, 2:]
				)

		for th in test_thresholds:
			# using threshold of th to get clip-level decision
			preds_th = np.zeros_like(preds)
			preds_th[preds >= th] = 1
			result = {}
			for i in range(len(lst)):
				pred = preds_th[i]
				frame_pred = frame_preds[i].copy()

				for j in range(CLASS):
					# If there is not any event for class j
					if pred[j] == 0:
						frame_pred[:, j] *= 0
					else:
						frame_pred[:, j] = uniform_filter1d(
							frame_pred[:, j], size=int(2.5*win_lens[j]))

				# making frame-level decision
				frame_decisions = dcase_util.data.ProbabilityEncoder() \
					.binarization(
					probabilities=frame_pred,
					binarization_type='global_threshold',
					threshold=th,
					time_axis=0)

				# using median_filter on decision for the second post-processing
				for j in range(CLASS):
					frame_decisions[:, j] = median_filter(
						frame_decisions[:, j], (win_lens[j]))

				# encode discrete decisions to continuous decisions
				for j in range(CLASS):
					estimated_events = decision_encoder \
						.find_contiguous_regions(
						activity_array=frame_decisions[:, j])

					if lst[i] not in result:
						result[lst[i]] = []

					for [onset, offset] in estimated_events:
						result[lst[i]] += [[str(onset * hop_len),
											str(offset * hop_len),
											label_lst[j]]]
			# save continuous decisions to a file
			path = os.path.join(self.evaluation_psds, mode, f'{round(th,2)}.tsv')
			self.get_predict_csv(result, path=path)
			pred_dfs[at_m, th] = pd.read_csv(path, sep='\t',header=None)
			pred_dfs[at_m, th].columns=["filename", "onset", "offset", "event_label"]

		return pred_dfs, F1_dfs

		
