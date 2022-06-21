import os
import numpy as np

def calcu(path, out_file):
	with open(path) as f:
		fs = f.readlines()
	labels = np.zeros([10])
	cnt = 0
	for f in fs:
		path = os.path.join('data/label', f.rstrip() + '.npy')
		labels += np.load(path)
		cnt += 1
	w = labels / (cnt - labels)
	print(w)
	np.save(out_file, np.reshape(w, [1, 10]))

if __name__ == '__main__':
	#path = 'data/text/weak-2018.lst'
	#out_file = 'weak_weights'
	path = 'data/text/syn-123.lst'
	out_file = 'syn_weights'
	calcu(path, out_file)
