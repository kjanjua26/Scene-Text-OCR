from scipy import misc
import numpy as np
import time
import os
import codecs
import sys
import tensorflow as tf
import cPickle as pickle
import glob
from label_generate import DIGITS

DIGITS = DIGITS.decode('utf-8')

LEARNING_RATE_DECAY_FACTOR = 0.9
INITIAL_LEARNING_RATE = 1e-2
DECAY_STEPS = 50
BATCH_SIZE = 125
BATCHES = 80
VAL_BATCHES = 100
TRAIN_SIZE = BATCH_SIZE * BATCHES
MOMENTUM = 0.9
REPORT_STEPS = 100
num_epochs = 1000
num_classes = len(DIGITS)+1
seq_length = 24
num_hidden = 256
model_layers = "CNN + LSTM + CTC"
data_set = {}


def get_data_set(dirname, start_index=None, end_index=None):
	fnames, inputs, codes = unzip(list(read_data_for_lstm_ctc(dirname, start_index, end_index)))
    	inputs = inputs.swapaxes(1, 2)
    	targets = [np.asarray(i) for i in codes]
    	sparse_targets = sparse_tuple_from(targets)
    	return fnames, inputs, sparse_targets#, seq_len


def read_data_for_lstm_ctc(dirname, start_index=None, end_index=None):
	label_dictionary = {}
	fname_list = []
	fname_batch_list = []
	labels_list = []
    	if not data_set.has_key(dirname):
        	with open(dirname) as f:
			image_names = f.readlines()
   			fname_list1 = [x.strip() for x in image_names]  #removes \n	
			data_set[dirname] = fname_list1
	fname_list=data_set.get(dirname)
	if start_index is None:
		fname_batch_list=fname_list
	else:
		for i in range(start_index, end_index):
            		fname_batch_list.append(fname_list[i])
	for x in fname_batch_list:
        	labels_list.append((os.path.splitext(x)[0]))
	for x in labels_list:
        	f = codecs.open( x +'.gt.txt','r','utf-8')
        	label_dictionary[x] = f.read().strip('\n')
        	label_dictionary[x]=label_dictionary[x][::-1]
        	f.close()

   	for fname in sorted(fname_batch_list):
        	im = misc.imread(fname,True,'L').astype(np.float32) / 255.
		im=misc.imresize(im,[32,100])
        	code = label_dictionary.get(os.path.splitext(fname)[0])
        	d = os.path.splitext(fname)[0]
		yield np.asarray(d), im, np.asarray([DIGITS.find(x) for x in list(code)])




def unzip(b):
	ns, xs, ys = zip(*b)

   	xs = np.array(xs)
    	ys = np.array(ys)
    	ns = np.array(ns)
	return ns, xs, ys


def sparse_tuple_from(sequences, dtype=np.int32):
	indices = []
    	values = []
       	for n, seq in enumerate(sequences):
		indices.extend(zip([n] * len(seq),xrange(len(seq))))
        	values.extend(seq)
       
	indices = np.asarray(indices, dtype=np.int64)
    	values = np.asarray(values, dtype=dtype)
	values[values<0]=0
    	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
	tensor= tf.SparseTensor(indices, values, shape)
	return tensor





