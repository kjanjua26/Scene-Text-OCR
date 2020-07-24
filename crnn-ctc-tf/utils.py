import editdistance
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
INITIAL_LEARNING_RATE = 0.1
DECAY_STEPS = 10000
BATCH_SIZE = 100
BATCHES = 100
VAL_BATCHES = 100
TEST_BATCHES = 
TRAIN_SIZE = BATCH_SIZE * BATCHES
MOMENTUM = 0.9
REPORT_STEPS = 100
num_epochs=4
num_classes = len(DIGITS)+1
seq_length = 24


data_set = {}


def get_data_set(dirname, start_index=None, end_index=None):
	fnames, inputs, codes = unzip(list(read_data_for_lstm_ctc(dirname, start_index, end_index)))
    	inputs = inputs.swapaxes(1, 2)
    	targets = [np.asarray(i) for i in codes]
    	sparse_targets = sparse_tuple_from(targets)
    #	seq_len = np.ones(inputs.shape[0]) * seq_length 
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
	#get list of paths without extension    	
	for x in fname_batch_list:
        	labels_list.append((os.path.splitext(x)[0]))
    
	#load ground truths to label array
	for x in labels_list:
        	f = codecs.open( x +'.gt.txt','r','utf-8')
        	label_dictionary[x] = f.read().strip('\n')
        	label_dictionary[x]=label_dictionary[x][::-1]
        	f.close()

   	for fname in sorted(fname_batch_list):
        	im = misc.imread(fname,True,'L').astype(np.float32) / 255.
       	#	get corresponding label
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


def decode_sparse_tensor(sparse_tensor):
	decoded_indexes = list()
    	current_i = 0
    	current_seq = []
    	rows=tf.unstack(sparse_tensor.indices,axis=1)
	for offset, row in enumerate(rows):
        	i = row
        	if i != current_i:
            		decoded_indexes.append(current_seq)
            		current_i = i
            		current_seq = list()
        	current_seq.append(offset) 
    	decoded_indexes.append(current_seq)
#    	print("decoded_indexes = ", decoded_indexes)
    	result = []
    	for index in decoded_indexes:
        	result.append(decode_a_seq(index, sparse_tensor))
    
	return result

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
	a=tf.gather(spars_tensor.values,m)
        str = DIGITS[a.eval()]
        decoded.append(str)
  
	return decoded

        
def report_accuracy(decoded_list, test_targets,test_names):
    	original_list = decode_sparse_tensor(test_targets)
    	detected_list = decode_sparse_tensor(decoded_list)
    	names_list = test_names.tolist()
    	total_ed = 0
    	total_len = 0

#	print original_list
#	print detected_list
    	if len(original_list) != len(detected_list):
#        	print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list)," test and detect length desn't match")
        	return
#   	print("T/F: original(length) <-------> detectcted(length)")
    	for idx, number in enumerate(original_list):
		if(idx>0):
			detect_number = detected_list[idx]
 	       		ed = editdistance.eval(number,detect_number)
        		ln = len(number)
        		edit_accuracy = (ln - ed) / ln
#			print("Edit: ", ed, "Edit accuracy: ", edit_accuracy, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        		total_ed+=ed
        		total_len+=ln

    	accuraccy = (total_len -  total_ed) / total_len
#    	print("Test Accuracy:", accuraccy)
    	return accuraccy

