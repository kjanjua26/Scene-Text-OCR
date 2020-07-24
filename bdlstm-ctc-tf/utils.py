from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from six.moves.urllib.request import urlretrieve
import os
import sys
import numpy as np
import common

last_percent_reported = None

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    
    for n, seq in enumerate(sequences):
        
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)
        

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape


# load the training or test dataset from disk
def get_data_set(dirname, start_index=None, end_index=None):
    fnames, inputs, codes = common.unzip(list(common.read_data_for_lstm_ctc(dirname, start_index, end_index)))
    inputs = inputs.swapaxes(1, 2)
   
    
    targets = [np.asarray(i) for i in codes]
    
    sparse_targets = sparse_tuple_from(targets)
    
    seq_len = np.ones(inputs.shape[0]) * common.OUTPUT_SHAPE[1]
    
    return fnames, inputs, sparse_targets, seq_len


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = common.DIGITS[spars_tensor[1][m]]
        decoded.append(str)
  
    return decoded


def decode_sparse_tensor(sparse_tensor):
   
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        
        i = i_and_index[0]
        if i != current_i:
            
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset) 
        
    decoded_indexes.append(current_seq)
    print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        
        result.append(decode_a_seq(index, sparse_tensor))
    
    return result
