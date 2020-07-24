import glob
import numpy
import cv2
import numpy as np
import cPickle as pickle
import time
import os
import codecs
__all__ = (
    'DIGITS'
)
OUTPUT_SHAPE = (48, 256)

#DIGITS = "~!%'()+,-.\/0123456789:ABCDEFGIJKLMNOPRSTUVWYabcdefghiklmnoprstuvwxz-V،د‘“ ؤب,گ0ذصط3وLِbT2dh9ٰٴxAڈlژ؛؟أGاpث4/س7ًtCهKیُS\"۔WOcgk…ٓosw(ﷺجڑ.آئکتخز6غEشہقنضDNR8ظ:fnrvzپچB’”لء%)ْFحر5عںھف!JمIM#ّےUYَae'Pimة1uٹ+" #url
DIGITS = "0123456789GHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`~!@#$%^&*()_+-=[];',./{}|:\"<>?\\" #all


DIGITS = DIGITS.decode('utf-8')

CHARS = DIGITS
LENGTHS =[15,16,17,18]
TEST_SIZE = 10
#100
ADD_BLANK = False
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
INITIAL_LEARNING_RATE = 1e-4
DECAY_STEPS = 5000

# parameters for bdlstm ctc
BATCH_SIZE = 75
BATCHES = 6
#100000


TRAIN_SIZE = BATCH_SIZE * BATCHES

MOMENTUM = 0.9
REPORT_STEPS = 1000

# Hyper-parameters
num_epochs = 2000
num_hidden = 256
num_layers = 1

num_classes = len(DIGITS)  + 1  # characters + ctc blank
print num_classes


data_set = {}
label_dictionary = {}

def get_labels(names):
    for x in names:
        f = codecs.open( x +'.gt.txt', 'r','utf-8')
        label_dictionary[x] = f.read().strip('\n')
        label_dictionary[x]=label_dictionary[x][::-1]
        f.close()

def load_data_set(dirname):
    with open(dirname) as f:
        image_names = f.readlines()
    fname_list = [x.strip() for x in image_names]  #removes \n
    result = dict()
    labels_list = []
    #get list of paths without extension
    for x in fname_list:
        labels_list.append((os.path.splitext(x)[0]))

    #load ground truths to label array
    get_labels(labels_list)

    for fname in sorted(fname_list):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        #get corresponding label
        code = label_dictionary.get(os.path.splitext(fname)[0])
        result[(os.path.splitext(fname)[0])] = (im, code)


    data_set[dirname] = result


def read_data_for_lstm_ctc(dirname, start_index=None, end_index=None):
    start = time.time()
    fname_list = []
    if not data_set.has_key(dirname):
        load_data_set(dirname)

    if start_index is None:
        with open(dirname) as f:
            image_names = f.readlines()
        fname_list = [x.strip() for x in image_names]  #removes \n



    else:
        with open(dirname) as f:
            inames = f.readlines()
            iname_list = [x.strip() for x in inames]  #removes \n
        for i in range(start_index, end_index):
            fname_index = iname_list[i]
            fname_list.append(fname_index)

    start = time.time()
    dir_data_set = data_set.get(dirname)


    for fname in sorted(fname_list):
        im, code = dir_data_set[os.path.splitext(fname)[0]]
        d = os.path.splitext(fname)[0]
        yield numpy.asarray(d), im, numpy.asarray([DIGITS.find(x)  for x in list(code)])


def unzip(b):
    ns, xs, ys = zip(*b)

    xs = numpy.array(xs)
    ys = numpy.array(ys)
    ns = numpy.array(ns)
    return ns, xs, ys
