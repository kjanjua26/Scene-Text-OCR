import common
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import time

# Utility functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)


def bias_variable(shape):
    #print(type(shape))
    #time.sleep(300)
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_train_model():
    inputs = tf.placeholder(tf.float32, [None, None, common.OUTPUT_SHAPE[0]])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell for forward and backward layer
    forwardH1 = rnn_cell.LSTMCell(common.num_hidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(common.num_hidden, use_peepholes=True, state_is_tuple=True)

    # The second output previous state and is ignored
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(forwardH1,backwardH1,inputs,seq_len,dtype=tf.float32)
    outputs=tf.concat(2,outputs)
    shape = tf.shape(inputs)

    batch_s, max_timesteps = shape[0], shape[1]
    weights = tf.Variable(tf.truncated_normal([common.num_hidden,
                                         common.num_classes],
                                        stddev=0.1), name="weights")
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, 2*common.num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    W = tf.Variable(tf.truncated_normal([2*common.num_hidden,
                                         common.num_classes],
                                        stddev=0.1), name="W")

    # Zero initialization
    b = tf.zeros(shape=[common.num_classes],name='b')

    # Doing the affine projection
    logits = tf.matmul(outputs, W)+b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, common.num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, targets, seq_len, W, b
