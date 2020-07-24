import tensorflow as tf
import numpy as np
import utils
from collections import namedtuple
from tensorflow.contrib import rnn
slim = tf.contrib.slim


crnnParameters = namedtuple('crnnpara',['nk','ks','ih','iw','batch_size','hidden','classes','seq_len'])

class crnnNet(object):
	
	params=crnnParameters(nk=[64, 128, 256, 256, 512, 512, 512],ks=[3,3,3,3,3,3,2],ih=32,iw=100,batch_size=utils.BATCH_SIZE,hidden=256,classes=utils.num_classes,seq_len=utils.seq_length)

	def net(self,inputs):
		with slim.arg_scope([slim.conv2d],activation_fn=None):
		
			def conv2d(inputs,i,batchNormalization=False):
				outChannels=self.params.nk[i]
				if i!=6:
					net = slim.conv2d(inputs,outChannels,3,scope = 'conv_{}'.format(i+1))
				else:
					net = slim.conv2d(inputs,outChannels,2,padding='VALID',scope = 'conv_{}'.format(i+1))
				if(batchNormalization):
					net = slim.batch_norm(net,scope='batchnorm{}'.format(i+1))
				net=tf.nn.relu(net)
				return net

	
		def blstm(layers,inputs,hidden,seq_length,classes):
			f_cell=rnn.LSTMCell(hidden,use_peepholes=True,state_is_tuple=True)
			b_cell=rnn.LSTMCell(hidden,use_peepholes=True,state_is_tuple=True)
			outputs,_=tf.nn.bidirectional_dynamic_rnn(f_cell,b_cell,inputs,seq_length,dtype=tf.float32)
			f_out,b_out=outputs
			forward_output = tf.reshape(f_out,[-1,hidden])
			backward_output = tf.reshape(b_out,[-1,hidden])
			shape = tf.shape(inputs)
			batch_s, max_timesteps = shape[0], shape[1]
			Wf = tf.Variable(tf.truncated_normal([hidden,classes],stddev=0.1), name="Wf")
			Wb = tf.Variable(tf.truncated_normal([hidden,classes],stddev=0.1), name="Wb")
			b = tf.zeros(shape=[classes],name='b')
			logits = tf.matmul(forward_output, Wf)+ tf.matmul(backward_output,Wb) + b
			logits = tf.reshape(logits, [batch_s, -1, classes])
			logits = tf.transpose(logits, (1, 0, 2))
			return logits, inputs, seq_length, Wf,Wb, b			


	
		with tf.variable_scope("crnnNet"):
			net = conv2d(inputs,0)
			net = slim.max_pool2d(net,[2,2],[2,2],scope='pool1')
			net = conv2d(net,1)
			net = slim.max_pool2d(net,[2,2],[2,2],scope='pool2')
			net = conv2d(net,2)
			net = conv2d(net,3)
			net = slim.max_pool2d(net,[1,2],[1,2],scope='pool3')
			net = conv2d(net,4,True)
			net = conv2d(net,5,True)
			net = slim.max_pool2d(net,[1,2],[1,2],scope='pool4')
			net = conv2d(net,6)
			net = tf.squeeze(net)
			seq_length=np.ones(self.params.batch_size)*self.params.seq_len
			return blstm(2,net,self.params.hidden,seq_length,self.params.classes)
			

	def loss(self,targets,logits,seq_len,scope='losses'):	
		with tf.name_scope(scope):
			loss = tf.nn.ctc_loss(targets,logits,seq_len)
			cost = tf.reduce_mean(loss)
		return cost
