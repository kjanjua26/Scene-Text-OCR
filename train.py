import time
import tensorflow as tf
import utils
import model2
import codecs

# Configs
num_epochs=utils.num_epochs
logs_path = './tf_logs'
num_epochs = utils.num_epochs
num_hidden = utils.num_hidden
model_layers = utils.model_layers
print ("\n")
print("Hidden Units: "), num_hidden
print ("Model: "), model_layers

def train():
	crnn=model2.crnnNet() #loading the model
	for batch in xrange(utils.BATCHES):	
		tf.reset_default_graph()
		global_step = tf.Variable(0, trainable=False)
       		learning_rate = tf.train.exponential_decay(utils.INITIAL_LEARNING_RATE,global_step,utils.DECAY_STEPS,utils.LEARNING_RATE_DECAY_FACTOR,staircase=True)
		start=time.time()
        	train_names, train_inputs, train_targets = utils.get_data_set('train.txt',batch*utils.BATCH_SIZE,(batch+1)*utils.BATCH_SIZE)
        	print("Data Time: "), time.time() - start
        	print ("\n")
        	logits, inputs, seq_len, Wforward,Wbackward, b = crnn.net(tf.expand_dims(tf.to_float(train_inputs),3))
        	cost = crnn.loss(train_targets,logits, seq_len)
       		optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        	decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        	acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), train_targets))
	    	cost_summary = tf.summary.scalar('cost', cost)
    		val_labelerror = tf.summary.scalar("Label Error on the Validation Set", acc)
	    	gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)
	    	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
			ckpt = tf.train.get_checkpoint_state("models")
	        	if ckpt and ckpt.model_checkpoint_path:
				session.run(tf.local_variables_initializer())
	            		saver = tf.train.Saver()
	            		saver.restore(session, ckpt.model_checkpoint_path)
	        	else:
	            		print("\nNo Checkpoint Found.")
	            		print("Training from scratch.")
	        		init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	            		session.run(init)
	            		saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
	            	summary_op = tf.summary.merge_all()
	            	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	        	for curr_epoch in xrange(num_epochs):
	            		train_cost = train_ler = 0
	   			start = time.time()
	                	c, steps,_,summary = session.run([cost, global_step, optimizer,summary_op])
	                	train_cost = c
	                	train_cost /= utils.TRAIN_SIZE #Dividing the cost by steps.              	
	            		writer.add_summary(summary, steps)
	            		c, val_ler = session.run([cost,acc])

	   	    		log = "\nEpoch {}/{}, steps = {}, train_cost = {:.7f},  val_err = {:.10f}, val_cost = {:10f}, time = {:.3f}s"
        	    		print(log.format(curr_epoch + 1, num_epochs, steps, train_cost,val_ler,c,time.time() - start))
			save_path = saver.save(session, "models/ocr.model-" + str(acc),global_step=steps)	
			print "Saved Model.\n"
		session.close()
	for batch in xrange(utils.VAL_BATCHES):
		tf.reset_default_graph()
		start=time.time()
		val_names, val_inputs, val_targets = utils.get_data_set('valid.txt',batch*utils.BATCH_SIZE,(batch+1)*utils.BATCH_SIZE)
		print("Data Time: ", time.time() - start)
		logits, inputs, seq_len, Wforward,Wbackward, b = crnn.net(tf.expand_dims(tf.to_float(val_inputs),3))
        	cost = crnn.loss(val_targets,logits, seq_len)
		decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        	acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), train_targets))
		with tf.Session() as session:
			ckpt = tf.train.get_checkpoint_state("models")
	        	if ckpt and ckpt.model_checkpoint_path:
				session.run(tf.local_variables_initializer())
	            		saver = tf.train.Saver()
	            		saver.restore(session, ckpt.model_checkpoint_path)
			val_cost = 0
	   		start = time.time()
	                c, val_ler,_ = session.run([cost,acc])
	                val_cost = c
	   	    	log = "Batch {}/{}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s"
        	    	print(log.format(batch+1,utils.VAL_BATCHES, val_cost, val_ler,time.time() - start))


if __name__ == '__main__':
	train()
