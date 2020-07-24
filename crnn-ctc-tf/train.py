import time
import tensorflow as tf
import utils
import mymodel
import codecs

# Configs
num_epochs=utils.num_epochs

def train():
	mycounter=3920
	crnn=mymodel.crnnNet()
	for curr_epoch in xrange(num_epochs):
		avg_acc2=0.0
		if curr_epoch % 3 == 0:
			mycounter +=1
		for batch in xrange(utils.BATCHES):	
			tf.reset_default_graph()
			global_step = tf.Variable(0, trainable=False)
       			learning_rate = tf.train.exponential_decay(utils.INITIAL_LEARNING_RATE,global_step,utils.DECAY_STEPS,utils.LEARNING_RATE_DECAY_FACTOR,staircase=True)
			start=time.time()
        		train_names, train_inputs, train_targets = utils.get_data_set('../train.txt',batch*utils.BATCH_SIZE,(batch+1)*utils.BATCH_SIZE)
        		print("get data time", time.time() - start)
        		logits, inputs, seq_len, Wforward,Wbackward, b = crnn.net(tf.expand_dims(train_inputs,3))
        		cost = crnn.loss(train_targets,logits, seq_len)
       			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        		decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        		acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), train_targets))

	   		with tf.Session() as session:
				ckpt = tf.train.get_checkpoint_state("models")
		       		if ckpt and ckpt.model_checkpoint_path:
					session.run(tf.local_variables_initializer())
	           			saver = tf.train.Saver()
	          			saver.restore(session, ckpt.model_checkpoint_path)
		     		else:
	         			print("No checkpoint found.")
	         			print("Trainng from scratch.")
	    	   			init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	           			session.run(init)
	           			saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
	          		train_cost = train_ler = 0.0
	 			start = time.time()
	 	            	c, steps,_,dd,accu= session.run([cost, global_step, optimizer,decoded[0],acc])
				accuracy=utils.report_accuracy(dd,train_targets,train_names)
				avg_acc2 += accuracy
	                	train_cost = c
				train_ler = accu
	   	    		log = "Batch {}/{} : Epoch {}/{}, steps = {}, train_cost = {:.3f}, accuracy = {:.7f}, time = {:.3f}s"
        	    		print(log.format(batch+1,utils.BATCHES,curr_epoch + 1, num_epochs, steps, train_cost,accuracy,time.time() - start))
				save_path = saver.save(session, "models/ocr.model-", global_step=mycounter)	
			session.close()
		avg_acc2 /=utils.BATCHES
		print("\n train set accuracy = ",avg_acc2)

		val_cost=0.0
		val_ler =0.0
		avg_acc1=0.0
		print("\n\n\Valid cost\n\n\n")
		for batch in xrange(utils.VAL_BATCHES):
			tf.reset_default_graph()
			start=time.time()
			val_names, val_inputs, val_targets = utils.get_data_set('../valid.txt',batch*utils.BATCH_SIZE,(batch+1)*utils.BATCH_SIZE)
			print("get data time", time.time() - start)
			logits, inputs, seq_len, Wforward,Wbackward, b = crnn.net(tf.expand_dims(val_inputs,3))
        		cost = crnn.loss(val_targets,logits, seq_len)
			decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
        		acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), val_targets))
			with tf.Session() as session:
				ckpt = tf.train.get_checkpoint_state("models")
		       		if ckpt and ckpt.model_checkpoint_path:
					session.run(tf.local_variables_initializer())
	           			saver = tf.train.Saver()
	          			saver.restore(session, ckpt.model_checkpoint_path)
					start = time.time()
		               		c,dd,log_prob,ler = session.run([cost,decoded[0],log_prob,acc])
					accuracy=utils.report_accuracy(dd,val_targets,val_names)
		               		avg_acc1+=accuracy
					val_cost += c
					val_ler += ler
	  	    			log = "Batch {}/{}, batch_cost = {:.3f}, batch_ler = {:.3f},acc = {:.3f}, time = {:.3f}s"
        	    			print(log.format(batch+1,utils.VAL_BATCHES,c,ler,accuracy,time.time() - start))
				else:
					session.close()
					print("no checkpoint found")
					break
			session.close()
		val_cost /=utils.VAL_BATCHES
		val_ler /=utils.VAL_BATCHES
		avg_acc1 /=utils.VAL_BATCHES
		log = "\n\nepoch = {}/{} , Avg val cost = {:.3f}, Avg val ler ={:.3f},avg accuracy ={:.3f}\n\n"
		print(log.format(curr_epoch+1,num_epochs,val_cost,val_ler,avg_acc1))



		test_cost=0.0
        	test_ler =0.0
		avg_acc =0.0
                print("\n\n\ntest cost\n\n\n")
                for batch in xrange(utils.TEST_BATCHES):
       	                tf.reset_default_graph()
        	        start=time.time()
                        test_names, test_inputs, test_targets = utils.get_data_set('../tests.txt',batch*utils.BATCH_SIZE,(batch+1)*utils.BATCH_SIZE)
                        print("get data time", time.time() - start)
                        logits, inputs, seq_len, Wforward,Wbackward, b = crnn.net(tf.expand_dims(test_inputs,3))
                        cost = crnn.loss(test_targets,logits, seq_len)
                        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
                        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), test_targets))
                        with tf.Session() as session:
                                ckpt = tf.train.get_checkpoint_state("models")
                   		if ckpt and ckpt.model_checkpoint_path:
                                	session.run(tf.local_variables_initializer())
                 	                saver = tf.train.Saver()
                	                saver.restore(session, ckpt.model_checkpoint_path)
                                        start = time.time()
                                        c,dd,log_probs,ler = session.run([cost,decoded[0],log_prob,acc])
                                        test_cost += c
                                        test_ler += ler
		   	                accuracy=utils.report_accuracy(dd,test_targets,test_names)
					avg_acc += accuracy
                                        log = "Batch {}/{}, batch_cost = {:.3f}, batch_ler = {:.3f},batch_accuracy = {:.3f}, time = {:.3f}s"
                                        print(log.format(batch+1,utils.VAL_BATCHES,c,ler,accuracy,time.time() - start))
				else:
                                        session.close()
                                        print("no checkpoint found")
                                        break
                        session.close()
                test_cost /=utils.TEST_BATCHES
		test_ler /=utils.TEST_BATCHES
		avg_acc /=utils.TEST_BATCHES
                log = "\n\nepoch = {}/{} , Avg test cost = {:.3f}, Avg test ler ={:.3f},Avg test accuracy = {:.3f}\n\n"
                print(log.format(curr_epoch+1,num_epochs,test_cost,test_ler,avg_acc))




if __name__ == '__main__':
        train()
