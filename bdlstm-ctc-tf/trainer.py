from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import editdistance
import time
import tensorflow as tf
import common, model
import utils
from utils import decode_sparse_tensor
import codecs
# Configs
num_epochs=common.num_epochs
print("num_hidden:", common.num_hidden, "num_layers:", common.num_layers)

def report_accuracy(decoded_list, test_targets,test_names):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    names_list = test_names.tolist()
    total_ed = 0
    total_len = 0


    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):

	detect_number = detected_list[idx]
        ed = editdistance.eval(number,detect_number)
        ln = len(number)
        edit_accuracy = (ln - ed) / ln
	print("Edit: ", ed, "Edit accuracy: ", edit_accuracy, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        total_ed+=ed
        total_len+=ln

    accuraccy = (total_len -  total_ed) / total_len
    print("Test Accuracy:", accuraccy)
    return accuraccy


def train():
    test_names, test_inputs, test_targets, test_seq_len = utils.get_data_set('valid.txt')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, targets, seq_len, W, b = model.get_train_model()
    loss = tf.nn.ctc_loss(logits, targets, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=common.MOMENTUM).minimize(cost, global_step=global_step)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))



    def do_report():
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        accuracy = report_accuracy(dd, test_targets,test_names)
        save_path = saver.save(session, "models/ocr.model-" + str(accuracy), global_step=steps)
        # decoded_list = decode_sparse_tensor(dd)

    def do_batch():
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        b_cost, steps, _ = session.run([cost, global_step, optimizer], feed)
        if steps > 0 and steps % common.REPORT_STEPS == 0:
            do_report()
        return b_cost, steps
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.49)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True,gpu_options=gpu_options)) as session:
        ckpt = tf.train.get_checkpoint_state("models")
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("no checkpoint found")
            # Initializate the weights and biases
            init = tf.initialize_all_variables()
            session.run(init)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)
        for curr_epoch in xrange(num_epochs):

            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            for batch in xrange(common.BATCHES):
                start = time.time()
                train_names, train_inputs, train_targets, train_seq_len = utils.get_data_set('train.txt', batch * common.BATCH_SIZE, (batch + 1) * common.BATCH_SIZE)

                print("get data time", time.time() - start)
                start = time.time()
                c, steps = do_batch()
                train_cost += c * common.BATCH_SIZE
                seconds = time.time() - start
                print("Step: ", steps, ", batch seconds: ", seconds)

            train_cost /= common.TRAIN_SIZE

            val_feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len}

            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler,
                             time.time() - start, lr))


if __name__ == '__main__':
    train()