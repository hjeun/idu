import numpy as np
import os
import tensorflow as tf
import pickle
import argparse
import model_IDU as IDU
import utils

all_class_name = ['Pick something up',
                   'Point',
                   'Drink',
                   'Stand up',
                   'Run',
                   'Sit down',
                   'Read',
                   'Smoke',
                   'Drive car',
                   'Open door',
                   'Give something',
                   'Use computer',
                   'Write',
                   'Close door',
                   'Go down stairway',
                   'Go up stairway',
                   'Throw something',
                   'Get in/out of car',
                   'Hang up phone',
                   'Eat',
                   'Answer phone',
                   'Clap',
                   'Dress up',
                   'Undress',
                   'Kiss',
                   'Fall/trip',
                   'Wave',
                   'Pour',
                   'Punch',
                   'Fire weapon']

parser = argparse.ArgumentParser()
parser.add_argument("--feat_type", type=str, default='incepv3')
parser.add_argument("--gpu_id", type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

BATCH_SIZE = 100
N_CHUNKS = 16
N_CLASSES = 31

feat_type = args.feat_type
test_data_fname = 'data/tvseries_' + feat_type + '_test_data.pickle'
test_sample_fname = 'data/tvseries_' + feat_type + '_test_samples_{}.pickle'.format(N_CHUNKS)
test_data = pickle.load(open(test_data_fname, 'rb'))
test_samples = pickle.load(open(test_sample_fname, 'rb'))

rgb_feat_dim = 2048
flow_feat_dim = 1024 if feat_type == 'anet2016' else 2048

rgb_feat_ph = tf.placeholder(tf.float32, shape=(None, N_CHUNKS, rgb_feat_dim))
flow_feat_ph = tf.placeholder(tf.float32, shape=(None, N_CHUNKS, flow_feat_dim))
class_ph = tf.placeholder(tf.float32, shape=(None, N_CHUNKS, N_CLASSES))
relevance_ph = tf.placeholder(tf.float32, shape=(None, N_CHUNKS))

model = IDU.Model(rgb_feat_ph, flow_feat_ph, class_ph, relevance_ph, 1.0)
test_op, gate_op = model.testing_operation()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'tvseries/models/best-' + feat_type)

    all_probs, all_classes = list(), list()
    print('*testing... IDU-{} on TVSeries.*'.format(feat_type))
    for step in range(len(test_samples)//BATCH_SIZE + 1):
        rgb_feat_batch, flow_feat_batch, class_batch, relevance_batch, t0_class_batch, _ = \
            utils.next_batch_for_test(step, test_samples, test_data, BATCH_SIZE, N_CHUNKS)
        prob_val = sess.run(test_op,
                            feed_dict={rgb_feat_ph: rgb_feat_batch,
                                       flow_feat_ph: flow_feat_batch,
                                       class_ph: class_batch,
                                       relevance_ph: relevance_batch})

        all_probs += list(prob_val)
        all_classes += t0_class_batch

    all_probs = np.asarray(all_probs).T
    all_classes = np.eye(N_CLASSES)[np.asarray(all_classes)].T
    results = {'probs': all_probs, 'labels': all_classes}        
    
    _, _, mcap, caps = utils.frame_level_map_n_cap(results)
    print('[IDU-{}] mcAP: {:.4f}\n'.format(feat_type, mcap))
    
for i, cap in enumerate(caps):
    cls_name = all_class_name[i]
    print('{}: {:.4f}'.format(cls_name, cap))


