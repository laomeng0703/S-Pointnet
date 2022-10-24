import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from utils.pointSIFT_util import pointSIFT_module, pointSIFT_res_module


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None, feature=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    net = feature

    #input_image = tf.expand_dims(point_cloud, -1)
    # CONV
    point_cloud, net, _ = pointSIFT_res_module(point_cloud, net, radius=0.5, out_channel=64,
                                                   is_training=is_training, bn_decay=bn_decay, scope='layer1_local')
    net = tf_util.conv1d(point_cloud, 64, 9, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 64, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 64, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    
    '''_, points_feat_local, _ = pointSIFT_res_module(point_cloud, net, radius=0.5, out_channel=64,
                                                   is_training=is_training, bn_decay=bn_decay, scope='layer1_local') #local feature'''
    
    net = tf_util.conv1d(net, 128, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 256, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    points_feat1 = tf_util.conv1d(net, 1024, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv6', bn_decay=bn_decay)
    
    
    _, points_feat_global, _ = pointSIFT_res_module(point_cloud, points_feat1, radius=0.5, out_channel=1024,
                                                    is_training=is_training, bn_decay=bn_decay, scope='layer1_global')
    pc_feat_global = tf.expand_dims(points_feat_global, -2)
    
    # MAX
    pc_feat_global = tf_util.max_pool2d(pc_feat_global, [num_point,1], padding='VALID', scope='maxpool1') #global feature
    # FC
    pc_feat1 = tf.reshape(pc_feat_global, [batch_size, -1])
    
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    #pc_feat1 = tf.squeeze(pc_feat1, [2])
   
    # CONCAT 
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, -1]), [1, num_point, 1])
    points_feat1_concat = tf.concat(axis=2, values=[points_feat_global, pc_feat1_expand])
    
    # CONV 
    net = tf_util.conv1d(points_feat1_concat, 512, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv7')
    net = tf_util.conv1d(net, 256, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv8')
    net = tf_util.conv1d(net, 128, 1, padding='SAME', stride=1,
                         bn=True, is_training=is_training, scope='conv9')
    _, net, _ = pointSIFT_module(point_cloud, net, radius=0.5, out_channel=128,
                                 is_training=is_training, bn_decay=bn_decay, scope='layer_feat')
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    
    net = tf_util.conv1d(net, 13, 1, padding='SAME', stride=1, activation_fn=None, scope='conv10')

    return net

def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32,4096,9))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,4096,9)})
            print(time.time() - start)
