# coding=utf-8

import tensorflow as tf
import numpy as np

_EPSION = 1e-9
_MIN_NUM = -np.Inf


def capsule_layer(in_x, out_caps_num, out_caps_dim, iter_num=3, scope=None):

    with tf.variable_scope(scope or 'capsule'):
        caps_uhat = shared_routing_uhat(in_x, out_caps_num, out_caps_dim, scope='caps_uhat')
        V, S = routing(caps_uhat, iter_num)
        return V


def shared_routing_uhat(in_x, out_caps_num, out_caps_dim, scope=None):

    b_sz = tf.shape(in_x)[0]
    tstp = tf.shape(in_x)[1]

    with tf.variable_scope(scope or 'shared_routing_uhat'):
        """ shape (b_sz, caps_num, out_caps_num*out_caps_dim)"""
        caps_uhat = tf.layers.dense(in_x, out_caps_num * out_caps_dim, activation=None)
        caps_uhat = tf.reshape(caps_uhat, shape=[b_sz, tstp, out_caps_num, out_caps_dim])
    return caps_uhat


def routing(caps_uhat, iter_num=3):
    """

    :param caps_uhat: shape(b_sz, tstp, out_caps_num, out_caps_dim)
    :param iter_num: number of iteration
    :return:
        V_ret: shape(b_sz, out_caps_num, out_caps_dim)
    """
    assert iter_num > 0
    b_sz = tf.shape(caps_uhat)[0]
    tstp = tf.shape(caps_uhat)[1]
    out_caps_num = int(caps_uhat.get_shape()[2])

    # shape(b_sz, tstp, out_caps_num)
    B = tf.zeros([b_sz, tstp, out_caps_num], dtype=tf.float32)
    for i in range(iter_num):
        C = tf.nn.softmax(B, dim=2) # shape(b_sz, tstp, out_caps_num)
        C = tf.expand_dims(C, axis=-1) # shape(b_sz,tstp, out_caps_num, 1)
        weighted_uhat = C * caps_uhat # shape(b_sz, tstp, out_caps_num, out_caps_dim)
        S = tf.reduce_sum(weighted_uhat, axis=1) # shape(b_sz, out_caps_num, out_caps_dim)

        V = _squash(S, axes=[2]) # shape(b_sz, out_caps_num, out_caps_dim)
        V = tf.expand_dims(V, axis=1) # shape(b_sz, 1 , out_caps_num, out_caps_dim)
        if i < iter_num - 1:
            B = tf.reduce_sum(caps_uhat * V, axis=-1) + B # shape(b_sz, tstp, out_caps_num)

    V_ret = tf.squeeze(V, axis=[1]) # shape(b_sz, out_caps_num, out_caps_dim)
    S_ret = S
    return V_ret, S_ret


def _squash(in_caps, axes):

    vec_squared_norm = tf.reduce_sum(tf.square(in_caps), axis=axes, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + _EPSION)
    vec_squashed = scalar_factor * in_caps # element-wise
    return vec_squashed



























