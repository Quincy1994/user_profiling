# coding=utf-8
import numpy as np
import tensorflow as tf

def cross_entroy(y, preds):
    # y = tf.argmax(y, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
    loss = tf.reduce_mean(loss)
    return loss

def margin_loss(y, preds):
    y = tf.cast(y, tf.float32)
    loss = y * tf.square(tf.maximum(0., 0.9 - preds)) + 0.25 * (1.0 -y) * tf.square(tf.maximum(0., preds - 0.1))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss
