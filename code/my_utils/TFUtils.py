# coding=utf-8
import tensorflow as tf


def entry_stop_gradients(target, mask):
    """

    :param target: a tensor
    :param mask: a boolean tensor that broadcast to the rank of that to target tensor
    :return:
        ret: a tensor have the same value of target,
            but some entry will have no gradient during backprop
    """
    mask_h = tf.logical_not(mask)
    mask = tf.cast(mask, dtype=target.dtype)
    mask_h = tf.cast(mask_h, dtype=target.dtype)
    ret = tf.stop_gradient(mask_h * target) + mask * target

    return ret