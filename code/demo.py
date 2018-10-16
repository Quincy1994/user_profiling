# # coding=utf-8
#
# import tensorflow as tf
import numpy as np
#
# # atten_size = 5
# # u_omega = tf.get_variable("u_omega", initializer= tf.truncated_normal([atten_size], stddev=0.1))
#
# # def attention_layer():
#
# batch = 2
# sequence_length = 5
# hidden_size = 10
# attention_size = 7
#
# inputs = tf.get_variable("inputs", initializer=tf.truncated_normal([batch, sequence_length, hidden_size], stddev=0.1))
#
# # shape of inputs: [batch, sequence_length, hidden_size]
# sequence_length = inputs.get_shape()[1].value
# hidden_size = inputs.get_shape()[2].value
#
# # Attention mechanism
# W_omega = tf.get_variable("W_omega", initializer= tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
# b_omega = tf.get_variable("b_omega", initializer= tf.truncated_normal([attention_size], stddev=0.1))
# u_omega = tf.get_variable("u_omega", initializer= tf.truncated_normal([attention_size, 1], stddev=0.1))
#
# v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
# vu = tf.matmul(v, u_omega)
# exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
# alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
# output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
# #     # return output
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     # inputs = tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1])
#     inputs = sess.run(alphas)
#     print(inputs)
#     print(np.shape(inputs))
#     # b = sess.run(tf.reshape(b_omega, [1, -1]))
#     # print(b)
#     # print(np.shape(b))

a = [1, 2, 3]
print(np.mean(a)[0])