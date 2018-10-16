# coding=utf-8

import tensorflow as tf
from Config import config
from my_utils.data_preprocess import get_vocab
from nn_utils.base_nn import kim_cnn_layer, biRNNLayer, auxAttention, gate_layer, fc_layer, self_attention, cnn_layer, max_pooling, avg_pooling
import numpy as np
from nn_utils.capsule_layer import capsule_layer

class CNN_RNN_ATT(object):

    def __init__(self, config):

        self.input_x = tf.placeholder(tf.int32, [None, config.word_seq_maxlen], name="input_x")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.input_y = tf.placeholder(tf.int32, [None, config.num_class], name="input_y")

        self.config = config
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.on_epoch = tf.Variable(0, name='epoch_count', trainable=False)
        self.on_epoch_accu = tf.assign_add(self.on_epoch, 1)

        with tf.name_scope("embedding_layer"):
            vocab = get_vocab()
            vocab_sz = len(vocab) + 2
            print("vocab sz is %d" %(vocab_sz))
            embed_matrix = tf.random_uniform([vocab_sz, self.config.embed_size], -0.25, 0.25)
            self.embedding = tf.Variable(embed_matrix, name="embedding", trainable=True)
            self.w_embedded = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("cnn_layer"):
            self.cnn_out, self.conv_ouput = kim_cnn_layer(self.w_embedded, config.embed_size, config.filers_sizes, num_filters=128, scope="cnn_kim")
        #     self.cnn_3, self.cnn_4, self.cnn_5 = self.conv_ouput[0], self.conv_ouput[1], self.conv_ouput[2]

        # with tf.name_scope("cnn_capsule_3"):
        #     # self.cnn = cnn_layer(self.w_embedded, config.embed_size, filter_size=3, num_filters=128, scope='cnn_3')
        #     self.capsule_encode = capsule_layer(self.cnn_3, out_caps_num=16, out_caps_dim=32, scope="capsule_3")
        #     self.capsule_rnn_3 = biRNNLayer(self.capsule_encode, hidden_size=50, scope="rnn_3")
        #     self.capsule_attn_3 = auxAttention(self.capsule_rnn_3, self.cnn_out, attention_size=64, scope='atten_3')
        #     # self.capsule_attn_3 = tf.reshape(self.capsule_attn_3, [-1, 16*32])
        #
        # with tf.name_scope("cnn_capsule_4"):
        #     # self.cnn = cnn_layer(self.w_embedded, config.embed_size, filter_size=4, num_filters=128, scope='cnn_4')
        #     self.capsule_encode = capsule_layer(self.cnn_4, out_caps_num=16, out_caps_dim=32, scope="capsule_4")
        #     self.capsule_rnn_4 = biRNNLayer(self.capsule_encode, hidden_size=50, scope="rnn_3", reuse=True)
        #     self.capsule_attn_4 = auxAttention(self.capsule_rnn_4, self.cnn_out, attention_size=64, scope='atten_4')
        #     # self.capsule_attn_4 = tf.reshape(self.capsule_attn_4, [-1, 16*32])
        #
        # with tf.name_scope("cnn_capsule_5"):
        #     # self.cnn = cnn_layer(self.w_embedded, config.embed_size, filter_size=5, num_filters=128, scope='cnn_5')
        #     self.capsule_encode = capsule_layer(self.cnn_5, out_caps_num=16, out_caps_dim=32, scope="capsule_5")
        #     self.capsule_rnn_5 = biRNNLayer(self.capsule_encode, hidden_size=50, scope="rnn_3", reuse=True)
        #     self.capsule_attn_5 = auxAttention(self.capsule_rnn_5, self.cnn_out, attention_size=64, scope='atten_5')

            # self.capsule_attn_5 = tf.reshape(self.capsule_attn_5, [-1, 16 * 32])


        # with tf.name_scope("atten_layer"):
        #     self.sent_encode = auxAttention(self.rnn_out, self.cnn_out, config.atten_size, scope='atten')

        # with tf.name_scope("atten_layer"):
        #     self.sent_encode = self_attention(self.rnn_out, attention_size=config.atten_size)
        #
        with tf.name_scope("capsule_layer"):
            self.capsule_encode = capsule_layer(self.w_embedded, 16, 64, scope="capsule")

        with tf.name_scope("rnn_layer"):
            self.rnn_out = biRNNLayer(self.capsule_encode, 50, scope='rnn_cap')

        with tf.name_scope("pool_layer"):
            self.max_out = max_pooling(self.rnn_out)
            self.avg_out = avg_pooling(self.rnn_out)
            self.rnn_cap_out = tf.concat([self.max_out, self.avg_out], axis=1)

        with tf.name_scope("rnn_layer"):
            self.rnn_out = biRNNLayer(self.w_embedded, 50, scope="rnn_word")

        with tf.name_scope("atten_layer"):
            self.sent_encode = auxAttention(self.rnn_out, self.rnn_cap_out, config.atten_size, scope='atten')

        #
        # with tf.name_scope("cnn_cap_layer"):
        #     self.cnn_cap_encode, _ = kim_cnn_layer(self.capsule_encode, embedding_size=64, filter_sizes=[3,4,5], num_filters=128, scope="cnn_caps")

        # with tf.name_scope("gate_layer"):
        #     self.sent_encode = gate_layer(self.cnn_out, self.cnn_cap_encode)

        with tf.name_scope("concat_layer"):
            #     self.sent_encode = tf.concat([self.capsule_attn_3, self.capsule_attn_4, self.capsule_attn_5], axis=1)
            #     self.out = fc_layer(self.sent_encode, 128, use_dropout=True, keep_prob=self.dropout_keep_prob,
            #                     scope='linear_layer_hidden')
            self.out = tf.concat([self.rnn_cap_out, self.sent_encode], axis=1)

        self.sent_encode = self.out
        with tf.name_scope("fc_layer"):
            self.out = fc_layer(self.sent_encode, config.num_class, use_dropout=True, keep_prob=self.dropout_keep_prob, scope='linear_layer_out')
            self.logit = tf.nn.softmax(self.out)

        with tf.name_scope("loss_and_pred"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logit,name="loss")
            self.loss = tf.reduce_sum(self.loss)
            self.prob = tf.argmax(self.logit, axis=1, name="prediction")

        # 优化部分
        # tvars = tf.trainable_variables()
        # grads, _= tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.grad_clip)
        optimizer = self.add_optimizer()
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars,global_step=self.global_step)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prob, tf.argmax(self.input_y, axis=1)), tf.float32))

        self.get_params()

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('ce_loss', self.loss)
        # tf.summary.scalar('opt_loss', self.opt_loss)

    def add_optimizer(self):
        lr = tf.train.exponential_decay(self.config.lr, self.global_step,
                                        self.config.decay_steps,
                                        self.config.decay_rate, staircase=True)
        self.learning_rate = tf.maximum(lr, 1e-5)
        if self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config.optimizer == 'grad':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adgrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        else:
            raise ValueError('No such Optimizer: %s' % self.config.optimizer)
        return optimizer

    def create_feed_dict(self, batch_data, dropout_keep_prob=1.0):
        batch_x, batch_wNum, batch_labels = batch_data[0], batch_data[1], batch_data[2]
        feed_dict = {
            self.input_x: batch_x,
            self.input_y: batch_labels,
            self.dropout_keep_prob: dropout_keep_prob
        }
        return feed_dict


    def get_params(self):
        var_list = [v for v in tf.trainable_variables()]
        # reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_var_list])
        self.param_cnt = np.sum([np.prod(v.get_shape().as_list()) for v in var_list])

        print("===" * 20)
        print('total reg parameter count: %.3f M' % (self.param_cnt / 1000000.))
        print('excluded variables from regularization')

        print("regularized variables")
        print(['%s:%.3fM' % (v.name, np.prod(v.get_shape().as_list()) / 1000000.) for v in var_list])
        print('===' * 20)