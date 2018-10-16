# coding=utf-8

import os,sys,time
import numpy as np
import tensorflow as tf

from my_utils import utils
from my_utils.TFUtils import entry_stop_gradients
from my_utils.data_preprocess import get_vocab
from nn_utils.base_nn import biGRU, biLSTM, cnn_layer, fc_layer, auxAttention, gate_layer, kim_cnn_layer
from nn_utils.capsule_layer import capsule_layer
from nn_utils.loss import margin_loss, cross_entroy
from nn_utils import nest

class DeliModel(object):

    def __init__(self, config):

        self.config = config
        self.EX_REG_SCOPE = []

        self.on_epoch = tf.Variable(0, name='epoch_count', trainable=False)
        self.on_epoch_accu = tf.assign_add(self.on_epoch, 1)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)


        # self.net_1 = tf.constant(value=1, dtype=tf.int32)
        # self.net_2 = tf.constant(value=2, dtype=tf.int32)

        self.build()

    def add_placeholders(self):

        # shape(b_sz, wNum)
        self.ph_input = tf.placeholder(shape=(None, self.config.word_seq_maxlen), dtype=tf.int32, name='ph_input')
        # shape(b_sz)
        self.ph_labels = tf.placeholder(shape=(None, self.config.num_class), dtype=tf.int32, name='ph_labels')
        # shape(b_sz): 用于做masking
        self.ph_wNum = tf.placeholder(shape=(None,), dtype=tf.int32, name='ph_wNum')
        # 是否trainable
        self.ph_train = tf.placeholder(dtype=tf.bool, name='ph_train')
        # 网络状态
        self.network_state = tf.placeholder(dtype=tf.int32, name='network_state')
        # keep_prob
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    #
    # def create_feed_dict(self, data_batch, train):
    #     """ data_batch: input, labels, wNum, example_weight"""
    #     phs = (self.ph_input, self.ph_labels, self.ph_wNum, self.ph_sample_weights, self.ph_train)
    #     feed_dict = dict(zip(phs, data_batch+(train, )))
    #     return feed_dict

    def add_embedding(self):

        vocab = get_vocab()
        vocab_sz = len(vocab)  # 词表的大小
        with tf.variable_scope('embedding') as scp:
            self.exclude_reg_scope(scp) # 是否需要正则
            if self.config.pre_trained:
                embed = utils.readEmbedding(self.config.embed_path)
                embed_matrix, valid_mask = utils.mkEmbedMatrix(embed, dict(self.config.vocab_dict))
                embedding = tf.Variable(embed_matrix, trainable=False, name='Embedding')
            else:
                embed_matrix = tf.random_uniform([vocab_sz, self.config.embed_size], -0.25, 0.25)
                embed_matrix[0] = np.zeros([self.config.embed_size], np.int32)
                embedding = tf.Variable(embed_matrix, name="Embedding", trainable=True)


        return embedding

    def exclude_reg_scope(self, scope):
        if scope not in self.EX_REG_SCOPE:
            self.EX_REG_SCOPE.append(scope)

    def embd_lookup(self, embedding, batch_x, dropout=None, is_train=False):
        """

        :param embedding: shape(v_sz, emb_sz)
        :param batch_x:  shape(b_sz, wNum)
        :param dropout:
        :param is_train:
        :return:
            shape(b_sz, wNUm, emb_sz)
        """
        inputs = tf.nn.embedding_lookup(embedding, batch_x)
        if dropout is not None:
            inputs = tf.layers.dropout(inputs, rate=dropout, training=is_train)
        return inputs

    def rnn_layer(self, in_x, wNum, scope=None):
        with tf.variable_scope(scope or 'encoding_birnn'):

            with tf.variable_scope('snt_enc'):
                if self.config.seq_encoder == 'bigru':
                    birnn_wd = biGRU(in_x, wNum, self.config.hidden_size, scope='biGRU')
                elif self.config.seq_encoder == 'bilstm':
                    birnn_wd = biLSTM(in_x, wNum, self.config.hidden_size, scope='biLSTM')
                else:
                    raise ValueError('no such encode %s' % self.config.seq_encoder)
        return birnn_wd

    def create_feed_dict(self, batch_data, net_state=1, train=True, dropout_keep_prob=1.0):
        batch_x, batch_wNum, batch_labels = batch_data[0], batch_data[1], batch_data[2]
        feed_dict = {
            self.ph_input : batch_x,
            self.ph_wNum: batch_wNum,
            self.ph_labels: batch_labels,
            self.ph_train: train,
            self.network_state: net_state,
            self.dropout_keep_prob: dropout_keep_prob
        }
        return feed_dict

    def build(self):

        self.add_placeholders()
        self.embedding = self.add_embedding() # 注意是否静态
        self.in_x = self.embd_lookup(self.embedding, self.ph_input)

        # self.cnn_input = cnn_layer(self.in_x, self.config.embed_size, 3, self.config.num_filters, scope="cnn_layer")
        # self.capsule = capsule_layer(self.cnn_input, self.config.out_caps_num, self.config.out_caps_dim, scope="capsule_layer")

        # 第一个网络, 纯CNN+capsule
        if self.config.net_state == 1:
            print('hello 1')
            self.capsule_flatten = tf.reshape(self.capsule, shape=[-1, self.config.out_caps_num * self.config.out_caps_dim])
            self.out = fc_layer(self.capsule_flatten, self.config.num_class, use_dropout=True, keep_prob=self.dropout_keep_prob)
            self.logit = tf.nn.softmax(self.out)
            self.loss = margin_loss(self.ph_labels, self.logit)

        # 第二个网络, capsule融入到atten当中, 然后再来一个gate
        elif self.config.net_state == 2:
            self.capsule_flatten = tf.reshape(self.capsule, shape=[-1, self.config.out_caps_num * self.config.out_caps_dim])
            self.birnn = self.rnn_layer(self.in_x, self.ph_wNum, scope="birnn")

            # 先对capsule做个fc层
            self.capsule_fc = fc_layer(self.capsule_flatten, self.config.hidden_size * 2, scope="capsule_fc")
            # 做辅助attention
            self.atten_out = auxAttention(self.birnn, self.capsule_flatten, self.config.atten_size, scope="aux_atten")
            # 做一层gate
            self.gate_out = gate_layer(self.capsule_fc, self.atten_out, use_dropout=False, keep_prob=self.dropout_keep_prob)
            # 全连接层
            self.fc_out = fc_layer(self.gate_out, self.config.num_class, use_dropout=True, keep_prob=self.dropout_keep_prob)
            self.logit = tf.nn.softmax(self.fc_out)
            self.loss = cross_entroy(self.ph_labels, self.logit)

        # cnn_kim
        elif self.config.net_state == 3:
            self.cnn_input = kim_cnn_layer(self.in_x, self.config.embed_size, filter_sizes=[3,4,5], num_filters=self.config.num_filters)
            self.fc_out = fc_layer(self.cnn_input, self.config.num_class, use_dropout=True,
                                   keep_prob=self.dropout_keep_prob)
            self.logit = tf.nn.softmax(self.fc_out)
            self.loss = cross_entroy(self.ph_labels, self.logit)
            # print(tf.shape(self.logit))
            # print(tf.shape(self.ph_labels))


        else:
            raise ValueError('network state should be (1,2), but get %d' % (
                self.network_state
            ))


        opt_loss = self.add_loss_op(self.loss, self.logit, self.ph_labels)
        train_op = self.add_train_op(opt_loss)

        self.train_op = train_op
        self.opt_loss = opt_loss
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('ce_loss', self.loss)
        tf.summary.scalar('opt_loss', self.opt_loss)



    def add_loss_op(self, loss, logits, labels):
        """
        :param loss:
        :param logits: shape(b_sz, c_num)
        :param labels: shape(b_sz, c_num)
        :return:
        """
        self.y_pred = tf.argmax(logits, axis=1, name="prediction")
        labels = tf.argmax(labels, axis=1, name="label")
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.y_pred, labels), tf.float32))

        # 增加l2正则
        exclude_vars = nest.flatten([v for v in tf.trainable_variables(o.name)] for o in self.EX_REG_SCOPE)
        exclude_vars_2 = [v for v in tf.trainable_variables() if '/bias:' in v.name]
        exclude_vars = exclude_vars + exclude_vars_2

        reg_var_list = [v for v in tf.trainable_variables() if v not in exclude_vars]
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_var_list])
        self.param_cnt = np.sum([np.prod(v.get_shape().as_list()) for v in reg_var_list])

        print("===" * 20)
        print('total reg parameter count: %.3f M' % (self.param_cnt / 1000000.))
        print('excluded variables from regularization')
        # print(exclude_vars)
        # if exclude_vars is not None:
        #     print([v.name for v in exclude_vars])
        #     print("===" * 20)

        print("regularized variables")
        print(['%s:%.3fM' % (v.name, np.prod(v.get_shape().as_list()) / 1000000.) for v in reg_var_list])
        print('===' * 20)
        reg = self.config.reg

        return loss + reg * reg_loss


    def add_train_op(self, loss):

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

        # tvars = tf.trainable_variables()
        # # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        # # grads_and_vars = tuple(zip(grads, tvars))
        # grads_and_vars = optimizer.compute_gradients(loss, tvars)
        # # capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in grads_and_vars]
        # train_op = optimizer.apply_gradients(grads_and_vars , global_step=self.global_step)

        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        return train_op





























