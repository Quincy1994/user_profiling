# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from model.nn_layer import cnn_layer, biLSTM, biGRU, mkMask_softmax, get_length, avg_pooling, mask_attention
from model.nest import flatten
import pickle as pkl

class CNN():

    def __init__(self, config):
        self.cfg = config
        self.EX_REG_SCOPE = []
        self.on_epoch = tf.Variable(0, name='epoch_count', trainable=False)
        self.on_epoch_add = tf.assign_add(self.on_epoch, 1)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.build()

    def add_placeholders(self):
        self.ph_input = tf.placeholder(shape=(None, self.cfg.maxseq), dtype=tf.int32, name="ph_input")
        self.ph_labels = tf.placeholder(shape=(None, self.cfg.num_class), dtype=tf.int32, name="ph_labels")
        self.ph_train = tf.placeholder(dtype=tf.bool, name='ph_train')

    def add_embedding(self, prefix=''):

        """Customized function to transform x into embeddings"""
        with tf.variable_scope(prefix + 'embed'):
            if self.cfg.fix_emb:
                assert (hasattr(self.cfg, 'W_emb'))
                W_emb = pkl.load(open(self.cfg.W_emb_path, 'rb'))
                W = tf.get_variable('W', initializer= W_emb, trainable=True)
                print("iniitalize word embedding finished")
            else:
                weightInit = tf.random_uniform_initializer(-0.001, 0.001)
                vocab = pkl.load(open(self.cfg.vocab_path, 'rb'))
                W = tf.get_variable('W', [len(vocab), self.cfg.emb_size], initializer=weightInit,trainable=True)
            if hasattr(self.cfg, 'relu_w') and self.cfg.relu_w:
                W = tf.nn.relu(W)
        return W

    def build(self):
        self.add_placeholders()
        W = self.add_embedding()
        user_word_emb = tf.nn.embedding_lookup(W, self.ph_input, name="user_word_emb") # (batch, maxsent, maxword, word_dim)
        sent_emb = self.sent_encode(user_word_emb) # (b_sz * maxsent, hidden_size *2 )or (b_sz * maxsent, len(filter_sizes) * filter_number)

        doc_res = sent_emb

        with tf.variable_scope("classifier"):
            logits = self.Dense(doc_res, dropout=self.cfg.dropout, is_train=self.ph_train, activation=None)
            self.loss = self.add_loss_op(logits)
            self.train_op =  self.add_train_op(self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('loss', self.loss)


    def sent_encode(self, user_word_emb, scope=None):
        with tf.variable_scope(scope or "sent_encode"):
            sent_emb = cnn_layer(user_word_emb, self.cfg.emb_size, filter_sizes=self.cfg.filter_sizes, num_filters=self.cfg.num_filters)
            return sent_emb

    def Dense(self, inputs, dropout=None, is_train=False, activation=None):
        loop_input = inputs
        if self.cfg.dense_hidden[-1] != self.cfg.num_class:
            raise ValueError("last hidden layer should be %d, but get %d" %
                             (self.cfg.num_class, self.cfg.dense_hidden[-1]))
        for i, hid_num in enumerate(self.cfg.dense_hidden):
            with tf.variable_scope('dense-layer-%d' %i):
                loop_input = tf.layers.dense(loop_input, units=hid_num)

            if i < len(self.cfg.dense_hidden) - 1:
                if dropout is not None:
                    loop_input = tf.layers.dropout(loop_input, rate=dropout, training=is_train)
                if activation is not None:
                    loop_input = activation(loop_input)
        logits = tf.nn.softmax(loop_input)
        return logits

    def add_loss_op(self, logits):

        with tf.name_scope("loss"):
            self.prediction = tf.argmax(logits, axis=1, name="prediction")
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.ph_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float", name="accuracy"))

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)
            losses = tf.reduce_mean(losses)

            exclude_vars = flatten([[v for v in tf.trainable_variables(o.name)]for o in self.EX_REG_SCOPE])
            exclude_vars_2  = [ v for v in tf.trainable_variables() if '/bias:' in v.name]
            exclude_vars = exclude_vars + exclude_vars_2

            reg_var_list = [v for v in tf.trainable_variables() if v not in exclude_vars]
            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_var_list])
            self.param_cnt = np.sum([np.prod(v.get_shape().as_list()) for v in reg_var_list])

            print("===" * 20)
            print('total reg parameter count: %.3f M' % (self.param_cnt / 1000000.))
            print('excluded variable from regularization')
            print([v.name for v in exclude_vars])
            print('===' * 20)

            print('regularized variables')
            print(['%s: %3.fM' % (v.name, np.prod(v.get_shape().as_list())/ 1000000.) for v in reg_var_list])
            print("===" * 20)

            return losses

    def add_train_op(self, loss):
        lr = tf.train.exponential_decay(self.cfg.lr, self.global_step,
                                        self.cfg.decay_steps,
                                        self.cfg.decay_rate, staircase=True)
        self.learning_rate = tf.maximum(lr, 1e-5)
        if self.cfg.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.cfg.optimizer == 'grad':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.cfg.optimizer == 'adgrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.cfg.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        else:
            raise ValueError('No such Optimizer: %s' % (self.cfg.optimizer))

        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.cfg.grad_clip)
        # train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        return train_op




