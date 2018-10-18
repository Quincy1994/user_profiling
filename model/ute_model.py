# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from model.nn_layer import cnn_layer, biLSTM, biGRU, mkMask_softmax, get_length, avg_pooling, mask_attention
from model.nest import flatten
import pickle as pkl

class UTE():

    def __init__(self, config):
        self.cfg = config
        self.EX_REG_SCOPE = []
        self.on_epoch = tf.Variable(0, name='epoch_count', trainable=False)
        self.on_epoch_add = tf.assign_add(self.on_epoch, 1)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.build()

    def add_placeholders(self):
        self.ph_input = tf.placeholder(shape=(None, self.cfg.maxsent, self.cfg.maxword), dtype=tf.int32, name="ph_input")
        self.ph_labels = tf.placeholder(shape=(None, self.cfg.num_class), dtype=tf.int32, name="ph_labels")
        self.ph_sNum = tf.placeholder(shape=(None,), dtype=tf.int32, name="ph_sNum")
        self.ph_wNum = tf.placeholder(shape=(None, None), dtype=tf.int32, name="ph_wNum")
        self.ph_topic = tf.placeholder(shape=(self.cfg.topic_num, self.cfg.topic_words), dtype=tf.int32, name="ph_topic")
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
                W = tf.get_variable('W', [len(vocab), self.cfg.emb_size], initializer=weightInit)
            if hasattr(self.cfg, 'relu_w') and self.cfg.relu_w:
                W = tf.nn.relu(W)
        return W

    def get_sent_dim(self):
        if self.cfg.ute_sent_encode == "cnn":
            hidden_dim = len(self.cfg.filter_sizes) * self.cfg.num_filters
        elif self.cfg.ute_sent_encode == "bigru" or self.cfg.ute_sent_encode == 'bilstm':
            hidden_dim = self.cfg.word_hidden * 2
        else:
            raise ValueError("no such sent encode %s" %(self.cfg.ute_sent_encode))
        return hidden_dim

    def build(self):
        self.add_placeholders()
        W = self.add_embedding()
        user_word_emb = tf.nn.embedding_lookup(W, self.ph_input, name="user_word_emb") # (batch, maxsent, maxword, word_dim)
        topic_word_emb = tf.nn.embedding_lookup(W, self.ph_topic, name="topic_word_emb") # (batch, topic_num, topic_word, word_dim)
        topic_emb = self.topic_embedding(topic_word_emb) # (topic_num, topic_dim)
        sent_emb = self.sent_encode(user_word_emb) # (b_sz * maxsent, hidden_size *2 )or (b_sz * maxsent, len(filter_sizes) * filter_number)

        # trans sent_emb
        hidden_dim = self.get_sent_dim()
        sent_emb = tf.reshape(sent_emb, [-1, self.cfg.maxsent, hidden_dim]) # shape: (b_sz, maxsen, hidden_emb)

        # topic_prob: (b_sz, topicNum)
        # sent_topic: (b_sz, maxsen, topic_emb)
        topic_prob, sent_topic = self.topicAttention(sent_emb, topic_emb, self.ph_sNum)
        sent_topic_emb = tf.concat([sent_emb, sent_topic], axis=2)
        sent_topic_emb = tf.reshape(sent_topic_emb, [-1, self.cfg.maxsent, hidden_dim + self.cfg.word_hidden * 2])
        doc_emb = self.doc_embbedding(sent_topic_emb, self.ph_sNum) # shape:(b_sz, sent_hidden * 2)
        doc_res = tf.concat([doc_emb, topic_prob], axis=1) # shape: (b_sz, sent_hidden * 2 + topic_num)

        with tf.variable_scope("classifier"):
            doc_res = tf.reshape(doc_res, [-1, self.cfg.sent_hidden *2+self.cfg.topic_num])
            logits = self.Dense(doc_res, dropout=self.cfg.dropout, is_train=self.ph_train, activation=None)
            self.loss = self.add_loss_op(logits)
            self.train_op =  self.add_train_op(self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('loss', self.loss)


    def doc_embbedding(self, sent_topic_emb, seqLen, scope=None):
        with tf.variable_scope(scope or "sent_topic_embedding"):
            if self.cfg.ute_doc_encode == "bigru":
                birnn_sent = biGRU(sent_topic_emb, self.cfg.sent_hidden, self.ph_sNum)
            elif self.cfg.ute_doc_encode == 'bilstm':
                birnn_sent = biLSTM(sent_topic_emb, self.cfg.sent_hidden, self.ph_sNum)
            else:
                raise ValueError("no such encoder %s" %(self.cfg.doc_encode))
            doc_emb = mask_attention(birnn_sent, self.cfg.maxsent, self.cfg.sent_hidden * 2, self.cfg.atten_size, seqLen) # (b_sz, sent_hidden * 2)
            return doc_emb

    def topic_embedding(self, topic_word_emb, scope=None):
            # topic_word_emb = tf.expand_dims(topic_word_emb, axis=0)
            with tf.variable_scope(scope or "topic_encode"):
                if self.cfg.topic_encode == "bigru":
                    topic_seqLen = get_length(topic_word_emb)
                    topic_seqLen = tf.reshape(topic_seqLen, [-1, ])
                    birnn_x = biGRU(topic_word_emb, self.cfg.word_hidden, topic_seqLen)
                elif self.cfg.topic_encode == 'bilstm':
                    topic_seqLen = get_length(topic_word_emb)
                    topic_seqLen = tf.reshape(topic_seqLen, [-1, ])
                    birnn_x = biLSTM(topic_word_emb, self.cfg.word_hidden, topic_seqLen)
                else:
                    raise ValueError("no such encoder %s" %(self.cfg.rnn_encode))
            topic_emb = avg_pooling(birnn_x, self.cfg.topic_num, self.cfg.word_hidden * 2) # shape: (1, topic, word_hidden)
            # topic_emb = mask_attention(birnn_x, self.cfg.topic_num, self.cfg.word_hidden * 2, self.cfg.atten_size, topic_seqLen) # sequence_length, dim, attn_size, seqLen
            topic_emb = tf.reshape(topic_emb, [self.cfg.topic_num, self.cfg.word_hidden * 2]) # shape: (topic_num, word_hidden)
            return topic_emb

    def sent_encode(self, user_word_emb, scope=None):
        # b_sz, ststp, wtstp, emb_sz = tf.unstack(tf.shape(user_word_emb))
        with tf.variable_scope(scope or "sent_encode"):
            sent_word = tf.reshape(user_word_emb, [-1, self.cfg.maxword, self.cfg.emb_size])
            sent_wNUm = tf.reshape(self.ph_wNum, [-1, ])

            if self.cfg.ute_sent_encode == "cnn":
                sent_emb = cnn_layer(sent_word, self.cfg.emb_size, filter_sizes=self.cfg.filter_sizes, num_filters=self.cfg.num_filters)
            elif self.cfg.ute_sent_encode == 'bigru':
                sent_emb = biGRU(sent_word,self.cfg.word_hidden, sent_wNUm)
                sent_emb = mask_attention(sent_emb, self.cfg.maxword, self.cfg.word_hidden * 2, self.cfg.atten_size, sent_wNUm)
            elif self.cfg.ute_sent_encode == 'bilstm':
                sent_emb = biLSTM(sent_word, self.cfg.word_hidden, sent_wNUm)
                sent_emb = mask_attention(sent_emb, self.cfg.maxword, self.cfg.word_hidden * 2, self.cfg.atten_size, sent_wNUm)
            else:
                raise ValueError("no such sent encode %s" % (self.cfg.ute_sent_encode))
            return sent_emb



    def topicAttention(self, input_sentence_embedding, topic_input_embed, seqLen, scope=None):

        """
        topic_input_embedding: (topic_number, topic_embed)
        sentence_input_embedding: (b_sz, maxsen, sent_emb)

        Note:  len(sent_emb) == len(sent_emb)

        :return:
            topic_prob: (b_sz, topicNum)
            sent_topic: (b_sz, maxsen, topic_emb)
        """
        with tf.variable_scope(scope or "topic_attention"):
            topic_num, topic_dim = tf.unstack(tf.shape(topic_input_embed))

            if self.cfg.ute_sent_encode == "cnn":
                hidden_dim = len(self.cfg.filter_sizes) *  self.cfg.num_filters
                self.cfg.is_trans = True
            elif self.cfg.ute_sent_encode == "bigru" or self.cfg.ute_sent_encode == 'bilstm':
                hidden_dim = self.cfg.word_hidden * 2
            else:
                raise ValueError(" no such sent encode %s"%(self.cfg.ute_sent_encode))
            # whether transform vector
            if self.cfg.is_trans:
                biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
                topic_input = layers.fully_connected(topic_input_embed, num_outputs=hidden_dim, biases_initializer=biasInit,
                                                     activation_fn=None, scope="_bias")
            else:
                topic_input = topic_input_embed

            # calculating Vtopic * Usent
            input_sentence_embedding = tf.reshape(input_sentence_embedding, [-1, hidden_dim])  # (b_sz * maxsen, sen_emb)
            topic_extend = tf.expand_dims(topic_input, 0)  # (1, topic_number, topic_embedding_size)
            multi_topic_sent = tf.multiply(topic_extend, tf.expand_dims(input_sentence_embedding,
                                                                        1))  # (1, topic_number, topic_emb) * (b_sz * maxsen, 1, sen_emb) = (b_sz*maxsen, topic_number, topic_emb * sen_emb)
            multi_topic_sent = tf.reduce_sum(multi_topic_sent, 2)  # (b_sz*maxsen, topic_num)
            multi_topic_sent = tf.reshape(multi_topic_sent, [-1, self.cfg.maxsent, topic_num])  # (b_sz, maxsen, topicNum)
            multi_topic_softmax = mkMask_softmax(multi_topic_sent, seqLen)  # (b_sz, maxsent, topicNum) -- softmax

            with tf.name_scope("topic"):
                # get total prob distribution
                topic_prob = tf.reduce_mean(multi_topic_softmax, axis=1,name="topic_prob")  # total topic prob distribution (b_sz, topicNum)

                # get sent topic vector representation
                alpha = tf.reshape(multi_topic_softmax, [-1, topic_num], name="sent_topic_prob")  # (b_sz * maxsen, topic_num)
                alpha_extend = tf.expand_dims(alpha, 2)  # (b_sz * maxsen, topic_num, 1)
                topic_input_extend = tf.expand_dims(topic_input, 0)  # (1, topic_num, topic_emb)
                sent_topic = tf.reduce_sum(tf.multiply(alpha_extend, topic_input_extend), axis=1)  # (b_sz*maxsen, topic_emb)
                sent_topic = tf.reshape(sent_topic, [-1, self.cfg.maxsent, topic_dim], name="topic_sent")  # (b_sz, maxsen, topic_emb)
            return topic_prob, sent_topic

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
            self.prediction = tf.argmax(logits, axis=-1, name="prediction")
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

        tvars = tf.trainable_variables()
        grads, _ =tf.clip_by_global_norm(tf.gradients(loss, tvars), self.cfg.grad_clip)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op




