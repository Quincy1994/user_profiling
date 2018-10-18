# coding=utf
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

def embedding(features, cfg, prefix=''):

    """Customized function to transform x into embeddings"""
    with tf.variable_scope(prefix + 'embed'):
        if cfg.fix_emb:
            assert (hasattr(cfg, 'W_emb'))
            assert (np.shape(np.array(cfg.W_emb)) == (cfg.n_words, cfg.embed_size))
            W = tf.get_variable('W', initializer=cfg.W_emb, trainable=True)
            print("iniitalize word embedding finished")
        else:
            weightInit = tf.random_uniform_initializer(-0.001, 0.001)
            W = tf.get_variable('W', [cfg.n_words, cfg.embed_size], initializer=weightInit)
        if hasattr(cfg, 'relu_w') and cfg.relu_w:
            W = tf.nn.relu(W)

    word_vectors = tf.nn.embedding_lookup(W, features)
    return word_vectors, W

def linear_layer(x, output_dim, prefix):
    input_dim = x.get_shape().as_list()[1]
    thres = np.sqrt(6.0/ (input_dim + output_dim))
    W = tf.get_variable(name=prefix+ "_W", shape=[input_dim, output_dim], initializer=tf.random_uniform_initializer(minval=thres, maxval=thres))
    b = tf.get_variable(name=prefix + "_b", shape=[output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, W) + b

def mkMask_softmax(sent, seqLen):
    """
    sent : shape(b_sz, maxsent, topicNum)
    seqLen; shape(b_sz,)
    """
    b_sz, maxsent, topicNum = tf.unstack(tf.shape(sent))
    mask = tf.cast(tf.sequence_mask(seqLen, maxlen=maxsent), tf.float32)
    for _ in range(len(sent.shape) - 2):
        mask = tf.expand_dims(mask, 2)
    mask = tf.tile(mask, [1, 1, topicNum])
    sent = sent - (1.0 - mask) * 1e12
    sent = tf.exp(sent)
    sent = tf.reshape(sent, [-1, topicNum]) # (batch* sent, topciNum)
    sent = sent / tf.reshape(tf.add(tf.reduce_sum(sent, 1), 0.01), [-1, 1]) # (batch* sent, topciNum)
    sent = tf.reshape(sent, [-1, maxsent, topicNum]) # (batch, maxsent, topicNum)
    return sent

# def topicAttention(input_sentence_embedding, topic_input_embed, seqLen, is_trans, prefix):
#
#     """
#     topic_input_embedding: (topic_number, topic_embed)
#     sentence_input_embedding: (b_sz, maxsen, sent_emb)
#
#     Note:  len(sent_emb) == len(sent_emb)
#
#     :return:
#         topic_prob: (b_sz, topicNum)
#         sent_topic: (b_sz, maxsen, topic_emb)
#     """
#
#     b_sz, maxsen, sen_embed = tf.unstack(tf.shape(input_sentence_embedding))
#     topic_num, topic_dim = tf.unstack(tf.shape(topic_input_embed))
#
#     # whether transform vector
#     if is_trans:
#         biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
#         topic_input = layers.fully_connected(topic_input_embed, num_outputs=topic_dim, biases_initializer=biasInit, activation_fn=tf.nn.tanh, scope=prefix + "_bias")
#     else:
#         topic_input = topic_input_embed
#
#     # calculating Vtopic * Usent
#     input_sentence_embedding = tf.reshape(input_sentence_embedding, [-1, sen_embed])  # (b_sz * maxsen, sen_emb)
#     topic_extend = tf.expand_dims(topic_input, 0) # (1, topic_number, topic_embedding_size)
#     multi_topic_sent = tf.multiply(topic_extend, tf.expand_dims(input_sentence_embedding, 1)) # (1, topic_number, topic_emb) * (b_sz * maxsen, 1, sen_emb) = (b_sz*maxsen, topic_number, topic_emb * sen_emb)
#     multi_topic_sent = tf.reduce_sum(multi_topic_sent,2) # (b_sz*maxsen, topic_num)
#     multi_topic_sent = tf.reshape(multi_topic_sent, [-1, maxsen, topic_num]) # (b_sz, maxsen, topicNum)
#     multi_topic_softmax = mkMask_softmax(multi_topic_sent, seqLen) # (b_sz, maxsent, topicNum) -- softmax
#
#     # get total prob distribution
#     topic_prob = tf.reduce_mean(multi_topic_softmax, axis=1)  # total topic prob distribution (b_sz, topicNum)
#
#     # get sent topic vector representation
#     alpha = tf.reshape(multi_topic_softmax, [-1, topic_num]) # (b_sz * maxsen, topic_num)
#     alpha_extend = tf.expand_dims(alpha, 2) # (b_sz * maxsen, topic_num, 1)
#     topic_input_extend = tf.expand_dims(topic_input_embed, 0) # (1, topic_num, topic_emb)
#     sent_topic = tf.reduce_sum(tf.multiply(alpha_extend, topic_input_extend), axis=1) # (b_sz*maxsen, topic_emb)
#     sent_topic = tf.reshape(sent_topic, [-1, maxsen, topic_dim]) # (b_sz, maxsen, topic_emb)
#     return topic_prob, sent_topic

def mask_attention(inputs, sequence_length, dim, attn_size, seqLen):

    # Attention mechansim
    W_omega = tf.get_variable("W_omega", initializer= tf.truncated_normal([dim, attn_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer= tf.truncated_normal([attn_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer= tf.truncated_normal([attn_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, dim]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    vu = tf.reshape(vu, [-1, sequence_length])
    vu = tf.expand_dims(vu, -1)
    mask = tf.cast(tf.sequence_mask(seqLen, maxlen=sequence_length), tf.float32)
    for _ in range(len(inputs.shape) - 2):
        mask = tf.expand_dims(mask, 2)
    vu = vu - (1-mask) * 1e12
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.add(tf.reduce_sum(exps, 1), 0.001), [-1, 1])
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    return output

def biGRU(inputs, hidden_size, seqLen):

    fw_cell = rnn.GRUCell(hidden_size)
    bw_cell = rnn.GRUCell(hidden_size)
    ((fw_outputs, bw_outputs), (_,_)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=inputs,
        sequence_length=seqLen,
        dtype=tf.float32
    )
    outputs = tf.concat((fw_outputs, bw_outputs), 2)
    return outputs

def biLSTM(inputs, hidden_size, seqLen):
    fw_cell = rnn.LSTMCell(hidden_size)
    bw_cell = rnn.LSTMCell(hidden_size)
    ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=inputs,
        sequence_length=seqLen,
        dtype=tf.float32
    )
    outputs = tf.concat((fw_outputs, bw_outputs), 2)
    return outputs

def max_pooling(rnn_out):
    sequence_length, hidden_cell_size = int(rnn_out.get_shape()[1], int(rnn_out.get_shape()[2]))
    rnn_out = tf.expand_dims(rnn_out, -1)
    output = tf.nn.max_pool(
        rnn_out,
        ksize=[1, sequence_length, 1,1],
        strides = [1,1,1,1],
        padding='VALID'
    )
    output = tf.reshape(output, [-1, hidden_cell_size])
    return output

def avg_pooling(rnn_out, sequence_length, hidden_cell_size ):
    rnn_out = tf.expand_dims(rnn_out, -1)
    output = tf.nn.avg_pool(
        rnn_out,
        ksize=[1, sequence_length, 1,1],
        strides=[1,1,1,1],
        padding='VALID'
    )
    output = tf.reshape(output, [-1, hidden_cell_size])
    return output

def cnn_layer(inputs, embedding_size, filter_sizes, num_filters):
    pooled_outputs = []
    sequence_length = inputs.shape[1]
    inputs = tf.expand_dims(inputs, -1)
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % (filter_size)):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_w")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b")
            conv = tf.nn.conv2d(
                inputs,
                W,
                strides=[1,1,1,1],
                padding="VALID",
                name="conv"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize = [1, sequence_length - filter_size + 1, 1, 1],
                strides = [1,1,1,1],
                padding = "VALID",
                name="pool"
            )
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    return h_pool_flat

def get_length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    seq_len = tf.cast(seq_len, tf.int32)
    return seq_len


