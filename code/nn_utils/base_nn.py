# coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

def biGRU(in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):

    with tf.variable_scope(scope or 'biGRU'):
        cell_fwd = tf.nn.rnn_cell.GRUCell(h_sz)
        cell_bwd = tf.nn.rnn_cell.GRUCell(h_sz)
        x_out, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fwd,
            cell_bw=cell_bwd,
            inputs=in_x,
            sequence_length=xLen,
            dtype=tf.float32,
            swap_memory=True,
            scope='birnn'
        )
        x_out = tf.concat(x_out, axis=2)
        if dropout is not None:
            x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
    return x_out

def biLSTM(in_x, xLen, h_sz, dropout=None, is_train=False, scope=None):

    with tf.variable_scope(scope or 'biLSTM'):
        cell_fwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
        cell_bwd = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
        x_out, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fwd,
            cell_bwd,
            in_x,
            xLen,
            dtype=tf.float32,
            swap_memory=True,
            scope='birnn'
        )
        x_out = tf.concat(x_out, axis=2)
        if dropout is not None:
            x_out = tf.layers.dropout(x_out, rate=dropout, training=is_train)
    return x_out

def masked_softmax(input, seqLen):
    seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)
    if len(input.get_shape()) != len(seqLen.get_shape()) + 1:
        raise ValueError('rank of seqLen should be %d, but have the rank %d.\n'
                         % (len(input.get_shape()-1), len(seqLen.get_shape())))
    mask = mkMask(seqLen, tf.shape(input)[-1])
    masked_input = tf.where(mask, input, tf.ones_like(input)* (-np.Inf))
    ret = tf.nn.softmax(masked_input)
    return ret

def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, maxLen])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)

def attention_mask(in_x, xLen, atten_sz, activation_fn=tf.tanh, dropout=None, is_train=False, scope=None):
    """

    :param in_x: shape(b_sz, tstp, dim)
    :param xLen: mask ---- shape(b_sz,)
    :param atten_sz:
    :param activation_fn:
    :param dropout:
    :param is_train:
    :param scope:
    :return:
    """

    assert len(in_x.get_shape()) == 3 and in_x.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        context_vector = tf.get_variable(name='context_vector', shape=[atten_sz],
                                         dtype=tf.float32)
        in_x_mlp = tf.layers.dense(in_x, atten_sz, activation=activation_fn, name='mlp')

        atten = tf.tensordot(in_x_mlp, context_vector, axes=[[2], [0]]) # shape(b_sz, tstp)
        atten_normed = masked_softmax(atten, xLen)

        atten_normed = tf.expand_dims(atten_normed, axis=-1)
        atten_ctx = tf.matmul(in_x, atten_normed, transpose_a=True) # shape(b_sz, dim, 1)
        atten_ctx = tf.squeeze(atten_ctx, axis=[2]) # shape(b_sz, dim)
        if dropout is not None:
            attn_ctx = tf.layers.dropout(atten_ctx, rate=dropout, training=is_train)
    return attn_ctx


def self_attention(inputs, attention_size):

    # shape of inputs: [batch, sequence_length, hidden_size]
    sequence_length = inputs.get_shape()[1].value
    hidden_size = inputs.get_shape()[2].value

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer= tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer= tf.truncated_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer= tf.truncated_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    output = inputs * tf.reshape(alphas, [-1, sequence_length, 1])
    return output

# auxiliary attention layer network
def auxAttention(input, aux, attention_size, scope=None):

    # shape of input: [batch, sequence_length, embedding_size]
    # shape of aux: [batch, vector_length]

    with tf.variable_scope(scope or 'aux_atten'):
        len_aux = int(aux.get_shape()[1])
        seq_input = int(input.get_shape()[1])
        emb_len_input = int(input.get_shape()[2])

        Wm_input = tf.Variable(tf.truncated_normal([emb_len_input, attention_size], stddev=0.1), name="Wm_input")
        Wm_aux = tf.Variable(tf.truncated_normal([len_aux, attention_size], stddev=0.1), name="Wm_aux")
        W_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_u")
        W_b = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_b")

        # extend auxiliary vector to matrix
        extend_aux = tf.expand_dims(aux, 1)
        matrix_aux = tf.tile(extend_aux, [1, seq_input, 1])
        reshape_aux = tf.reshape(matrix_aux, [-1, len_aux])

        # attention
        v = tf.matmul(tf.reshape(input, [-1, emb_len_input]), Wm_input) + tf.matmul(reshape_aux, Wm_aux) + tf.reshape(W_b, [1, -1])
        v = tf.tanh(v)
        vu = tf.matmul(v, tf.reshape(W_u, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, seq_input])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        output = input * tf.reshape(alphas, [-1, seq_input, 1])
        output = tf.reduce_sum(output, 1)
    return output



# CNN layer
# ===============================================


def fc_layer(input, output_size, use_bias=True, bias_start= 0., use_dropout=False, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        if use_dropout:
            input = tf.nn.dropout(input, keep_prob)
        input_size = input.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size,output_size], dtype=tf.float32)
        if use_bias:
            bias = tf.get_variable('bias', shape=[output_size], dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(input, W) + bias
        else:
            out = tf.matmul(input, W)
    return out

def gate_layer(input_a, input_b, use_bias=True, bias_start=0., use_dropout=False, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or 'gate_layer') as scope:
        output_size = input_a.get_shape()[-1]
        trans_a = fc_layer(input_a, output_size, use_bias=False, scope='trans_a')
        trans_b = fc_layer(input_b, output_size, use_bias=False, scope='trans_b')
        gate = tf.nn.sigmoid(trans_a + trans_b)
        out = gate * trans_a + (1-gate) * trans_b
        if use_dropout:
            out = tf.nn.dropout(out, keep_prob)
    return out

def cnn_layer(inputs, embedding_size, filter_size, num_filters, scope=None):
    with tf.variable_scope(scope or 'cnn_layer') as scope:
        inputs = tf.expand_dims(inputs, -1)
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b")
        conv = tf.nn.conv2d(
            inputs,
            W,
            strides=[1,1,1,1],
            padding="VALID",
            name="conv"
            )
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  #  tanh ---- Duyu Tang 2015
        conv = tf.squeeze(h, axis=2)
        return conv

def kim_cnn_layer(inputs, embedding_size, filter_sizes, num_filters, scope=None):
    with tf.variable_scope(scope or 'cnn_layer') as scope:
        pooled_outputs = []
        conv_outputs = []
        sequence_length = inputs.shape[1]
        inputs = tf.expand_dims(inputs, -1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b")
                conv = tf.nn.conv2d(
                    inputs,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  #  tanh ---- Duyu Tang 2015
                conv = tf.squeeze(h, axis=2)
                conv_outputs.append(conv)
                pooled = tf.nn.max_pool(  # avg_pool  --- Duyu Tang 2015
                    h,
                    ksize = [1, sequence_length - filter_size + 1, 1, 1],
                    strides = [1,1,1,1],
                    padding = "VALID",
                    name= "pool"
                )
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        return h_pool_flat, conv_outputs


# RNN layer
# ==================================================
# 返回一个序列中每个元素的长度
def get_length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    # abs 求绝对值,
    # reduce_max 求最大值, reduction_indices 在哪一个维度上求解
    # sign 返回符号-1 if x < 0 ; 0 if x == 0 ; 1 if x > 0
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    # 计算输入tensor元素的和,或者按照reduction_indices指定的轴进行
    return tf.cast(seq_len, tf.int32)
    # 将x的数据格式转化为int32

# bi-LSTM layer network
def biRNNLayer(inputs, hidden_size, scope=None, reuse=None):

    with tf.variable_scope(scope or 'biRNN',reuse=reuse) as scope:
        # fw_cell = rnn_cell.LSTMCell(hidden_size)
        # bw_cell = rnn_cell.LSTMCell(hidden_size)
        fw_cell = rnn.GRUCell(hidden_size)  # 前向GRU, 输入的参数为隐藏层的个数
        bw_cell = rnn.GRUCell(hidden_size)  # 后向GRU, 输入的参数为隐藏层的个数
        ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=inputs,
            sequence_length=get_length(inputs),
            dtype=tf.float32
        )
        # outputs的size是[batch_size, max_time, hidden_size *2 ]
        outputs = tf.concat((fw_outputs, bw_outputs), 2)  # 按行拼接
        return outputs

# max pooling of rnn out layer
def max_pooling(lstm_out):

    # shape of lstm_out: [batch, sequence_length, rnn_size * 2 ]
    # do max-pooling to change the (sequence_length) tensor to 1-length tensor
    sequence_length, hidden_cell_size = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(   # change to : tf.nn.average_pooling
        lstm_out,
        ksize=[1, sequence_length, 1, 1],
        strides=[1,1,1,1],
        padding='VALID'
    )
    output = tf.reshape(output, [-1, hidden_cell_size])
    return output

# max pooling of rnn out layer
def avg_pooling(lstm_out):

    # shape of lstm_out: [batch, sequence_length, rnn_size * 2 ]
    # do max-pooling to change the (sequence_length) tensor to 1-length tensor
    sequence_length, hidden_cell_size = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.avg_pool(   # change to : tf.nn.average_pooling
        lstm_out,
        ksize=[1, sequence_length, 1, 1],
        strides=[1,1,1,1],
        padding='VALID'
    )
    output = tf.reshape(output, [-1, hidden_cell_size])
    return output