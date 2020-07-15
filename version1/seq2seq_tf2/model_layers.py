# --*-- coding: utf-8 --*--
# Created by WangShiYang at 3/12/20

from utils.config import embedding_matrix_path, save_wv_model_path
from utils.gpu_utils import config_gpu
from utils.wv_loader import get_vocab, load_word2vec_file
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_sz):
        """
        构建Encoder  因为是构建框架，所以这个初始化函数没有返回值
        :param vocab_size: 词表的大小 总共有多少个词
        :param embedding_dim: 每一个词使用多少维的向量
        :param embedding_matrix: 预处理好的词向量矩阵    这个参数可以不设置，让模型自己训练，比较耗时
        :param enc_units: encoder的单元个数
        :param batch_sz: 一次传入的batch的大小  即一次传入多少个句子
        """
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],  # 预训练的词向量
                                                   trainable=False)  # 设置模型不进行词向量训练
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        """
        return_sequences:布尔类型参数。决定是返回输出序列中的最后一个时间步的输出，还是返回完整序列(包含每一个时间步的输出)
                            输出维度是：[32, 200, 256]。默认值:False, 若是False的话，返回的就是最后一个时间步的输出，
                            维度是：[32, 256]。
        return_state: 布尔类型参数。是否返回输出（上边整个序列的输出）之外的最后一个时间步的状态,
                          这里由于还需要传入最后一个时间步的隐藏层状态到decoder层，所以要将最后一个时间步的隐藏层状态返回出来。
                          默认值:False。
        """

    def call(self, x, hidden):
        """
        Encoder层的前向传播的过程--Forward
        :param x: Encoder层的输入
        :param hidden: GRU层的隐藏层状态
        :return: 返回Encoder层的输出，GRU层的隐藏层状态的输出
        """
        # 输入数据经过embedding层
        x = self.embedding(x)
        # embedding层的输出进入gru层，得到Encoder层的输出和GRU层隐藏层状态的输出
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """
        初始化编码过程中GRU隐藏层状态
        :return: 编码过程中GRU隐藏层的初始化状态
        """
        return tf.zeros(shape=(self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        """
        搭建注意力层的框架
        :param units: 注意力层的单元个数，通常和Encoder层，Decoder层相同
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """
        注意力层的前向传播的过程
        :param query: 上一个时间步的编码过程中GRU隐藏层状态的输出
        :param values: 编码器(Encoder)的编码结果输出(enc_output)
        :return: 上下文向量：context_vector, 注意力权重：attention_weights
        """
        # 在seq2seq模型中，St是后面的query向量，而编码过程的隐藏状态hi是values。？？

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # 这样做是为了执行加法以计算分数
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 分数的形状 == （批大小，最大长度，1）
        # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V  ?
        # 在应用 self.V 之前，张量的形状是（批大小，最大长度，units）
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # axis=1: 在max_len的维度上，计算的是每一个词的注意力权重
        # 最大长度 （max_length） 是我们的输入的长度。因为我们想为每个输入分配一个权重，所以softmax应该用在这个轴上。
        # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
        attention_weights = tf.nn.softmax(score, axis=1)

        # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_sz):
        """
        构建Decoder框架
        :param vocab_size: 词表的大小
        :param embedding_dim: 使用多少维的向量表示一个词
        :param embedding_matrix: 预训练好的词向量
        :param dec_units:  Decoder层的单元数
        :param batch_sz: 一个batch的大小
        """
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        """
        return_sequences: 布尔类型参数。决定是返回输出序列中的最后一个时间步的输出，是返回完整序列(包含每一个时间步的输出)。
                            默认值:False。
        return_state: 布尔类型参数。是否返回输出（上边整个序列的输出）之外的最后一个时间步的状态。
                            默认值:False。
        """
        # self.fc = tf.keras.layers.Dropout(0.5)
        self.fc = tf.keras.layers.Dense(vocab_size)
        # self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, x, hidden, enc_output, context_vector):
        """
        Decoder层的前向传播过程，一个实现预测的过程
        :param x: Decoder层的输入
        :param hidden: Decoder中上一个时间步的隐藏层状态的输出
        :param enc_output:
        :param context_vector:
        :return:
        """
        # 使用上一个时间步的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # 将上一时间步的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 将合并后的向量传送到 GRU
        output, state = self.gru(x)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        prediction = self.fc(output)

        return prediction, state


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab, revers_vocab = get_vocab(save_wv_model_path)
    # 计算vocab size
    vocab_size = len(vocab)
    # 使用Gensim训练好的embedding matrix
    embedding_matrix = load_word2vec_file(save_wv_model_path)

    input_sequence_len = 250
    BATCH_SIZE = 64
    embedding_dim = 500
    units = 1024

    # 编码器结构
    encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    # example_input
    example_input_batch = tf.ones(shape=(BATCH_SIZE, input_sequence_len), dtype=tf.int32)
    # sample input
    sample_hidden = encoder.initialize_hidden_state()

    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    sample_decoder_output, _, = decoder(tf.random.uniform((64, 1)),
                                        sample_hidden, sample_output, attention_result)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
