# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/15/20

import os
import pathlib
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

from utils.gpu_utils import config_gpu
from utils.data_loader import build_dataset, load_dataset
from utils.config import train_data_path, test_data_path, vocab_path
from utils.wv_loader import load_vocab, load_embedding_matrix
from seq2seq_no_Attention.test import test


def seq2seq(input_length, output_sequence_length, embedding_matrix, vocab_size):
    model = Sequential()
    # 添加一个embedding层
    # (batch_size, input_length) => (batch_size, input_length, output_dim)
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=500,  # 要和预训练好的向量维度相匹配，embedding_size=500
                        weights=[embedding_matrix],
                        trainable=False,
                        input_length=input_length))
    # 添加一个双向GRU层
    model.add(Bidirectional(GRU(units=512, return_sequences=False, return_state=False)))
    # 添加一个Dense层
    model.add(Dense(units=vocab_size, activation='relu'))  # (batch_size, input_dim) => (batch_size, units)
    model.add(RepeatVector(output_sequence_length))  # (num_samples, features) => (num_samples, n, features)

    # 添加一个双向的GRU
    model.add(Bidirectional(GRU(units=512, return_sequences=True)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(1e-3))

    model.summary()

    return model


if __name__ == '__main__':
    print('使用的TensorFlow的版本：', tf.__version__)

    # GPU资源配置
    config_gpu(use_cpu=False)

    # 数据预处理  程序第一次运行的时候使用，其他的时候不用再多次运行
    # build_dataset(train_data_path=train_data_path, test_data_path=test_data_path)

    # 加载数据
    train_X, train_Y, test_X = load_dataset()
    print('训练集输入模型的数据：\n{} \n 训练集数据的shape：\n {}'.format(train_X, train_X.shape))
    """模型输入数据形式：
        [[31816   415   903 ... 31818 31818 31818]
         [31816   813 31819 ... 31818 31818 31818]
         [31816  1393    88 ...  3321  6567  2232]
         ...
         [31816   225   894 ... 31818 31818 31818]
         [31816 12684  3145 ... 31818 31818 31818]
         [31816  3275    75 ...   409     1     3]]
        训练集数据： (82873, 200)
    """

    # 加载 Vocab
    vocab, reverse_vocab = load_vocab(vocab_path)
    print('词表数据展示：\n{} \n 词表大小： {}'.format(vocab, len(vocab)))

    # 加载预训练的词向量矩阵--embedding_matrix
    embedding_matrix = load_embedding_matrix()
    print('训练好的词向量矩阵：\n{} \n 词向量的大小：{}'.format(embedding_matrix, embedding_matrix.shape))

    # 设置采用多少数据进行训练
    sample_num = 640
    train_X = train_X[:sample_num]
    train_Y = train_Y[:sample_num]

    # 设置构建模型需要的参数
    # 使用训练数据的数量
    BUFFER_SIZE = len(train_X)
    # 输入的长度   x  max_len
    input_length = train_X.shape[1]
    # 输出的长度  y  max_len
    output_sequence_length = train_Y.shape[1]
    # 词表大小
    vocab_size = len(vocab)
    # 词向量矩阵
    embedding_matrix = embedding_matrix
    # batch_size
    BATCH_SIZE = 16

    # 构建训练集
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # 模型搭建--盖楼
    model = seq2seq(input_length, output_sequence_length, embedding_matrix, vocab_size)

    # 模型训练
    model.fit(train_X, train_Y, batch_size=16, epochs=3, validation_split=0.2)

    # 模型保存
    model.save('model_save.h5')

    del model

    model = tf.keras.models.load_model('model_save.h5')
    print('模型已经导入')




