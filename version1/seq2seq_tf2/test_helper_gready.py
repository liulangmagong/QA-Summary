# -*- coding:utf-8 -*-
# Created by WangShiYang at 3/20/20

import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm


def greedy_decode(model, data_X, batch_size, vocab, reverse_vocab, params):
    # 存储结果
    results = []
    # 样本数量
    sample_size = len(data_X)
    # batch 操作轮数   math.ceil: 向上取整   最后有小数 +1，去掉小数
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = math.ceil(sample_size / batch_size)

    # [0,steps_epoch]
    for i in tqdm(range(steps_epoch)):
        # 一个batch一个batch的取
        batch_data = data_X[i * batch_size:(i + 1) * batch_size]
        results += batch_greedy_decode(model, batch_data, vocab, reverse_vocab, params)
    return results


def batch_greedy_decode(model, batch_data, vocab, reverse_vocab, params):
    """
    批量预测
    :param model: Seq2Seq模型
    :param batch_data:  测试集中的一个batch数据
    :param vocab: 词表
    :param params: 定义的参数
    :return:
    """
    # 判断输入的长度
    batch_size = len(batch_data)
    # 开辟存储结果的list
    predicts = [''] * batch_size
    inps = tf.convert_to_tensor(batch_data)
    # 0.初始化隐藏层输入
    hidden = [tf.zeros((batch_size, params['enc_units']))]
    # 1.构建encoder
    enc_output, enc_hidden = model.encoder(inps, hidden)
    # 2.复制
    dec_hidden = enc_hidden
    # <START> * batch_size
    dec_input = tf.expand_dims([vocab['<START>']] * batch_size, 1)

    context_vector, _ = model.attention(dec_hidden, enc_output)

    for t in range(params['max_dec_len']):
        # 计算上下文
        context_vector, attention_weights = model.attention(dec_hidden, enc_output)
        # 单步预测
        predictions, dec_hidden = model.decoder(dec_input,
                                                dec_hidden,
                                                enc_output,
                                                context_vector)

        # id转换  贪婪搜索
        predicted_ids = tf.argmax(predictions, axis=1).numpy()

        for index, predicted_id in enumerate(predicted_ids):
            # 这样写的话，要是预测的id不vocab里边的话怎么办
            predicts[index] += reverse_vocab[predicted_id] + ' '

        # useing teacher forcing
        dec_input = tf.expand_dims(predicted_ids, 1)

    results = []
    # 对预测得到的句子进行处理
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 把结果中，！。去掉，把所有的空格删掉
        predict = predict.lstrip(' ，！。')
        predict = predict.replace(' ', '')
        if predict == '':
            predict = '随时联系'
        # 句子小于max_len就结束了  截断
        if '<STOP>' in predict:
            # 截断stop
            predict = predict[:predict.index('<STOP>')]
        # 保存结果
        results.append(predict)
    return results





