# -*- coding:utf-8 -*-
# Created by WangShiYang at 3/17/20

import tensorflow as tf


def config_gpu():
    """
    RNN在跑并行的时候，它需要很大的GPU显存，所以跑的时候经常会报错或者跑着跑着就崩了，它这里下边是由这些选项，
    在跑模型的时候要配置一下下边的这些选项
    :return:
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # 设置GPU的memory为可以增长的模型
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
