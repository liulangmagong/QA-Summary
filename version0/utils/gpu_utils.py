# -*- coding:utf-8 -*-
# Created by WangShiYang at 3/17/20

import tensorflow as tf


def config_gpu(use_cpu):
    """
    RNN在跑并行的时候，它需要很大的GPU显存，所以跑的时候经常会报错或者跑着跑着就崩了，它这里下边是由这些选项，
    在跑模型的时候要配置一下下边的这些选项
    :return:
    """
    # 首先是确定是否使用GPU
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # gpu报错 使用cpu运行
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:  # 识别一下当前有几块GPU
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4930)])  # 限制GPU的显存使用
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

                # for gpu in gpus:
                #     tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)
