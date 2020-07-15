# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/14/20

import os
import time
import numpy as np
import tensorflow as tf

from utils.data_loader import build_dataset, load_dataset
from utils.config import train_data_path, test_data_path, vocab_path
from utils.wv_loader import load_vocab, load_embedding_matrix
from seq2seq_tf2.model_layers import Encoder, BahdanauAttention, Decoder
from utils.gpu_utils import config_gpu
from utils.params_utils import get_params
from seq2seq_tf2.seq2seq_model import Seq2Seq
from utils.config import checkpoint_dir
from seq2seq_tf2.batcher import train_batch_generator
import test
import train

from utils.data_loader import preprocess_sentence


if __name__ == '__main__':

    config_gpu(use_cpu=False)

    # 数据预处理  程序第一次运行的时候使用，其他的时候不用再多次运行
    # build_dataset(train_data_path=train_data_path, test_data_path=test_data_path)

    vocab, reverse_vocab = load_vocab(vocab_path)

    embedding_matrix = load_embedding_matrix()

    # 设置构建模型需要的参数
    params = {}
    params['batch_size'] = 32
    params['vocab_size'] = len(vocab)
    params['embed_size'] = 500
    params['units'] = 512
    params["enc_units"] = 512
    params["attn_units"] = 512
    params["dec_units"] = 512
    params["epochs"] = 10

    # 构建训练集
    # dataset, steps_per_epoch = train_batch_generator(params['batch_size'], sample_sum=640)
    dataset, steps_per_epoch = train_batch_generator(params['batch_size'])

    # 构建模型
    print('开始构建模型...')
    model = Seq2Seq(params)

    # 保存点设置
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # 训练模型
    train.train(dataset, steps_per_epoch, model, vocab, params, checkpoint_manager)
    print('训练完成')

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    sentence = '漏机油 具体 部位 发动机 变速器 正中间 位置 拍 中间 上面 上 已经 看见'

    test.test(sentence, vocab, reverse_vocab, model)