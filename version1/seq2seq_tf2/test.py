# --*-- coding: utf-8 --*--
# Created by WangShiYang at 3/20/20

import tensorflow as tf
from utils.gpu_utils import config_gpu
from seq2seq_tf2.seq2seq_model import Seq2Seq
from utils.params_utils import get_params
from utils.wv_loader import load_vocab, get_vocab
from utils.config import checkpoint_dir, test_data_path, checkpoint_prefix
from utils.data_loader import load_test_dataset
from seq2seq_tf2.test_helper import greedy_decode, beam_decode
import pandas as pd


def test(params):
    # Gpu资源的配置
    config_gpu()

    # 创建模型
    print("Building the model")
    model = Seq2Seq(params)

    # 创建词表
    # 这里务必注意：如果有多个返回值就一定要有多个变量进行接收，用不到的就用"_"来接收，否则的话得到的就不是相应的返回的变量，而是所有的变量
    # 进行len()之后得到的也是返回值的数量
    vocab, reverse_vocab = load_vocab(params["vocab_path"])
    print(len(vocab))

    # # 如果检查点存在，则恢复最新的检查点。
    # ckpt.restore(ckpt_manager.latest_checkpoint)
    # print("Model restored")

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")
    print('-------------------1--------------------')
    predict_result(model, params, vocab, reverse_vocab, params['result_save_path'])
    print('-------------------2--------------------')


def predict_result(model, params, vocab, reverse_vocab, result_save_path):
    test_X = load_test_dataset(params['max_enc_len'])
    # 预测结果
    results = greedy_decode(model, test_X, params['batch_size'], vocab, reverse_vocab, params)
    print(results)
    # 保存结果
    save_predict_result(results, result_save_path)


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(test_data_path)
    # 填充结果
    test_df['Prediction'] = results
    # 提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    # 获取参数
    params = get_params()

    test(params)
