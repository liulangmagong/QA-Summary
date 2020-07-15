# --*-- coding: utf-8 --*--
# Created by WangShiYang at 3/13/20

import tensorflow as tf
from seq2seq_tf2.batcher import train_batch_generator
from seq2seq_tf2.seq2seq_model import Seq2Seq
from utils.config import save_wv_model_path
from utils.gpu_utils import config_gpu
from utils.wv_loader import get_vocab
import time


def train_model(model, vocab, params, checkpoint_manager):
    """

    :param model:
    :param vocab:
    :param params:
    :param checkpoint_manager:
    :return:
    """
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    pad_index = vocab['<PAD>']
    nuk_index = vocab['<UNK>']
    start_index = vocab['<START>']

    # 计算vocab size
    params['vocab_size'] = len(vocab)
    # 对于参数文件中定义好的参数，可以在使用的时候进行更改，就像是运行的时候传入的参数，就不会再使用参数文件中的参数值

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        # 掩盖住<PAD><NUK>这一部分值，不计算它们的损失
        pad_mask = tf.math.equal(real, pad_index)
        nuk_mask = tf.math.equal(real, nuk_index)
        mask = tf.math.logical_not(tf.math.logical_or(pad_mask, nuk_mask))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # 训练
    @tf.function
    def train_step(enc_inp, dec_target):
        """

        :param enc_inp:
        :param dec_target:
        :return:
        """
        batch_loss = 0
        with tf.GradientTape() as tape:
            # 这里的model就是定义的Seq2Seq模型
            # 拿到encoder层的输出，准备构建decoder的第一次输入
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            print('enc_hidden is ', enc_hidden)
            print('enc_output ', enc_output)

            # 训练的时候，第一次的输入，比较特殊，通常拿出来单独处理一下
            # 第一个decoder输入  开始标签<START>
            dec_input = tf.expand_dims([start_index] * batch_size, 1)
            # 第一个隐藏层输入
            dec_hidden = enc_hidden
            # 逐个预测序列
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)
            # 因为一次预测的是一个batch，预测一次，进行一次损失计算
            batch_loss = loss_function(dec_target[:, 1:], predictions)
            # [:, 1:]: 即取所有数据的第1到最后一列数据（含左不含右）

            # 获取需要进行更新的参数
            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables

            # 对参数进行梯度更新
            gradients = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss
            # 到这里，一个batch的训练过程就结束了，接下来就是将所有的batch循环训练epoch轮的过程

    # 通过traintrain_batch_generator，一次获取一个batch的数据，并且返回需要多少步才能训练完所有的batch
    dataset, steps_per_epoch = train_batch_generator(batch_size)

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            # 这个for循环的作用就是：每循环一次就获得一个batch数量的训练集和标签集
            # dataset.take(steps_per_epoch): 创建一个包含steps_per_epoch个元素的数据集，一个batch大小的数据集
           # 这里之所以使用(batch, (inputs, target))来接收数据是((32, 200)(32, 41)),一次一个batch数量的训练集和标签集
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每两个epoch保存一下模型(checkpoint)
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
