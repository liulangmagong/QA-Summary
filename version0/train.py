# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/14/20

import os
import time
import tensorflow as tf


def train(dataset, steps_per_epoch, model, vocab, params, checkpoint_manager):

    pad_index = vocab['<PAD>']

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, pad_index))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(enc_inp, dec_target):
        batch_loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([vocab['<START>']] * params['batch_size'], 1)
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)
            batch_loss = loss_function(dec_target[:, 1:], predictions)

            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables

            gradients = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    for epoch in range(params["epochs"]):
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
