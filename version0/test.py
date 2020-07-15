# -*- coding: UTF-8 -*-
# Created by WangShiYang at 07/14/20

import numpy as np
import tensorflow as tf
from utils.data_loader import preprocess_sentence
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker
from seq2seq_tf2.model_layers import Decoder, BahdanauAttention, Encoder

# 解决中文乱码
font = font_manager.FontProperties(fname="data/TrueType/simhei.ttf")
encoder = Encoder


def evaluate(sentence, vocab, reverse_vocab, model, max_length_inp=200, max_length_targ=50):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    inputs = preprocess_sentence(sentence, max_length_inp, vocab)
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, 512))]

    enc_output, enc_hidden = model.encoder(inputs,
                                           hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([vocab['<START>']], 0)

    for t in range(max_length_targ):
        context_vector, attention_weights = model.attention(dec_hidden, enc_output)
        predictions, dec_hidden = model.decoder(dec_input,
                                                dec_hidden,
                                                enc_output,
                                                context_vector)

        attention_weights = tf.reshape(attention_weights, (-1, ))

        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += reverse_vocab[predicted_id] + ' '
        if reverse_vocab[predicted_id] == '<STOP>':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14, 'fontproperties': font}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def test(sentence, vocab, reverse_vocab, model):
    result, sentence, attention_plot = evaluate(sentence, vocab, reverse_vocab, model)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))
