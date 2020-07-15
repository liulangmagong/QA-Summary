# -*- coding:utf-8 -*-
# Created by WangShiYang at 3/17/20

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize(lang):
    lang_tokenizer = Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer
