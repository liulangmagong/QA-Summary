# -*- coding:utf-8 -*-
# Created by WangShiYang at 3/17/20

import argparse
from utils.config import vocab_size
from utils.config import *
from utils.file_utils import get_result_filename


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_enc_len", default=200, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=41, help="Decoder input max sequence length", type=int)
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--epochs", default=10, help="train epochs", type=int)
    parser.add_argument("--vocab_size", default=vocab_size, help="batch size", type=int)

    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--embed_size", default=500, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=512, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=512, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=512, help="[context vector, decoder state, decoder input] feedforward \
                            result dimension - this result is used to compute the attention weights",
                        type=int)

    parser.add_argument("--learning_rate", default=1e4, help="Learning rate", type=float)
    parser.add_argument("--checkpoints_save_steps", default=5, help="Save checkpoints every N steps", type=int)

    parser.add_argument("--vocab_path", default=vocab_path, help="vocab path", type=str)
    parser.add_argument("--save_wv_model_path", default=save_wv_model_path, help="save_wv_model_path", type=str)
    parser.add_argument("--result_save_path", default=get_result_filename(batch_size, epochs, 200, 300),
                        help='result_save_path', type=str)

    args = parser.parse_args()
    params = vars(args)

    return params


if __name__ == '__main__':
    print(get_params())