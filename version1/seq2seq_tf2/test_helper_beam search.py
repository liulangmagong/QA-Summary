# --*-- coding: utf-8 --*--
# Created by WangShiYang at 3/13/20

import tensorflow as tf


class Hypothesis:
    """
    要进行Beam搜索的话，每一层的中间状态我们要保存下来，这样的话最好就是我们先定义一个类，
    来存储我们每一个时间步的中间结果。这里边包含：
    tokens: 每个（当前）时间步的输入
    log_probs： 使用log算一下概率
    hidden, attn_dists： 隐藏层和Attention的权重，这两个要保存下来才能进行下一步的操作

    假设对象序列可以这样理解：
    在sequence2sequence模型中，beam search的方法只用在测试的情况，因为在训练过程中，
    每一个decoder的输出是有正确答案的，也就不需要beam search去加大输出的准确率。

    测试的时候，假设词表大小为3，内容为a, b, c. beam size是2
    decoder解码的时候：

    1： 生成第1个词的时候，选择概率最大的2个词，假设为a,c,那么当前序列就是a,c

    2：生成第2个词的时候，我们将当前序列a和c，分别与词表中的所有词进行组合，得
    到新的6个序列: aa ab ac ca cb cc,然后从其中选择2个得分最高的，作为当前
    序列，假如为: aa cb

    3：后面会不断重复这个过程，直到遇到结束符为止。最终输出2个得分最高的序列。
    这里的序列是会一直保存着的，到最后边之所以选择两个得分最高的序列是因为，这里
    的出来的分也是在不断的累加的。
    """
    def __init__(self, tokens, log_probs, hidden, attn_dist):
        self.tokens = tokens  # 从第0个时间步到当前时间步的所有token的列表
        self.log_probs = log_probs  # 所有的token的log(概率)组成的列表
        self.hidden = hidden  # 上一个token编码后的decoder隐藏层
        self.attn_dists = attn_dist  # 所有token的注意力列表
        self.abstract = ""

    def extend(self, token, log_prob, hidden, attn_dists):
        return Hypothesis(tokens=self.tokens + [token],   # 添加解码后的token
                          log_probs=self.log_probs + [log_prob],  # 添加解码后token的log(概率)
                          hidden=hidden,  # 更新状态
                          attn_dists = self.attn_dists + [attn_dists])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        # tot:total
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def beam_decode(model, batch, vocab, params):
    # 初始化mask
    start_index = vocab['<START>']
    stop_index = vocab['<STOP>']

    batch_size = params['batch_size']

    # 单步decoder
    def decoder_onestep(enc_output, dec_input, dec_hidden):
        # 单个时间步运行
        preds, dec_hidden, context_vector, attention_weights = model.call_decoder_onestep(dec_input,
                                                                                          dec_hidden,
                                                                                          enc_output)
        # 拿到top k个index 和概率
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(preds), k=params["beam_size"])
        # 计算log概率
        top_k_log_probs = tf.math.log(top_k_probs)
        # 返回需要保存的中间结果和概率
        return preds, dec_hidden, context_vector, attention_weights,top_k_log_probs, top_k_ids

    # 计算encoder的输出
    enc_output, enc_hidden = model.call_encoder(batch)
    # 初始化batch_size个假设对象
    # 为一个batch里边的每一句话初始化一个假设对象列表，用于存储中间结果和概率
    hyps = [Hypothesis(tokens=[start_index],
                       log_probs=[0, 0],
                       hidden=enc_hidden[0],
                       attn_dist=[]) for _ in range(batch_size)]

    # 初始化结果集
    results = []  # 列表保存顶部的beam_size假设
    # 遍历步数
    steps = 0  # 初始化步数

    # 第一个decoder输入 开始标签：<START>
    dec_input = tf.expand_dims([start_index] * batch_size, 1)
    # 第一个隐藏层的输入
    dec_hidden = enc_hidden

    # 长度不够 并且 结果还不够，继续搜索
    while steps < params['max_dec_stpes'] and len(results) < params['beam_size']:
        # 获取最新待使用token
        latest_tokens = [h.latest_token for h in hyps]
        # 获取所有隐藏层的状态
        hiddens = [h.hidden for h in hyps]
        # 单步运行decoder 计算需要的值
        preds, dec_hidden, context_vector, attention_weights, top_k_log_probs, top_k_ids = decoder_onestep(enc_output,
                                                                                                           dec_input,
                                                                                                           dec_hidden)
        # 现阶段所有可能情况
        all_hyps = []
        # 原有的可能情况数量
        num_orig_hyps = 1 if steps == 0 else len(hyps)

        # 遍历所有可能的结果
        for i in range(num_orig_hyps):
            h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
            # 分裂 添加beam size 种可能性
            for j in range(params['beam_size']):
                # 构造可能的情况
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=top_k_log_probs[i, j],
                                   hidden=new_hidden,
                                   attn_dists=attn_dist)
                # 添加可能情况
                all_hyps.append(new_hyp)

        # 重置
        hyps = []
        # 按照概率来排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)

        # 筛选top 前beam_size句话
        for h in sorted_hyps:
            if h.latest_token == start_index:
                # 长度符合预期，遇到句尾，添加到结果集
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # 未到结束，添加到假设集
                hyps.append(h)

            # 如果假设的句子正好等于beam_size 或者结果集正好等于beam_size 就不再添加
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break
        steps += 1
    if len(results) == 0:
        results = hyps

    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]
    # best_hyp.abstract = " ".join([reverse_vocab[index] for index in best_hyp.tokens])
    return best_hyp






