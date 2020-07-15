AutoMaster_TrainSet 和 AutoMaster_TestSet 拷贝到data 路径下 再使用 .


代码结构
+ data  数据集
    + AutoMaster_TrainSet 拷贝数据集到该路径
    + AutoMaster_TestSet  拷贝数据集到该路径
    + 停用词
    + checkpoint 模型存储
    + word2vec
    	+ 预训练词向量矩阵
    	+ 数据预处理之后得到的词表vocab/reverse vocab
    	+ Word2vec模型文件
    	+ 模型训练的参数

    ....
+ result 结果保存路径
    ....    
+ seq2seq_tf2 模型结构
    + test-基于beam_search
    + test-基于贪心的模型
    ....
+ utils 工具包
    + config  配置文件
    + data_loader 数据处理模块
    + multi_proc_utils 多进程数据处理

    
    

训练步骤:
1. 拷贝数据集到data路径下
2. 运行utils\data_loader.py可以一键完成 预处理数据 构建数据集
3. 训练模型 运行seq2seq_tf2\train.py脚本 或者 05_seq2seq-Train.ipynb 或者在终端运行project_by_myself\train.py均可以完成训练
4. 测试 参照05_seq2seq-Train.ipynb中有现成代码,包括结果生成
5. 结果提交,参考5_2研讨课提交流程,线上提交验证结果.

* 05_seq2seq-Train.ipynb 线上得分27.8分
* beam search 代码在seq2seq_tf2\test_helper.py中 ,自行参考实现.
* GPU运行速度, embedding_dim_300 enc_max_len=200 units=512 500s/epochs (8W训练集) 
  

可以将每一次修改的改进都写到ReadMe里边
提交结果:

1. score 19.3103
第一次提交拿的是上一节课的代码，跑了一下，gru_units是1024
> score 19.3103 gru 1024 ,embedding_dim 300 batch_size=16 Epoch=10 训练时间2700s,去除标点符号

2. score 22.537
改变了输入长度，batch_size 提升了一下，loss在1左右，分数达到二十多分
> loss 1.03  

> batch_size_64_epochs_10_max_length_inp_299_embedding_dim_300

3. score 23.1084
在训练的过程中改变学习率，再就是之前训练词向量的时候没有使用skip gram，这里加进去之后又提升了1.几分，说明我们预训练的词向量对我们的模型是有一定的影响的
skip gram + learning rate epochs_4 1e-5 -> epochs_1 1e-4

> 2019_12_06_18_19_33_batch_size_64_epochs_4_max_length_inp_299_embedding_dim_300.csv


4. score 27.9942   
这里发现，。等标点符号影响会很大，于是把常见的标点符号又加了进去
> loss 1.0

learning rate 1e-4 优化数据预处理 添加标点 去除`[]`,优化切词  max_len将近达到400，但是400机器跑不起来，所以这里从前边取了200个词来做，就有27分了，要是把整个句子都加进去的话会得到一个更高的分数

> 2019_12_07_12_10_34_batch_size_32_epochs_4_max_length_inp_200_embedding_dim_300


5. score 28.4136
将05_seq2seq-Train里边的3训练里边的额loss_function做了修改，将<PAD><UNK>这两个符号的loss都给忽略掉，发现也有一定的提升
> 2019_12_07_15_43_36_batch_size_32_epochs_10_max_length_inp_200_embedding_dim_300_4_1_submit_proc_add_masks_loss.csv


6. score 27.965

> 2019_12_07_17_56_34_batch_size_32_epochs_10_max_length_inp_200_embedding_dim_500_4_1_submit_proc_add_masks_loss.csv

loss 0.7~0.9