from __future__ import absolute_import, division, print_function

import argparse
import glob # glob是Python标准库中的一个模块，用于查找符合特定规则的文件路径名
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler # DistributedSampler是PyTorch中的一个类，用于在分布式训练中对数据进行采样。在分布式训练中，每个进程只能看到部分数据，因此需要对数据进行划分和采样，以保证每个进程训练的数据不重复、不遗漏。

# 这段代码的作用是导入TensorBoard的可视化工具，用于可视化模型训练过程中的损失函数、准确率等指标。
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import sklearn.metrics as metrics  # 这段代码的作用是导入sklearn库中的metrics模块，用于计算模型的性能指标，如准确率、精确率、召回率、F1值等。
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer,
                          AlbertConfig,
                          AlbertForSequenceClassification,
                          AlbertTokenizer,
                          XLMRobertaConfig,
                          XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer,
                          )

from transformers import AdamW, get_linear_schedule_with_warmup # AdamW：一种基于梯度下降的优化器，可以用于微调预训练语言模型；get_linear_schedule_with_warmup：一种学习率调度器，可以在微调过程中逐渐降低学习率，以提高模型的性能。

# from transformers.data.metrics import  matthews_corrcoef
# from sklearn.metrics import matthews_corrcoef
from utils_tcm import TcmProcessor as processor
from utils_tcm import tcm_convert_examples_to_features as convert_examples_to_features
# from utils_tcm import store_preds_labels

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ()) # 得到所有模型的名称，并统计数量

from modelling_tcm import RobertaForTCMclassification,BertForTCMClassification # 自己修改了两个模型一个是BERT,一个是Roberta
MODEL_CLASSES = {
    'bert': (BertConfig, BertForTCMClassification, BertTokenizer), # 这两个是自己改了一下
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForTCMclassification, RobertaTokenizer), # 这两个是自己改了一下
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


MODEL_CLASSES = {
    'bert': (BertConfig, BertForTCMClassification, BertTokenizer), # 这两个是自己改了一下
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForTCMclassification, RobertaTokenizer), # 这两个是自己改了一下
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
tcm_bert_model = model_class.from_pretrained("/root/autodl-tmp/bert_kg_v1/output")
tcm_bert_tokenizer = tokenizer_class.from_pretrained("/root/autodl-tmp/bert_kg_v1/output")
tcm_bert_model.to("cuda")

tcm_bert_tokenizer = tokenizer_class.from_pretrained("/root/autodl-tmp/bert_kg_v1/output", do_lower_case=True) # ，如果你使用了默认的 tokenizer 名称，那么可以直接使用 from_pretrained() 函数加载 tokenizer，而不需要手动指定 tokenizer 的路径。



