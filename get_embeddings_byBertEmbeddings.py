from __future__ import absolute_import, division, print_function

import argparse
import glob # glob是Python标准库中的一个模块，用于查找符合特定规则的文件路径名
import logging
import os
import random
import json
import gc
import tqdm

import torch.nn as nn

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler # DistributedSampler是PyTorch中的一个类，用于在分布式训练中对数据进行采样。在分布式训练中，每个进程只能看到部分数据，因此需要对数据进行划分和采样，以保证每个进程训练的数据不重复、不遗漏。

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
from transformers.modeling_bert import BertEmbeddings
# from transformers.data.metrics import  matthews_corrcoef
# from sklearn.metrics import matthews_corrcoef
from utils_tcm import TcmProcessor as processor
from utils_tcm import tcm_convert_examples_to_features as convert_examples_to_features
# from utils_tcm import store_preds_labels

from modelling_tcm import BertForTCMClassification # 自己修改了两个模型一个是BERT,一个是Roberta
MODEL_CLASSES = {
    'bert': (BertConfig, BertForTCMClassification, BertTokenizer), # 这两个是自己改了一下
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


MODEL_CLASSES = {
    'bert': (BertConfig, BertForTCMClassification, BertTokenizer), # 这两个是自己改了一下
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

config_name = "/root/autodl-tmp/ZY-BERT"
cache_dir = "/root/autodl-tmp/bert_kg_v1"
model_name_or_path = "/root/autodl-tmp/ZY-BERT"
tokenizer_name = "/root/autodl-tmp/ZY-BERT"
do_lower_case = True

herb_dictionary_path = "/root/autodl-tmp/get_herb_embeddings/herb_dictionary.json"

if __name__ == '__main__':

    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    tcm_bert_config = config_class.from_pretrained(config_name if config_name else model_name_or_path,
                                            # num_labels=128,
                                            # finetuning_task=task_name,
                                            cache_dir=cache_dir if cache_dir else None)
    tcm_bert_tokenizer = tokenizer_class.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir if cache_dir else None)
    
    # 加载数据
    herb_list = []
    with open(herb_dictionary_path, 'r') as f:
        herb_dict = json.load(f)
    for herb, count in herb_dict.items():
        if count >= 12 and herb:
            herb_list.append(herb)
    herb_list = sorted(herb_list)
    with open('/root/autodl-tmp/get_herb_embeddings/herb_dict.json', 'w', encoding='utf-8') as f:
        json.dump(herb_list, f, ensure_ascii=False, indent=2)

    max_herb_len = max(len(herb) for herb in herb_list)

    inputs = []
    for herb in herb_list:
        input = tcm_bert_tokenizer.encode_plus(herb, add_special_tokens=False, max_length=max_herb_len)
        input_ids = input['input_ids']
        token_type_ids = input['token_type_ids']
        attention_mask = input['attention_mask']

        input_ids = input_ids + (max_herb_len-len(input_ids))*[0]
        token_type_ids = token_type_ids + (max_herb_len-len(token_type_ids))*[0]
        attention_mask = attention_mask + (max_herb_len-len(attention_mask))*[0]

        input['input_ids'] = torch.tensor(input_ids).unsqueeze(0).to('cuda')
        input['token_type_ids'] = torch.tensor(token_type_ids).unsqueeze(0).to('cuda')
        input['attention_mask'] = torch.tensor(attention_mask).unsqueeze(0).to('cuda')

        inputs.append(input)
    
    bertEmbedding = BertEmbeddings(tcm_bert_config).to('cuda')
    dropout = nn.Dropout(tcm_bert_config.hidden_dropout_prob)
    
    all_herb_embeddings = []
    for input in tqdm(inputs, desc="processing>>>"):
        
        output = bertEmbedding(input['input_ids'])
        attention_mask = input['attention_mask'].unsqueeze(2)
        masked_output =  output * attention_mask
        
        # 平均池化
        sum_output = torch.sum(masked_output, dim=1)
        mean_pooled_output = sum_output/attention_mask.sum(dim=1)

        result = dropout(mean_pooled_output)

        all_herb_embeddings.append(result)
        print()




    all_herb_embeddings = torch.cat(all_herb_embeddings, dim=0)

    torch.save(all_herb_embeddings, '/root/autodl-tmp/get_herb_embeddings/BertEmbeddings_embeddings')

    


    
    