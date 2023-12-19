# -*- coding：utf-8 -*-

from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import copy
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer
import sklearn.metrics as metrics
logger = logging.getLogger(__name__)
import xlrd
import random

def read_knowledge_base(pathtoknowledge="/root/autodl-tmp/ZY-BERT-main/to_LJW/syndrome_knowledge.json"):
    with open(pathtoknowledge, 'r', encoding='utf-8') as wb:
            lines = wb.readlines()
    
    instances = []
    for line in lines:
        data_raw = json.loads(line.strip('\n'))
        Name = data_raw['Name']
        Definition = data_raw['Definition']
        Typical_performance = data_raw['Typical_performance']
        Common_isease = data_raw['Common_isease']

        instance = {
            "Name":Name,
            "Definition":Definition,
            "Typical_performance":Typical_performance,
            "Common_isease":Common_isease,
        }
    
        instances.append(instance)

    unique_instances = [dict(t) for t in {tuple(d.items()) for d in instances}]
    kb_in_dict={}
    for syndrome in unique_instances:
        kb_in_dict[syndrome['Name']]=syndrome
    
    return kb_in_dict

    '''
    wb = xlrd.open_workbook(filename=pathtoknowledge)
    sheet_1 = wb.sheet_by_name("Sheet1")
    instances = []
    # all_description=[]
    for line_no in range(1,2139):
        Name = sheet_1.cell(line_no, 4).value
        Definition = sheet_1.cell(line_no, 5).value
        # Definition = sheet_1.cell(line_no, 5).value.replace(Name,'').replace("中医病证名","").replace("中医病名",'').replace("，。",'').replace("，，",'')
        Typical_performance = sheet_1.cell(line_no, 6).value
        Common_isease  =sheet_1.cell(line_no, 7).value

        instance = {
            "Name":Name,
            "Definition": Definition,
            "Typical_performance": Typical_performance,
            "Common_isease": Common_isease,
        }

        instances.append(instance)
    unique_instances = [dict(t) for t in {tuple(d.items()) for d in instances}]
    kb_in_dict={}
    for syndrome in unique_instances:
        kb_in_dict[syndrome['Name']]=syndrome
    
    return kb_in_dict
    '''
    

## jiahao.geng
##def read_knowledge

class InputExample(object):
    """A single training/test example"""
    def __init__(self, prescription_id, symptoms, individual_characteris, herb_collection):
        '''
        :param prescription_id: 处方的ID
        :param symptoms:     症状集合    
        :param individual_characteris:  个体特征
        :param herb_collection: 中药集合
        '''
        self.prescription_id = prescription_id
        self.symptoms = symptoms
        self.individual_characteris = individual_characteris
        self.herb_collection = herb_collection


class InputFeatures(object):
    """

    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir,args=None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir,args=None):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir,args=None):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()



class TcmProcessor(DataProcessor):
    """Processor for the TCM data set."""
    def get_train_examples_jiahao(self, data_dir,args=None):
        """See base class."""

        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples_jiahao(self._read_json(os.path.join(data_dir, "train.json")))

    def get_train_examples(self, data_dir,args=None):
        """See base class."""

        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")))

    def get_dev_examples(self, data_dir,args=None):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")))

    def get_test_examples(self, data_dir, args=None):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")))

    def get_labels(self):
        """See base class."""
        labels = []
        # /home/mcren/TCM/data_preprocess/syndrome_vocab.txt
        # E:\Project-TCM\TCM\data_preprocess\syndrome_vocab.txt
        # jiahao.geng: /root/autodl-tmp/ZY-BERT-main/TCM/data_preprocess/syndrome_vocab.txt
        with open("/root/autodl-tmp/ZY-BERT-jiahao.geng/TCM/data_preprocess/syndrome_vocab.txt", 'r',encoding='utf-8') as f:
            for line in f:
                labels.append(line.strip('\n'))
        return labels

    def get_herb_labels(self):
        herb_labels = []
        with open('/root/autodl-tmp/bert_kg/kg-embeddings/src_data/custom_graph/entity_relation/中药_entity_dict.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split('\t')
                herb_labels.append(line[1])
        return herb_labels
    

    def _get_kb(selfs,data_dir):
        kb = read_knowledge_base(os.path.join(data_dir,"syndrome_knowledge.json"))
        return kb
    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            return lines
        
    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            # initialize current valid knowledge

            data_raw = json.loads(line.strip('\n')) # 用于将 JSON 格式的字符串转换为 Python 对象，strip('\n') 函数用于去除字符串末尾的换行符，以避免解析出错。

            prescription_id = data_raw['prescription_id']
            symptoms = data_raw['symptoms']
            individual_characteris = data_raw['individual_characteris']
            herb_collection = data_raw['herb_collection']


            examples.append(InputExample(
                prescription_id=prescription_id,
                symptoms=symptoms,
                individual_characteris=individual_characteris,
                herb_collection=herb_collection
            ))

        # if knowledge_as_example:
        #     u_id =1000000
        #     for k,v in knowledge_examples.items():
        #         # print(v)
        #         examples.append(InputExample(
        #             user_id=u_id,
        #             chief_complaint=v[k]['Typical_performance'],
        #             history=v[k]['Definition'],
        #             detection=v[k]['Typical_performance'],
        #             syndrome_name=k,
        #             syndrome_label=k,
        #             knowledge=v[k]['Definition'],
        #         ))
        #         u_id+=1

        return examples
    
def tcm_convert_examples_to_features(examples, tokenizer, # transformers.tokenization_bert.BertTokenizer
                                      max_length=512,
                                      label_list=None,
                                      pad_on_left=False, # 没有用xlnet,所以为false
                                      pad_token=0, # 
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True,
                                      differentiation_element = None,
                                      do_train= False,
                                      has_individual_characteris= False,
                                     ):
    """
    :param examples:
    :param tokenizer:
    :param max_length:
    :param pad_on_left:
    :param pad_token:
    :param pad_token_segment_id:
    :param mask_padding_with_zero:
    :return:

    """
    processor = TcmProcessor() # 实例化类
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
    # for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if  has_individual_characteris :
            inputs = tokenizer.encode_plus(
                text=example.symptoms,
                text_pair= example.individual_characteris,
                add_special_tokens=True,
                max_length=max_length,
                # truncation_strategy="only_first"
            )
        else:
            inputs = tokenizer.encode_plus(
                text=example.symptoms,
                text_pair=None,
                add_special_tokens=True,
                max_length=max_length,
            ) # encode_plus 方法接受两个文本参数 text 和 text_pair，分别表示主诉和病史，将它们编码后返回一个字典对象。完成转换后是这样CLS 主诉 SEP 病史 SEP。inputs: input_ids + token_type_ids + attention_mask.
            
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else: # 对齐输入长度
            input_ids = input_ids + ([pad_token] * padding_length) # 
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        label = [0] * len(label_list)
        for herb in example.herb_collection:
            label[label_map[herb]] = 1


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.prescription_id))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info('input_tokens: %s' % " ".join([str(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("herb_labels : %s" % " ".join([str(x) for x in label]))


        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label
                            ))

    return features
    





