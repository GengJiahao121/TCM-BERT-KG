# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""
# # 
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
logger = logging.getLogger(__name__) # 这段代码的作用是创建一个名为__name__的logger对象，用于记录程序运行时的日志信息。

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

torch.cuda.empty_cache()

def simple_accuracy(preds, labels): # 统计预测正确的样本数，然后除以总样本数
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed) # 设置python的随机种子
    np.random.seed(args.seed) # 设置numpy的随机种子
    torch.manual_seed(args.seed) # 设置pytorch的随机种子
    if args.n_gpu > 0: # 设置cuda的随机种子
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, evaluate=False,test=False): # 加载数据 并 处理（batch和tensor）
    tcm_processor = processor() # 类的实例化，也就是说变成了对象
    if args.local_rank not in [-1, 0] and not evaluate: # 暂停其它进程，只留下主进程
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    if evaluate:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'dev',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(args.differentiation_element)
        ))
    elif test:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'test' ,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(args.differentiation_element)
            ))
    else:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(args.differentiation_element)
        )) # cached_train_ZY-BERT_128_full # cached_features_file 是一个缓存文件，用于存储预处理后的特征，在使用缓存文件时，通常会先检查缓存文件是否存在
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file) # 不覆写，直接加载
    else: # 创造特征向量
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = tcm_processor.get_herb_labels() 
        # if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta', 'xlmroberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            # label_list[1], label_list[2] = label_list[2], label_list[1]
        if evaluate:
            examples = tcm_processor.get_dev_examples(args.data_dir,args)
        elif test:
            examples = tcm_processor.get_test_examples(args.data_dir,args)
        else:
            # examples = tcm_processor.get_train_examples(args.data_dir,args)
            examples = tcm_processor.get_train_examples(args.data_dir,args)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                pad_on_left=bool(args.model_type in ['xlnet']), # 如果 args.model_type 的取值为 'xlnet'，则 pad_on_left 的值为 True，否则为 False
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0], 
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                differentiation_element=args.differentiation_element, # ???
                                                do_train=(not evaluate) and (not test),
                                                has_individual_characteris=args.has_individual_characteris,
                                                )
       
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)



    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) # 200数量 * 128长度
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)


    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def load_symptoms_kg_embeddings(args, evaluate=False,test=False):

    if args.local_rank not in [-1, 0] and not evaluate: # 暂停其它进程，只留下主进程
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    kg_best_model_path = args.kg_best_model
    symptoms_entities_ids_path = args.sympotm_entities_ids

    state_dict = torch.load(kg_best_model_path, map_location= 'cpu' if args.no_cuda else 'cuda') # load on cpu
    entity_embedding = state_dict['embeddings.0.weight'] # get entity embeddings
    relation_embedding = state_dict['embeddings.1.weight'] # get relation embeddings

    entity_mask = [False] * entity_embedding.size(0)

    with open(symptoms_entities_ids_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            entity_id = line.strip().split('\t')
            entity_mask[int(entity_id[1])] = True

    # 取出来的前半部分为实部，后半部分为虚部
    sympotms_embeddings = entity_embedding[entity_mask]

    sympotms_embeddings = sympotms_embeddings[:, :1024]

    # kg_embeddings = TensorDataset(sympotms_embeddings)
    kg_embeddings = sympotms_embeddings

    logger.info("load_symptoms_kg_embeddings : %s", args.kg_best_model)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset

    return kg_embeddings

def load_herb_kg_embeddings(args, evaluate=False,test=False):
    if args.local_rank not in [-1, 0] and not evaluate: # 暂停其它进程，只留下主进程
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    kg_best_model_path = args.kg_best_model
    herb_entities_ids_path = args.herb_entities_ids

    state_dict = torch.load(kg_best_model_path, map_location= 'cpu' if args.no_cuda else 'cuda') # load on cpu
    entity_embedding = state_dict['embeddings.0.weight'] # get entity embeddings
    relation_embedding = state_dict['embeddings.1.weight'] # get relation embeddings

    entity_mask = [False] * entity_embedding.size(0)

    with open(herb_entities_ids_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            entity_id = line.strip().split('\t')
            entity_mask[int(entity_id[1])] = True

    # 取出来的前半部分为实部，后半部分为虚部
    herb_embeddings = entity_embedding[entity_mask]

    herb_embeddings = herb_embeddings[:, :1024]

    # kg_embeddings = TensorDataset(sympotms_embeddings)
    kg_embeddings = herb_embeddings

    logger.info("load_herb_kg_embeddings : %s", args.kg_best_model)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset

    return kg_embeddings

def evaluate(args, model, symptoms_kg_embeddings, herb_kg_embeddings, tokenizer, prefix=""):
    results = {}

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=args.do_eval,test=args.do_test)

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'herb_labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            
            inputs['symptoms_kg_embeddings'] = symptoms_kg_embeddings.unsqueeze(0).repeat(batch[0].size(0), 1, 1)
            inputs['herb_kg_embeddings'] = herb_kg_embeddings.unsqueeze(0).repeat(batch[0].size(0), 1, 1)

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item() # .mean()：这个函数计算 tmp_eval_loss 张量的平均值。它计算张量中所有元素的平均值。.item()：这个函数将平均值张量转换为 Python 标量。它从张量中提取一个单独的元素，并将其返回为原生的 Python 数据类型（如浮点数或整数）

        nb_eval_steps += 1
        if preds is None:
            # 预测值
            preds = logits.detach().cpu().numpy()
            # 真实值
            out_label_ids = inputs['herb_labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['herb_labels'].detach().cpu().numpy(), axis=0)    

    # store_preds_labels(eval_output_dir,out_label_ids.tolist(),preds.tolist())
    eval_loss = eval_loss / nb_eval_steps
    #preds = np.argmax(preds, axis=1) # 将 preds 数组沿着第一个轴（axis=1）的方向找出最大值的索引,也就是148个类别中的一个数，这个是代表了证型对应的数字id

    # result = {
    #     "acc": simple_accuracy(preds, out_label_ids),
    #     # "mcc": matthews_corrcoef(out_label_ids, preds)
    # }
    # 1. 精确度
    threshold = 0.5
    preds = (preds > threshold).astype(int)
    accuracy = accuracy_score(out_label_ids, preds)
    # 2. 精确度、召回率、F1分数
    precision = precision_score(out_label_ids, preds, average='micro')
    recall = recall_score(out_label_ids, preds, average='micro')
    f1 = f1_score(out_label_ids, preds, average='micro')

    # 3. Top K 准确度
    def top_k_accuracy(out_label_ids, predicted_probs, k):
        top_k_preds = np.argsort(predicted_probs, axis=1)[:, -k:]
        true_labels_top_k = np.argsort(out_label_ids, axis=1)[:, -k:]
        
        # 计算交集
        intersection = np.intersect1d(top_k_preds, true_labels_top_k)
        
        top_k_acc = len(intersection) / len(out_label_ids)
        return top_k_acc

    k = 5
    top_k_accuracy_value = top_k_accuracy(out_label_ids, preds, k)

    result = {}
    result['acc'] = accuracy
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1
    result['top_k_accuracy_value'] = top_k_accuracy_value
    results.update(result)

    if args.do_eval:
        output_eval_file = os.path.join(eval_output_dir, prefix, "dev_eval_results.txt")
    elif args.do_test:
        output_eval_file = os.path.join(eval_output_dir, prefix, "test_eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        # for key in sorted(result.keys()):
        #     logger.info("  %s = %s", key, str(result[key]))
            # writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info("  %s = %s", "accuracy", str(result['acc']))
        logger.info("  %s = %s", "precision", str(result['precision']))
        logger.info("  %s = %s", "recall", str(result['recall']))
        logger.info("  %s = %s", "f1", str(result['f1']))
        logger.info("  %s = %s", "top_k_accuracy_value", str(result['top_k_accuracy_value']))
        writer.write(json.dumps(results,indent=2))

    return results

def train(args, train_dataset, symptoms_kg_embeddings, herb_kg_embeddings, tcm_bert_model, tcm_bert_tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset) #  具体来说，RandomSampler 是一个随机采样器，它会随机地从数据集中选择样本，用于训练模型。而 DistributedSampler 是一个分布式采样器，它会将数据集分成多个部分，每个部分由不同的进程或设备负责采样，用于分布式训练模型。
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs # 首先计算每个 epoch 中的步数， 然后将每个 epoch 的步数乘以训练的总 epoch 数 

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in tcm_bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in tcm_bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 统计参数的数量
    total_params = 0
    for group in optimizer_grouped_parameters:
        total_params += sum(p.numel() for p in group['params'])
    logger.info("Total number of parameters: %d", total_params)
    #print(f"Total number of parameters: {total_params}")


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total) # 学习率调度器，这个函数会根据总的训练步数 t_total，以及预热步数 num_warmup_steps 来计算每个训练步的学习率。预热步数是指在训练开始时，先使用一个较小的学习率进行预热，然后再逐渐增加学习率。这个过程可以帮助模型更好地收敛。
    #from apex import ApexImplementatio
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        tcm_bert_model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        tcm_bert_model = torch.nn.DataParallel(tcm_bert_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        tcm_bert_model = torch.nn.parallel.DistributedDataParallel(tcm_bert_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps) # .gradient_accumulation_steps梯度累积是指将多个小批量的梯度累积起来，然后一次性执行一步优化。
    logger.info("  Total optimization steps = %d", t_total) # 

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tcm_bert_model.zero_grad() # 每次更新模型参数之前，我们需要将之前计算的梯度清零
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]) # disable=args.local_rank not in [-1, 0] 表示如果当前进程不是主进程，则禁用进度条的显示。
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]) # train_dataloader 是一个 PyTorch 的 DataLoader 对象，用于加载训练数据
        for step, batch in enumerate(epoch_iterator): # 通过 epoch_iterator 来迭代训练数据集中的每个 batch
            tcm_bert_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'herb_labels': batch[3]
                      }
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                         'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            inputs['symptoms_kg_embeddings'] = symptoms_kg_embeddings.unsqueeze(0).repeat(batch[0].size(0), 1, 1)
            inputs['herb_kg_embeddings'] = herb_kg_embeddings.unsqueeze(0).repeat(batch[0].size(0), 1, 1)
            
            # tcm_bert_oupputs: pooled_output, first_sentence_pooled, second_sentence_pooled
            tcm_bert_outputs = tcm_bert_model(**inputs)

            loss = tcm_bert_outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16: # 梯度裁剪是一种防止梯度爆炸的技术，它可以限制梯度的范数，使得梯度不会变得过大。
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(tcm_bert_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                tcm_bert_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and (global_step) % args.logging_steps == 0: # 每隔多少个步骤输出一次日志
                        logs = {}
                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(args, tcm_bert_model, symptoms_kg_embeddings, herb_kg_embeddings, tcm_bert_tokenizer)
                            for key, value in results.items():
                                eval_key = 'eval_{}'.format(key)
                                logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            if isinstance(value, (int, float)):
                                value = {key: value}
                            tb_writer.add_scalars(key, value, global_step)
                        print(json.dumps({**logs, **{'step': global_step}}))

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model,
                #                                             'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tcm_bert_model_to_save = tcm_bert_model.module if hasattr(tcm_bert_model,
                                                'module') else tcm_bert_model  # Take care of distributed/parallel training
        tcm_bert_model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving tcm_bert_model checkpoint to %s", output_dir)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def main():
    
    parser = argparse.ArgumentParser() # 创建参数解析器对象

    ## Required parameters
    # 微调数据的路径：../../to_LJW/
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # 预训练transformers模型的具体模型
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # transformer所在的路径
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    # 输出文件夹，包括模型的预测结果和checkpoints（训练过程中保存模型的某个特定时刻的状态）
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # 
    parser.add_argument("--differentiation_element", default="full", type=str, required=True,
                        help="The input element for syndrome differentiation")
    ## Other parameters
    # 
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    # 
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    # 
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    # 
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    # 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    # parser.add_argument("--do_shuffle", action='store_true',
    #                     help="Whether to shuffle input sentences.")
    # 
    parser.add_argument("--has_knowledge", action='store_true',
                        help="Whether to use additional knowledge")
    # 
    parser.add_argument("--no_knowledge", default=4, type=int,
                        help="Number of additional knowledge")
    # parser.add_argument("--has_knowledge_as_instances", action='store_true',
    #                     help="Whether to use additional knowledge as additional examples")
    # 在每个轮次进行RUL评估
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    # 全部小写
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    # 训练时 batch size = 8
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    # 评估时 batch size = 8
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    # 
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    # 覆写输出文件夹
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    # 覆写经过预处理过的特征向量，如果第一次处理过了，就不用覆写了，那样会浪费很多时间
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    # 默认只用一台机器训练
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    # ip和端口号
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--has_individual_characteris', action='store_true',
                        help='Whether add individual_characteris')
    parser.add_argument('--kg_best_model', type=str, default='/root/autodl-tmp/bert_kg_v1/kg-embeddings/best_valid.model')
    parser.add_argument('--sympotm_entities_ids', type=str, default='/root/autodl-tmp/bert_kg_v1/kg-embeddings/src_data/custom_graph/entity_relation/症状_entity.txt')
    parser.add_argument('--herb_entities_ids', type=str, default='/root/autodl-tmp/bert_kg_v1/kg-embeddings/src_data/custom_graph/entity_relation/中药_entity.txt')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    tcm_processor = processor() # 实例化数据处理类 
    # label_list = tcm_processor.get_labels() # 证型列表,此时为汉字 $
    # num_labels = len(label_list) # 证型列表的长度 $
    herb_dict = tcm_processor.get_herb_labels()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]: # 进程不是-1或者0的阻塞，只留下主进程
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower() # 将模型名->小写
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tcm_bert_config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          # num_labels=128,
                                          # finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    
    #tcm_bert_config.tcm_multiHeadCrossAttention_config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
    #                                                                                    cache_dir=args.cache_dir if args.cache_dir else None)
    tcm_bert_config.tcm_multiHeadCrossAttention_config_hidden_size = 1024
    tcm_bert_config.tcm_multiHeadCrossAttention_config_num_attention_heads = 2
    tcm_bert_config.tcm_multiHeadCrossAttention_config_max_postion_embeddings = 512
    tcm_bert_config.tcm_multiHeadCrossAttention_config_is_decoder = True
    tcm_bert_config.tcm_multiHeadCrossAttention_config_herb_nums = 947
    
    tcm_bert_tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    tcm_bert_model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=tcm_bert_config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0: # 进程为0的阻塞，言外之意就是释放所有进程
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tcm_bert_model.to(args.device)  

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tcm_bert_tokenizer, evaluate=False)
        sympotms_kg_embeddings = load_symptoms_kg_embeddings(args)
        herb_kg_embeddings = load_herb_kg_embeddings(args)

        global_step, tr_loss = train(args, train_dataset, sympotms_kg_embeddings, herb_kg_embeddings, tcm_bert_model, tcm_bert_tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = tcm_bert_model.module if hasattr(tcm_bert_model,
                                                'module') else tcm_bert_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tcm_bert_tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        tcm_bert_model = model_class.from_pretrained(args.output_dir)
        tcm_bert_tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        tcm_bert_model.to(args.device)

    # Evaluation
    results = {}
    if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]:
        tcm_bert_tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case) # ，如果你使用了默认的 tokenizer 名称，那么可以直接使用 from_pretrained() 函数加载 tokenizer，而不需要手动指定 tokenizer 的路径。
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints: 
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            tcm_bert_model = model_class.from_pretrained(checkpoint)
            tcm_bert_model.to(args.device)
            result = evaluate(args, tcm_bert_model, sympotms_kg_embeddings, herb_kg_embeddings,tcm_bert_tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results
        
if __name__ == "__main__":
    main() # 主函数
