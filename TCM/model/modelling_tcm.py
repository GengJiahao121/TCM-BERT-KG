# -*- coding：utf-8 -*-
'''
PLAN: Make several different methods to solve.
1. Sequence classification
'''



from transformers.modeling_bert import  BertSelfAttention, BertIntermediate, BertOutput, BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu # 因此，BertModel 是一个具体的 BERT 模型实现类，而 BertPreTrainedModel 是一个抽象基类，用于定义加载预训练模型所需的通用方法和属性。
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,RobertaModel,RobertaClassificationHead

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss,NLLLoss
             
class RobertaForTCMclassification(BertPreTrainedModel): # 继承BertPreTrainedModel类，形成自己的TCM分类模型
    '''
    almost same as the sequence classification task
    '''
    config_class = RobertaConfig # 参数设置一样
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP  # 是 Hugging Face Transformers 库中的一个字典，它包含了预训练的 RoBERTa 模型的名称和对应的下载链接。该字典的键是模型名称，值是对应的下载链接。
    base_model_prefix = "roberta"

    def __init__(self,config):
        super(RobertaForTCMclassification,self).__init__(config) # 初始化父类
        self.roberta = RobertaModel(config) # 模型
        self.classifier = RobertaClassificationHead(config) # 紧接着模型的分类器
        self.init_weights() # 权重

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class TCMForMultiHeadCrossAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)




    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        outputs = super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        return outputs

class BertForTCMClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTCMClassification, self).__init__(config)
        # self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # self.init_weights()
        self.tcm_multiHeadCrossAttention_config = config
        self.tcm_multiHeadCrossAttention_config.hidden_size = config.tcm_multiHeadCrossAttention_config_hidden_size
        self.tcm_multiHeadCrossAttention_config.num_attention_heads = config.tcm_multiHeadCrossAttention_config_num_attention_heads
        self.tcm_multiHeadCrossAttention_config.max_postion_embeddings = config.tcm_multiHeadCrossAttention_config_max_postion_embeddings
        self.tcm_multiHeadCrossAttention_config.is_decoder = config.tcm_multiHeadCrossAttention_config_is_decoder
        self.tcm_multiHeadCrossAttention_config.herb_nums = config.tcm_multiHeadCrossAttention_config_herb_nums

        self.sympotms_attenion = TCMForMultiHeadCrossAttention(self.tcm_multiHeadCrossAttention_config)
        self.herb_attenion = TCMForMultiHeadCrossAttention(self.tcm_multiHeadCrossAttention_config)

        self.symptoms_intermediate = BertIntermediate(self.tcm_multiHeadCrossAttention_config)
        self.herb_intermediate = BertIntermediate(self.tcm_multiHeadCrossAttention_config)

        self.output = BertOutput(config)

        # self.classifier = nn.Linear(config.tcm_multiHeadCrossAttention_config.hidden_size, config.tcm_multiHeadCrossAttention_config.herb_nums)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, symptoms_kg_embeddings=None, herb_kg_embeddings=None, herb_labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        # 截止到这里，经过了bert的上下文予以嵌入！！！
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)


        # 对输出进行pooling
        # 获取实际位置的掩码
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        first_sentence_mask = (token_type_ids == 0).unsqueeze(-1).expand(hidden_states.size())
        second_sentence_mask = (token_type_ids == 1).unsqueeze(-1).expand(hidden_states.size())
        first_sentence_hidden_states = hidden_states * mask * first_sentence_mask
        second_sentence_hidden_states = hidden_states * mask * second_sentence_mask
        # 计算平均池化向量
        first_sentence_pooled = torch.sum(first_sentence_hidden_states, dim=1) / torch.sum(mask * first_sentence_mask, dim=1)
        second_sentence_pooled = torch.sum(second_sentence_hidden_states, dim=1) / torch.sum(mask * second_sentence_mask, dim=1)

        # 症状层面的多头交叉注意力机制
        outputs = torch.stack((pooled_output, first_sentence_pooled, second_sentence_pooled), dim=1)
        symptoms_attention_outputs = self.sympotms_attenion.forward(hidden_states=outputs, attention_mask=None, encoder_hidden_states=symptoms_kg_embeddings, encoder_attention_mask=None)

        symptoms_attention_output = symptoms_attention_outputs[0] # 上下文向量 batchsize * 3 * 1024
        intermediate_symptoms_attention_output = self.symptoms_intermediate(symptoms_attention_output)
        symptoms_layer_output = self.output(intermediate_symptoms_attention_output, outputs) # batchsize * 3 * 1024

        # 中药层面的多头交叉注意力机制
        herb_attention_ouputs = self.herb_attenion.forward(hidden_states=herb_kg_embeddings, attention_mask=None, encoder_hidden_states=symptoms_layer_output, encoder_attention_mask=None)
        herb_attention_ouputs = herb_attention_ouputs[0]
        intermediate_herb_attention_output = self.herb_intermediate(herb_attention_ouputs)
        ## 每个中药都有对应的症候向量
        herb_layer_output = self.output(intermediate_herb_attention_output, torch.zeros_like(herb_attention_ouputs)) # batchsize * herb_nums * 1024
         
        pred_output =torch.sigmoid(torch.sum(herb_layer_output * herb_kg_embeddings, dim=2))

        # 同herb_labels计算损失
        loss = F.mse_loss(pred_output, herb_labels.float())

        # 打印计算得到的损失
        print("MSE Loss:", loss.item())

        return (loss, pred_output)
    









