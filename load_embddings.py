

import torch

cls_embeddings = torch.load('/root/autodl-tmp/get_herb_embeddings/output/cls_embeddings')

BertEmbeddings_embeddings = torch.load('/root/autodl-tmp/get_herb_embeddings/output/BertEmbeddings_embeddings')

hidden_states_embeddings = torch.load('/root/autodl-tmp/get_herb_embeddings/output/hidden_states_embeddings') # herb_len x max_char_len x dim
hidden_states_attention_mask_list = torch.load('/root/autodl-tmp/get_herb_embeddings/output/hidden_states_attention_mask_list') # herb_len x max_char_len
# 将 attention_mask 扩展到与 hidden_states_embeddings 相同的维度
extended_attention_mask = hidden_states_attention_mask_list.unsqueeze(2)
# 使用 attention_mask 进行掩码，将不需要考虑的位置置零
masked_hidden_states = hidden_states_embeddings * extended_attention_mask
# 对最后一个维度进行求和，并除以非零元素的数量，得到平均值
sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
mean_pooled_states = sum_hidden_states / extended_attention_mask.sum(dim=1)
# mean_pooled_states 就是 herb_len x dim 的向量矩阵

