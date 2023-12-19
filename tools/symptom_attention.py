
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.modeling_bert import BertSelfAttention
import torch
from torch.nn import Module, Parameter

class Symptoms_MultiHeadCrossAttention(BertSelfAttention):
    def __init__(self, config):
        pass



    def forward(self):
        pass



if __name__ == '__main__':
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.hidden_size = 1024
    config.num_attention_heads = 12
    config.max_position_embeddings = 512
    config.is_decoder = True

    
    print()
    
