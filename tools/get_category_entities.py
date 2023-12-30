
import torch
state_dict = torch.load('/root/autodl-tmp/ssl-relation-prediction/src/tmp/model/custom_graph/custom_graph_ComplEx_Rank1024_RegN3_Lmbda0.1/best_valid.model', map_location='cpu') # load on cpu
entity_embedding = state_dict['embeddings.0.weight'] # get entity embeddings
relation_embedding = state_dict['embeddings.1.weight'] # get relation embeddings

entity_mask = [False] * entity_embedding.size(0)

with open("/root/autodl-tmp/ssl-relation-prediction/src/src_data/custom_graph/entity_relation/症状_entity.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        entity_id = line.strip().split('\t')
        entity_mask[int(entity_id[1])] = True

# 取出来的前半部分为实部，后半部分为虚部
sympotms_embeddings = entity_embedding[entity_mask]

sympotms_embeddings = sympotms_embeddings[:, :1024]

print(sympotms_embeddings)












