import torch
state_dict = torch.load('/root/autodl-tmp/ssl-relation-prediction/src/tmp/model/custom_graph/custom_graph_ComplEx_Rank1024_RegN3_Lmbda0.1/best_valid.model', map_location='cpu') # load on cpu
entity_embedding = state_dict['embeddings.0.weight'] # get entity embeddings
relation_embedding = state_dict['embeddings.1.weight'] # get relation embeddings

print(entity_embedding.shape)