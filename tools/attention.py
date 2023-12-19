from transformers import BertTokenizer, BertModel
import torch

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 输入文本
text = "Hello, how are you doing today?"

# 使用分词器对文本进行分词和编码
inputs = tokenizer(text, return_tensors="pt")

# 获取模型的输出
outputs = model(**inputs)

# 提取模型的多头注意力机制的输出
attention_outputs = outputs.attentions

# 打印多头注意力机制的输出（这是一个列表，每个元素对应一个注意力头）
for i, attention_head in enumerate(attention_outputs):
    print(f"Attention Head {i + 1}: {attention_head}")
