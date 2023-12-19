## 调用参数
import json
import random

# 读取数据集
with open('/root/autodl-tmp/bert_kg_v1/data/cleaned_result.json', 'r') as f:
    dataset = json.load(f)

# 按照比例拆分数据集
random.shuffle(dataset)
total_samples = len(dataset)
train_samples = int(total_samples * 0.877)
dev_samples = int(total_samples * 0.057)
test_samples = int(total_samples * 0.066)

train_data = dataset[:train_samples]
dev_data = dataset[train_samples:train_samples+dev_samples]
test_data = dataset[train_samples+dev_samples:]

# 保存拆分后的数据集
with open('/root/autodl-tmp/bert_kg_v1/data/train.json', 'w') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('/root/autodl-tmp/bert_kg_v1/data/dev.json', 'w') as f:
    json.dump(dev_data, f, ensure_ascii=False, indent=2)

with open('/root/autodl-tmp/bert_kg_v1/data/test.json', 'w') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("数据集拆分完成！")

