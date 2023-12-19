import json
import re

dataset_file_path = '/root/autodl-tmp/bert_kg_v1/data/cleaned_result.json'
with open(dataset_file_path, 'r') as file:
    dataset = json.load(file)

herb_entity_dict = {}
for sample in dataset:
    herb_list = sample['herb_list']
    if 'add_or_sub_list' in sample:
        add_or_sub_list = sample['add_or_sub_list']
    for herb in herb_list:
        if herb in herb_entity_dict:
            herb_entity_dict[herb] += 1
        else:
            herb_entity_dict[herb] = 1
    for add_or_sub in add_or_sub_list:
        herbs = add_or_sub['herbs']
        for herb in herbs:
            if '去' in herb or '减' in herb or '裁' in herb:
                herb = re.sub('[去减裁]', '', herb)
                if herb in herb_entity_dict:
                    herb_entity_dict[herb] += 1
                else:
                    herb_entity_dict[herb] = 1
            else:
                if herb in herb_entity_dict:
                    herb_entity_dict[herb] += 1
                else:
                    herb_entity_dict[herb] = 1

herb_entity_dict = dict(sorted(herb_entity_dict.items(), key=lambda item: item[1]))

with open('/root/autodl-tmp/bert_kg_v1/data/herb_dictionary.json', 'w') as file:
    json.dump(herb_entity_dict, file, ensure_ascii=False, indent=2)