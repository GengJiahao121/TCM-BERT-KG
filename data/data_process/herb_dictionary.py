import json

dataset_file_path = '/root/autodl-tmp/bert_kg/data/cleaned_result.json'
with open(dataset_file_path, 'r') as file:
    dataset = json.load(file)

herb_entity_dict = {}
for sample in dataset:
    herb_list = sample['herb_list']
    for herb in herb_list:
        if herb in herb_entity_dict:
            herb_entity_dict[herb] += 1
        else:
            herb_entity_dict[herb] = 1
    

herb_entity_dict = dict(sorted(herb_entity_dict.items(), key=lambda item: item[1]))

with open('/root/autodl-tmp/bert_kg/data/herb_dictionary.json', 'w') as file:
    json.dump(herb_entity_dict, file, ensure_ascii=False, indent=2)