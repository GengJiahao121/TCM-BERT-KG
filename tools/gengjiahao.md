## 数据格式的转换

将baseline_all_kg_triples.txt中的三元组转换成适用于程序运行的格式

1. head tail relation --> head relation tail

```
cd /root/autodl-tmp/ssl-relation-prediction/src/src_data/custom_graph/data_process

python exchange_postion.py
```

2. 拆分成train/valid/test

```
cd /root/autodl-tmp/ssl-relation-prediction/src/src_data/custom_graph/data_process

python split_dataset.py 

cd /root/autodl-tmp/ssl-relation-prediction/src/src_data/custom_graph

// rm train valid test

mv train.txt train

mv valid.txt valid

mv test.txt test
```

3. preprocess_datasets.py

```
rm -rf /root/autodl-tmp/ssl-relation-prediction/src/data/custom_graph

cd /root/autodl-tmp/ssl-relation-prediction/src

python preprocess_datasets.py
```

4. 对实体进行分类，以entity entity_id的形式存储，方便嵌入后，通过entity_id取出对应类别的嵌入向量，进行融合操作

```
cd /root/autodl-tmp/ssl-relation-prediction/src/src_data/custom_graph/data_process

python entity_category.py 
```

## 知识图谱的嵌入执行操作

**TCM-KG数据集：**

实体数量：36669

关系数量：16

三元组数量：123358

**wn18rr数据集：**

实体数量：40943

关系数量：22

三元组数量：93003

采用和wn18-rr相同的参数（因为实体和关系数量差不多）：

```

cd /root/autodl-tmp/ssl-relation-prediction/src

python main.py --dataset custom_graph --score_rel True --model ComplEx --rank 1024 --learning_rate 1e-1 --batch_size 100 --lmbda 0.10 --w_rel 0.05 --max_epochs 500 --cache_eval ./tmp/eval/{dataset}/{alias}/ --model_cache_path ./tmp/model/{dataset}/{alias}/ 

```

