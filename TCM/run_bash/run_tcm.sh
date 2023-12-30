#!/bin/bash
exp_path=/root/autodl-tmp/jiahao.geng/log

if [ ! -d  $exp_path  ]; then
  echo  making new dir.. 
  mkdir -p  $exp_path 
fi

DATE=$(date +%Y-%m-%d-%H_%M_%S)
python -u /root/autodl-tmp/jiahao.geng/TCM/model/run.py \
    --data_dir /root/autodl-tmp/jiahao.geng/data \
    --model_type bert \
    --model_name_or_path /root/autodl-tmp/ZY-BERT \
    --output_dir /root/autodl-tmp/jiahao.geng/output \
    --differentiation_element full \
    --config_name /root/autodl-tmp/ZY-BERT \
    --tokenizer_name /root/autodl-tmp/ZY-BERT \
    --cache_dir /root/autodl-tmp/jiahao.geng \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --do_test \
    --no_knowledge 4 \
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3.0 \
    --max_steps -1 \
    --warmup_steps 0 \
    --logging_steps 50 \
    --save_steps 50 \
    --eval_all_checkpoints \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed 42 \
    --local_rank -1 \
    --has_indiviual_characters \
    2>&1 | tee $exp_path/training_$DATE.log
# --has_knowledge \
# --no_cuda \
# --fp16 \
# --fp16_opt_level \
# --server_ip \
# --server_port \