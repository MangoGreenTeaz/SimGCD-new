#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行训练
python train_text.py \
    --data_path "data/data.csv" \
    --bert_model_path "../bert/final_model" \
    --exp_root "outputs" \
    --exp_name "simgcd_text_$(date +%Y%m%d_%H%M%S)" \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.1 \
    --num_labeled_classes 107 \
    --num_unlabeled_classes 50 \
    --max_length 512 \
    --print_freq 10 \
    --save_freq 10