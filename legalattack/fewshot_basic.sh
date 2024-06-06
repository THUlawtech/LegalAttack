#!/bin/bash
export OPENAI_API_BASE=""
export OPENAI_API_KEY=""
export DASHSCOPE_API_KEY=""
export base_url='',

dataset="Leven"

for item in "Attack_4element_type" "Attack_4element"; do
  for model in "baichuan-inc/Baichuan2-7B-Chat" "ZhipuAI/chatglm3-6b" "Llama3-Chinese-8B-Instruct" "tongyifarui-890"; do
    python -u basic_gpt.py \
        --model_type=$model \
        --dataset_type=$dataset \
        --fewshot=True \
        --attack=True \
        --attack_type=$item
    done
done

