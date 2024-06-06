#!/bin/bash
export OPENAI_API_BASE=""
export OPENAI_API_KEY=""
export DASHSCOPE_API_KEY=""
export base_url='',

dataset="Leven"

for item in "Attack_4element_type" "Attack_4element"; do
  for model in "baichuan-inc/Baichuan2-7B-Chat" "ZhipuAI/chatglm3-6b" "Llama3-Chinese-8B-Instruct" "tongyifarui-890"; do
    python -u basic_gpt.py \
        --prompt="请根据案情事实，判断该案件属于以下哪个罪名。每道题仅有一个正确选项，请根据刑法的四要件理论，按照犯罪主体、犯罪客体、犯罪的主观方面、犯罪的客观方面逐步进行推理。输出格式：【正确选项】+【推理过程】。" \
        --model_type=$model \
        --dataset_type=$dataset \
        --attack=True \
        --attack_type=$item
    done
done
