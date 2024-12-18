#!/bin/bash
export OPENAI_API_BASE= # if use openai
export OPENAI_API_KEY= # if use openai
export Deployment_Name= # if use Azure API
export DASHSCOPE_API_KEY= # if use farui
export base_url= # if use farui

model=""
dataset="CrimePrediction"

python -u basic_attack.py \
    --model_type=$model \
    --dataset_type=$dataset \
    --attack=False \

for item in "ChoiceOrderA" "ChoiceOrderC" "ChoiceOrderD"; do
  python -u basic_attack.py \
      --model_type=$model \
      --dataset_type=$dataset \
      --attack=True \
      --attack_type=$item
done

for item in "Prompt_simCrime"; do
  python -u basic_attack.py \
      --model_type=$model \
      --dataset_type=$dataset \
      --attack=True \
      --attack_type=$item
done

python -u basic_attack.py \
    --model_type=$model \
    --dataset_type=$dataset \
    --attack=True \
    --attack_type="Attack_casefact" \

for item in "Attack_4element_type" "Attack_4element"; do
  python -u basic_attack.py \
      --model_type=$model \
      --dataset_type=$dataset \
      --attack=True \
      --attack_type=$item
done

for item in "Simword_replace" "4element_Simword_replace"; do
  python -u basic_attack.py \
      --model_type=$model \
      --dataset_type=$dataset \
      --attack=True \
      --attack_type=$item
done

for item in "行为人之前犯过{}罪。" "这是风雨交加的一天。" "这是风和日丽的一天。" "这是月黑风高夜，杀人放火天。"; do
  python -u basic_attack.py \
      --model_type=$model \
      --dataset_type=$dataset \
      --attack=True \
      --attack_type="Sentence_Single" \
      --sentence=$item
done

# rag：加入对应罪名的法条的结果
python -u basic_attack.py \
  --model_type=$model \
  --dataset_type=$dataset  \
  --rag=True \
  --attack=False

# 加入相似罪名的法条的结果
python -u basic_attack.py \
  --model_type=$model \
  --dataset_type=$dataset  \
  --rag=False \
  --attack=True \
  --attack_type="rag_simCrime"


for item in "小学生" "群众" "律师" "法学生" "法官"; do
  python -u basic_attack.py \
      --model_type=$model \
      --dataset_type='CrimePrediction' \
      --attack=True \
      --attack_type="Sentence_Noice" \
      --sentence="有一名"$item"认为行为人犯了{}"
done