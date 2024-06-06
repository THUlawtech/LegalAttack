# @Time : 2024/1/31 下午9:58 
# @Author : LiuHuanghai
# @File : basic.py
# @Project: PyCharm
import argparse
import os
import random
from distutils.util import strtobool
import sys
import jieba

sys.path.append('/Users/mac/Documents/GitHub/LegalPromptRobust/')
#sys.path.append('/root/LegalPrompt/')
import promptbench as pb
import pandas as pd
import json
from collections import Counter
from tqdm import tqdm
import re

from nlpcda import Randomword

from modelscope import snapshot_download


# model_dir = snapshot_download('FlagAlpha/Llama3-Chinese-8B-Instruct', cache_dir="/root/autodl-tmp")
# model_dir = snapshot_download('baichuan-inc/Baichuan2-7B-Chat', cache_dir="/root/autodl-tmp")
# model_dir = snapshot_download('ZhipuAI/chatglm3-6b', cache_dir="/root/autodl-tmp")

def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_type", type=str, required=True, help="model type")
    arg_parser.add_argument("--prompt", type=str,
                            default="请根据案情事实，判断该案件属于以下哪个罪名。每道题仅有一个正确选项，你只需要返回正确选项的序号。",
                            help="prompt")
    arg_parser.add_argument(
        "--dataset_type", type=str, default="./Data", help="Data directory"
    )
    arg_parser.add_argument(
        "--fewshot",
        type=strtobool,
        default=False,
        help="few shot: Whether give some example.",
    )
    arg_parser.add_argument(
        "--fewshot_number",
        type=strtobool,
        default=False,
        help="Whether give some example.",
    )
    arg_parser.add_argument(
        "--rag",
        type=strtobool,
        default=False,
        help="Whether attack",
    )
    arg_parser.add_argument(
        "--attack",
        type=strtobool,
        default=False,
        help="Whether attack",
    )
    arg_parser.add_argument(
        "--attack_type", type=str, default="", help="attack type"
    )
    arg_parser.add_argument(
        "--sentence", type=str, default="", help="insert sentence"
    )
    arg_info = arg_parser.parse_args(args=in_args)

    return arg_info


def chooce_label_choice(ques, label):
    content = ques.split("\n选项：")[-1]
    choices = content.split(";")
    choice = []
    for x in choices:
        x = x[2:].strip()
        if "." in x:
            x = x.split(".")[-1]
        choice.append(x)
    label_choice = choice[ord(label) - ord('A')]
    return label_choice


def attack_simword_replace(input, smw):
    rs1 = smw.replace(input)
    return rs1[0]


def attack_4element_simword_replace(ques, label, attable, base_file):
    label_choice = chooce_label_choice(ques, label)
    crime_dict = attable[label_choice]
    element_words = set()
    for col in range(1, 6):
        ele_words = set(crime_dict[f"{col}"].split("、"))
        element_words = element_words.union(ele_words)
    element_words.discard('')
    content = ques.split("\n选项：")[0]
    words = jieba.lcut(content)
    # 遍历分词后的结果，随机替换集合中的词语
    new_words = []
    for word in words:
        if word in element_words:
            try:
                # 如果词语在集合中，随机替换为除此之外的词
                new_word = random.choice(base_file[word])
                new_words.append(new_word)
            except Exception:
                print(f"词语{word}不在同义词表中")
                new_words.append(word)
        else:
            # 如果不在集合中，保持原样
            new_words.append(word)
    # 将替换后的词语重新组合成字符串
    new_content = ''.join(new_words)

    ques = new_content + "\n选项：" + ques.split("\n选项：")[-1]
    return ques


def attack_choice_order(ques, label, attack_type):
    # f"\n选项：A.{choice[0]}; B.{choice[1]};C.{choice[2]}; D.{choice[3]} "
    content = ques.split("\n选项：")[-1]
    choices = content.split(";")
    choice = []
    for x in choices:
        x = x[2:].strip()
        if "." in x:
            x = x.split(".")[-1]
        choice.append(x)
    label_choice = choice[ord(label) - ord('A')]
    choice.remove(label_choice)  # 从列表中删除该元素
    if attack_type == "ChoiceOrderA":
        choice.insert(0, label_choice)  # 放在A
        label = "A"
    elif attack_type == "ChoiceOrderC":
        choice.insert(2, label_choice)  # 放在C
        label = "C"
    elif attack_type == "ChoiceOrderD":
        choice.insert(3, label_choice)  # 放在D
        label = "D"
    ques = ques.split("选项：")[0] + f"选项：A.{choice[0]};B.{choice[1]};C.{choice[2]};D.{choice[3]} "
    # print(ques,label)
    return ques, label


def attack_choice_wuzui(ques, label, attack_type):
    """
    example: 根据案情事实，判断该案件属于以下哪个罪名。每道题仅有一个正确选项，你只需要返回正确选项的序号。案情事实： 南通市通州区人民检察院指控，2015年1月20日2时许，被告人贾某在南通市通州区五
  镇江苏某某船舶重工有限公司1号宿舍楼335室，趁同宿舍被害人付某甲睡着之机，将其挂在床头的一件棉衣内侧口袋钱包中的人民币3700元窃走，并用黑色胶带缠好藏于自己的被套内。同日9时许，南通市
  州区公安局民警接警后在现场查获。被告人贾某于2015年1月20日被抓获归案，后如实供述了自己的犯罪事实。案发后，公安机关已将赃款全部发还被害人。
  选项：A.无罪;B.盗窃;C.挪用特定款物;D.侵占
    :param ques:
    :param label:
    :param attack_type:
    :return:
    """
    content = ques.split("\n选项：")[-1]
    choices = content.split(";")
    choice = []
    for x in choices:
        x = x[2:].strip()
        if "." in x:
            x = x.split(".")[-1]
        choice.append(x)
    label_choice = choice[ord(label) - ord('A')]
    replace_idx = random.choice([0, 1, 2, 3])
    while (replace_idx == label_choice):
        replace_idx = random.choice([0, 1, 2, 3])
    choice[replace_idx] = "无罪"
    ques = ques.split("选项：")[0] + f"选项：A.{choice[0]};B.{choice[1]};C.{choice[2]};D.{choice[3]} "
    return ques, label


def random_select_choice(ques, label):
    content = ques.split("\n选项：")[-1]
    choices = content.split(";")
    choice = []
    for x in choices:
        x = x[2:].strip()
        if "." in x:
            x = x.split(".")[-1]
        choice.append(x)
    label_choice = choice[ord(label) - ord('A')]
    choice.remove(label_choice)  # 从列表中删除该元素
    sim_crime = random.choice(choice)
    return sim_crime


def attack_rag_simCrime(ques, label, idx2law, crime2idx):
    sim_crime = random_select_choice(ques, label)
    idxs = crime2idx[sim_crime]
    ques += "\n相关法条:"
    for idx in idxs:
        law = idx2law[str(idx)]
        ques += law + "\n"
    return ques, sim_crime


def attack_prompt_simcrime(ques, label, attack_type):
    sim_crime = random_select_choice(ques, label)
    input_text = f"请根据案情事实，判断该行为是否构成{sim_crime}或者其他罪？每道题仅有一个正确选项，你只需要返回正确选项的序号。案情事实： {ques}\n答案："
    return input_text


def attack_sentence_noice(ques, label, sentence, attable=None):
    """
    插入句子到所有位置
    :param ques: 案情事实和选项
    :param label: 标准答案
    :param sentence: 需要添加的噪音。包括：
        "有一名"+item+"认为行为人犯了{}"
    :return:
    """
    content = ques.split("\n选项：")[0]
    choice = "\n选项：" + ques.split("\n选项：")[-1]
    sens = content.split("。")
    sim_crime = ""
    if "{" in sentence:
        # 有一名**认为行为人犯了{}
        sim_crime = random_select_choice(ques, label)
        sentence = sentence.format(sim_crime) + "。"
    if "NOICE" in sentence:
        noice_choice = random_select_choice(ques, label)
        label_choice = choice[ord(label) - ord('A')]
        while label_choice == noice_choice:
            noice_choice = random.choice(choice)
        noice = ""
        crime_dict = attable[noice_choice]
        for col in range(1, 6):
            if "type" in sentence:
                noice += crime_dict[f"{col}_type"]
            else:
                noice += crime_dict[f"{col}"]
        # ques = ques.split("\n选项：")[0] + noice + "\n选项：" + ques.split("\n选项：")[1]
        sentence = noice
    if "OBJECT" in sentence:
        sim_crime = random_select_choice(ques, label)
        data = attable.loc[sim_crime].dropna().tolist()
        sentence = random.choice(data) + "。"
    new_sens = [sentence + ques]  # 加在头
    for i in range(1, len(sens) - 1):
        new_sen = "。".join(sens[:i]) + "。" + sentence + "。".join(sens[i:]) + choice + "。"
        new_sens.append(new_sen)
    new_sens.append(content + sentence + choice)
    return new_sens, sim_crime


def attack_sentence_noice_single(ques, label, sentence, attable=None):
    """
    插入句子到结尾
    :param ques: 案情事实和选项
    :param label: 标准答案
    :param sentence: 需要添加的噪音。包括：
        行为人之前{行为}{客观要件}
        行为人之前犯过{}
        行为人自己受伤或死亡。
        这是风雨交加的一天。
    :return:
    """
    content = ques.split("\n选项：")[0]
    choice = "\n选项：" + ques.split("\n选项：")[-1]
    sens = content.split("。")
    sim_crime = ""
    if "{" in sentence:
        sim_crime = random_select_choice(ques, label)
        sentence = sentence.format(sim_crime)
    new_sens = content + sentence + choice
    return new_sens, sim_crime


def attack_4element(ques, label, attable, attack_type):
    """
    四要件结尾噪音
    """
    content = ques.split("\n选项：")[-1]
    choices = content.split(";")
    choice = []
    for x in choices:
        x = x[2:].strip()
        if "." in x:
            x = x.split(".")[-1]
        choice.append(x)
    label_choice = choice[ord(label) - ord('A')]
    # choice.remove(label_choice)  # 从列表中删除该元素
    noice_choice = random.choice(choice)
    while noice_choice == label_choice:
        noice_choice = random.choice(choice)
    noice = ""
    crime_dict = attable[noice_choice]
    for col in range(1, 6):
        if "type" in attack_type:
            noice += crime_dict[f"{col}_type"]
        else:
            noice += crime_dict[f"{col}"]
    ques = ques.split("\n选项：")[0] + noice + "\n选项：" + ques.split("\n选项：")[1]
    # print(ques)
    return chr(choice.index(noice_choice) + ord('A')), ques


def attack_casefact(data, attable, dataset_type):
    ques = data["content"]
    label = data["label"]

    label_choice = chooce_label_choice(ques, label)
    content = ques.split("\n选项：")[0]
    # 遍历分词后的结果，随机替换集合中的词语

    replace_set = set()

    if dataset_type == "Leven":
        with open("../promptbench/data/Leven_event.json", "r", encoding='utf-8') as f:
            leven_event = json.load(f)
        for triger in data["events"]:
            replace_cans = set(leven_event[triger])
            for word in data["events"][triger]:
                new_word = random.choice(list(replace_cans - {word}))
                content = content.replace(word, new_word)
                replace_set.add(word)
    else:
        new_words = []
        crime_dict = attable[label_choice]
        element_words = set()
        for col in range(1, 6):
            ele_words = set(crime_dict[f"{col}"].split("、"))
            element_words = element_words.union(ele_words)
        element_words.discard('')
        words = jieba.lcut(content)
        for word in words:
            if word in element_words:
                for col in range(1, 6):
                    ele_words = set(crime_dict[f"{col}"].split("、"))
                    if word in ele_words and len(ele_words) > 1:
                        # 如果词语在集合中，且该词语对应的要件有多个词，随机替换为除此之外的词
                        new_word = random.choice(list(ele_words - {word}))
                        new_words.append(new_word)
                        replace_set.add(word)
                        break
            else:
                # 如果不在集合中，保持原样
                new_words.append(word)
        # 将替换后的词语重新组合成字符串
        content = ''.join(new_words)
    ques = content + "\n选项：" + ques.split("\n选项：")[-1]
    return ques, replace_set


def proj_func(pred):
    if not pred:
        return -1
    result = re.findall("[a-zA-Z]+", pred)
    if result == []:
        return -1
    return result[0].upper()
    # return pred


if __name__ == "__main__":
    in_argv = parse_args()

    dataset = pb.DatasetLoader.load_dataset(in_argv.dataset_type, in_argv.model_type)

    model_name = in_argv.model_type.split("/")[-1]

    # load a model.
    # - closed source
    if in_argv.model_type in ['gpt-3.5-turbo']:
        model = pb.LLMModel(model=in_argv.model_type, max_new_tokens=200, temperature=0.0001,
                            api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_API_BASE"))
    elif in_argv.model_type in ['Azure-gpt35', 'Azure-gpt4']:
            model = pb.LLMModel(model=in_argv.model_type, max_new_tokens=200, temperature=0.0001,
                                api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_API_BASE"), deployment_name=os.environ.get("Deployment_Name"))

    else:
        # - open source
        model = pb.LLMModel(model=in_argv.model_type, max_new_tokens=200, temperature=0.0001, device='cuda')

    prompt = pb.Prompt([in_argv.prompt + "案情事实： {content} \n答案："])[0]

    dir=in_argv.dataset_type
    if in_argv.rag:
        model_name += "_rag"
    if in_argv.fewshot:
        model_name += "_fewshot"
        dir = "fewshot"
    if in_argv.attack:
        model_name += "_" + in_argv.attack_type + in_argv.sentence

    path = f"{dir}/{in_argv.dataset_type}_{model_name}_预测结果.jsonl"

    total_replace_list = []
    res_list = []
    preds = []
    labels = []
    # header = ['rawpred', 'pred', 'label']
    res_dict = {}
    # writer = csv.writer(f)
    # writer.writerow(header)
    if in_argv.attack_type in ["4element_Simword_replace", "Attack_4element", "Attack_4element_type",
                               "Attack_casefact"]:
        with open("../promptbench/data/CrimePrediction/attach_4element_240317.json", "r",
                  encoding='utf-8') as f:
            at4element = json.load(f)
    if in_argv.attack_type in ["Simword_replace"]:
        smw = Randomword(create_num=1, change_rate=0.3)
    if in_argv.attack_type in ["4element_Simword_replace"]:
        with open("../promptbench/data/processed_同义词表.json", "r",
                  encoding='utf-8') as f:
            base_file = json.load(f)
    with open(path, 'w+', encoding='utf-8') as file:
        for data in tqdm(dataset):
            if in_argv.fewshot:
                with open("../promptbench/data/few-shot-crime.json", "r", encoding='utf-8') as f:
                    fewshot_crime = json.load(f)
                choices = [chooce_label_choice(data['content'], data['label']),random_select_choice(data['content'], data['label'])]
                fewshot = ""
                for choice in choices:
                    fewshot_examples = fewshot_crime[choice]
                    fewshot_example = random.choice(fewshot_examples)
                    while fewshot_example["content"].split("\n选项：")[-1] == data["content"].split("\n选项：")[-1]:
                        fewshot_example = random.choice(fewshot_examples)
                    prompt_tmp = pb.Prompt(["参考案例，案情事实： {content}\n答案："])[0]
                    fewshot_text = pb.InputProcess.basic_format(prompt_tmp, fewshot_example)
                    fewshot+=fewshot_text+fewshot_example["label"]+"\n"
                prompt = pb.Prompt([in_argv.prompt +"前两个案情事实为参考案例，最后一个为问题。你只需要返回问题对应的正确选项的序号。"+fewshot+ "问题，案情事实： {content}\n答案：\n"])[0]
            if in_argv.rag:
                with open("../promptbench/data/刑法.json", "r", encoding='utf-8') as f:
                    idx2law = json.load(f)
                rag_laws = "\n相关法条:"
                if "law" in data:
                    for law in data["law"]:
                        rag_laws += idx2law[str(law)] + "\n"
                else:
                    with open("../promptbench/data/罪名-法条.json", "r", encoding='utf-8') as f:
                        crime2idx = json.load(f)
                    label_choice = chooce_label_choice(data['content'], data['label'])
                    idxs = crime2idx[label_choice]
                    for idx in idxs:
                        law = idx2law[str(idx)]
                        rag_laws += law + "\n"
                prompt = pb.Prompt([in_argv.prompt + "案情事实： {content}" + rag_laws + "\n答案："])[0]
            # process input
            if in_argv.attack:
                if in_argv.attack_type in ["ChoiceOrderA", "ChoiceOrderC", "ChoiceOrderD"]:
                    data['content'], data['label'] = attack_choice_order(data['content'], data['label'],
                                                                         in_argv.attack_type)
                elif in_argv.attack_type in ["Simword_replace"]:
                    data['content'] = attack_simword_replace(data['content'], smw)
                elif in_argv.attack_type in ["4element_Simword_replace"]:
                    data['content'] = attack_4element_simword_replace(data['content'], data['label'], at4element,
                                                                      base_file)
                elif in_argv.attack_type in ["Choice_wuzui"]:
                    data['content'], data['label'] = attack_choice_wuzui(data['content'], data['label'],
                                                                         in_argv.attack_type)
                elif in_argv.attack_type in ["Attack_4element", "Attack_4element_type"]:
                    noice_choice, data['content'] = attack_4element(data['content'], data['label'], at4element,
                                                                    in_argv.attack_type)
                elif in_argv.attack_type == "Attack_casefact":
                    data['content'], replace_set = attack_casefact(data, at4element,
                                                                   in_argv.dataset_type)
                    total_replace_list += list(replace_set)
                elif in_argv.attack_type == "Attack_casefact_leventrigger":
                    data['content'], replace_set = attack_casefact(data, at4element,
                                                                   in_argv.dataset_type)
                elif in_argv.attack_type in ["rag_simCrime"]:
                    with open("../promptbench/data/刑法.json", "r", encoding='utf-8') as f:
                        idx2law = json.load(f)
                    with open("../promptbench/data/罪名-法条.json", "r", encoding='utf-8') as f:
                        crime2idx = json.load(f)
                    data['content'], sim_crime = attack_rag_simCrime(data['content'], data['label'], idx2law, crime2idx)

                if in_argv.attack_type in ["Sentence_Noice"]:
                    if "NOICE" in in_argv.sentence:
                        with open("../promptbench/data/CrimePrediction/attach_4element_240317.json", "r",
                                  encoding='utf-8') as f:
                            at4element = json.load(f)
                        contents, sim_crime = attack_sentence_noice(data['content'], data['label'], in_argv.sentence,
                                                                    at4element)
                    elif "OBJECT" in in_argv.sentence:
                        path = "../promptbench/data/CrimePrediction/罪名预测易混淆罪名及标注示例-造句.xlsx"
                        df = pd.read_excel(path, sheet_name="要件造句", header=None)  # 打开excel文件
                        df.set_index(0, inplace=True)
                        contents, sim_crime = attack_sentence_noice(data['content'], data['label'], in_argv.sentence,
                                                                    df)
                    else:
                        contents, sim_crime = attack_sentence_noice(data['content'], data['label'], in_argv.sentence)
                elif in_argv.attack_type in ["Sentence_Single"]:
                    data["content"], sim_crime = attack_sentence_noice_single(data['content'], data['label'],
                                                                              in_argv.sentence)
                    input_text = pb.InputProcess.basic_format(prompt, data)
                elif in_argv.attack_type in ["Prompt_simCrime"]:
                    input_text = attack_prompt_simcrime(data['content'], data['label'], in_argv.attack_type)
                else:
                    input_text = pb.InputProcess.basic_format(prompt, data)
            else:
                input_text = pb.InputProcess.basic_format(prompt, data)

            if in_argv.attack and in_argv.attack_type in ["Sentence_Noice"]:
                raw_pred = []
                pred = []
                for content in contents:
                    data['content'] = content
                    input_text = pb.InputProcess.basic_format(prompt, data)
                    cur_raw_pred = model(input_text)
                    cur_pred = proj_func(cur_raw_pred)
                    # cur_raw_pred = cur_raw_pred.replace('\n', '#')
                    raw_pred.append(cur_raw_pred)
                    pred.append(cur_pred)
            else:
                raw_pred = model(input_text)
                # process output
                pred = proj_func(raw_pred)

                preds.append(pred)
                labels.append(data['label'])

            if in_argv.attack and in_argv.attack_type in ["Attack_4element", "Attack_4element_type"]:
                res_dict = {"raw_pred": raw_pred, "pred": pred,
                            "label": data['label'], "input_text": input_text, "noice_choice": noice_choice, }
            # elif in_argv.attack and in_argv.attack_type == "Attack_casefact":
            #     res_dict = {"raw_pred": raw_pred, "pred": pred,
            #                 "label": data['label'],"input_text":input_text,"replace_set":replace_set}
            else:
                res_dict = {"raw_pred": raw_pred, "pred": pred,
                            "label": data['label'], "input_text": input_text}
            res_list.append(res_dict)
            json.dump(res_dict, file, ensure_ascii=False)
            file.write("\n")

    if len(preds) > 0:
        score = pb.Eval.compute_cls_accuracy(preds, labels)
        print(f"{score:.3f}")
    if in_argv.attack_type == "Attack_casefact":
        print(Counter(total_replace_list))
    print(in_argv.attack_type, in_argv.sentence)
