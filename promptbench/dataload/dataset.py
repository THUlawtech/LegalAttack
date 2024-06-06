# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import random
import requests
import json

from ..models import MAXLEN

def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)


class Dataset(object):
    def __init__(self, dataset_name, model):
        self.data = []
        self.dataset_name = dataset_name
        self.model = model

        # Get the parent directory
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(cur_dir), 'data')

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        # check if the dataset exists, if not, download it
        if dataset_name == "gsm8k":
            self.filepath = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
        else:
            self.filepath = os.path.join(self.data_dir, f"{dataset_name}.json")

        if not os.path.exists(self.filepath):
            if dataset_name == "gsm8k":
                url = f'https://wjdcloud.blob.core.windows.net/dataset/promptbench/dataset/{dataset_name}.jsonl'
            else:
                url = f'https://wjdcloud.blob.core.windows.net/dataset/promptbench/dataset/{dataset_name}.json'
            print(f"Downloading {dataset_name} dataset...")
            response = requests.get(url)
            with open(self.filepath, 'wb') as f:
                f.write(response.content)

    def __len__(self):
        assert len(self.data) > 0, "Empty dataset. Please load data first."
        return len(self.data)

    def __getitem__(self, idx):
        assert len(self.data) > 0, "Empty dataset. Please load data first."
        return self.data[idx]

    def extract_answer(self, output):
        return output

import tiktoken
def num_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

class CrimePrediction(Dataset):
    """
    cail2018 dataset

    Example data format:
    {"fact": "公诉机关指控,2016年3月22日19时许,临沭县青云镇白旄西街张某到其大伯哥刘某乙兴家查看婆婆去世的买菜账本,
    与张某的侄子刘某甲发生争执,后被被告人刘某甲殴打致伤,同年4月8日,被害人张某报案至临沭县公安局白旄派出所,同年4月22日
    经法医鉴定:张某之损伤构成轻伤二级。\n公诉机关认为,被告人刘某甲故意伤害他人身体,致人轻伤;其行为触犯了《中华人民共和
    国刑法》××之规定,犯罪事实清楚,证据确实充分,应当以××追究其刑事责任。\n", "meta": {"punish_of_money": 0,
    "accusation": ["故意伤害"], "relevant_articles": ["234"], "criminals": ["刘某甲"],
    "term_of_imprisonment": {"death_penalty": false, "imprisonment": 8, "life_imprisonment": false}}}
    """

    def __init__(self, model):
        super().__init__("CrimePrediction", model)
        self.data = []
        # path = f"../data/CrimePrediction/data_valid_test/data_valid_typed_240427.json"
        # generate_space=100
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        instruction = "请根据案情事实，判断该案件属于以下哪个罪名。每道题仅有一个正确选项，你只需要返回正确选项的序号。案情事实："
        self.data = []
        cutcount=0
        maxlen=MAXLEN[model]-100 #预留空间
        for typed_data in data:
            if num_tokens(instruction+typed_data["content"]) > maxlen:
                input_list = typed_data["content"].split("。")
                input = ""
                for sen in input_list:
                    if num_tokens(instruction + input + sen + "。") < maxlen:
                        input += sen + "。"
                    else:
                        break
                typed_data["content"] = input
                cutcount +=1
            typed_data["content"] +=typed_data["choice"]
            self.data.append(typed_data)
        print(f"{model},maxlen:{MAXLEN[model]},cut document:{cutcount}")
        # for text in f:
            #     d = json.loads(text)
            #     self.data.append({"content": d["fact"], "label": d['meta']["accusation"]} )


    def extract_answer(self, output):
        if self.dataset_name == "bigbench_date":
            answer = re.findall(r'A|B|C|D|E|F', output)
        elif self.dataset_name == "bigbench_object_tracking":
            answer = re.findall(r'A|B|C', output)

        answer = answer[0] if len(answer) > 0 else ""

        # print(answer)
        return answer


class Leven(Dataset):
    """
    Leven dataset

    Example data format:
      {
        "content": "被告人王某于2012年1月至10月间，伙同叶某某（已判刑）、吕某某（另案处理）等人伪造保险事故，骗取保险公司理赔款共计人民币115569元。1、2012年1月31日，被告人王某伙同叶某某等人在锡宜高速公路上利用车牌号分别为苏B7559A、苏B077A0的两辆汽车故意制造保险事故，骗得中国太平洋财产保险股份有限公司无锡分公司交通事故理赔款共计人民币97234元。2、2012年8月6日，被告人王某伙同吕某某等人，利用车牌号为苏B9S090的汽车故意制造保险事故，骗得中国大地财产保险股份有限公司交通事故理赔款共计人民币7585元。3、2012年10月19日上午，被告人王某伙同吕某某等人，利用车牌号分别为苏B3903N、苏BQP683的两辆汽车编造保险事故，骗得紫金财产保险股份有限公司交通事故理赔款共计人民币10750元。被告人王某归案后如实供述了其参与诈骗的犯罪事实。案发后，涉案人员叶某某家属代为退出赃款人民币60000元。在本案审理期间，被告人王某的亲属代为退出赃款人民币55569元。\n选项：A.票据诈骗;B.集资诈骗;C.贷款诈骗;D.诈骗 ",
        "label": "D"
      }
    """

    def __init__(self, model):
        super().__init__("Leven", model)
        self.data = []

        # path = f"../data/CrimePrediction/data_valid_test/data_valid_typed_240427.json"
        # generate_space=100
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        instruction = "请根据案情事实，判断该案件属于以下哪个罪名。每道题仅有一个正确选项，你只需要返回正确选项的序号。案情事实："
        self.data = []
        cutcount=0
        maxlen = MAXLEN[model] - 100  # 预留空间
        for typed_data in data:
            typed_data["choice"] = "选项："+ typed_data["content"].split("选项：")[1]
            typed_data["content"]= typed_data["content"].split("选项：")[0]
            if num_tokens(instruction+typed_data["content"]) > maxlen:
                input_list = typed_data["content"].split("。")
                input = ""
                for sen in input_list:
                    if num_tokens(instruction + input + sen + "。") < maxlen:
                        input += sen + "。"
                    else:
                        break
                typed_data["content"] = input
                cutcount +=1
            typed_data["content"] +=typed_data["choice"]
            self.data.append(typed_data)
        print(f"{model},maxlen:{MAXLEN[model]},cut document:{cutcount}")
        # for text in f:
            #     d = json.loads(text)
            #     self.data.append({"content": d["fact"], "label": d['meta']["accusation"]} )

