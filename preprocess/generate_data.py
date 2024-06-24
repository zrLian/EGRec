import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
model_name = 'Qwen/Qwen1.5-0.5B'
max_padding_len = 512
ctr_hist_len = 5
bert =False



# 加载JSON Lines文件以获取reviewerID到uid的映射
def load_mapping(jsonl_file):
    id_to_uid_or_iid = {}
    with open(jsonl_file, 'r') as file:
        for line in file:
            try:
                # 解析每一行为一个JSON对象
                item = json.loads(line)
                 # 检查是使用uid还是iid作为下标
                if 'uid' in item:
                    value = item['uid']
                elif 'iid' in item:
                    value = item['iid']
                else:
                    continue
                
                if 'reviewerID' in item:
                    key = item['reviewerID']
                elif 'asin' in item:
                    key = item['asin']
                else:
                    continue

                id_to_uid_or_iid[key] = value

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return id_to_uid_or_iid


# 加载pkl文件以获取用户的嵌入向量
def load_embeddings(pkl_file):
    with open(pkl_file, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

# 为所有reviewerID生成对应的嵌入向量字典
def generate_embeddings_dict(json_file, pkl_file):
    id_to_uid = load_mapping(json_file)
    embeddings = load_embeddings(pkl_file)
    reviewerID_to_emb = {}

    for reviewerID, uid in id_to_uid.items():
        emb = embeddings[uid]  # 假设embeddings是一个可以通过uid索引的结构
        reviewerID_to_emb[reviewerID] = emb

    return reviewerID_to_emb

def generate_ctr_data(sequence_data, lm_hist_idx, uid_set,datamap,is_train=False):
    # print(list(lm_hist_idx.values())[:10])
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    id2user = datamap['id2user']

    


    full_data = []
    total_label = []
    prompts= [] 
    # rlmRec_user_emb = []
    # rlmRec_item_emb = []

    user_have = 0
    user_no = 0


    auto_tokenizer = AutoTokenizer.from_pretrained(model_name)

    auto_tokenizer.padding_side = "right"
    input_ids = []

    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq,review_seq,summary_seq = sequence_data[str(uid)]
        no_des=0
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            iid = item_seq[idx]
            full_data.append([uid, idx, label])
            total_label.append(label)
            
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
                  sum(total_label) / len(total_label))
            
    return full_data







if __name__ == '__main__':
    random.seed(12345)
    #DATA_DIR = '../data/'
    DATA_DIR = ''

    DATA_SET_NAME = ''

    if DATA_SET_NAME == 'ml-1m':
        rating_threshold = 3
    else:
        rating_threshold = 4
    # PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME)
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')

    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    datamap = load_json(DATAMAP_PATH)
    print('final loading data')
    print(list(item2attribute.keys())[:10])


    print('generating ctr train dataset')
    train_ctr= generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'],datamap,True)
    print('generating ctr test dataset')
    test_ctr= generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'],datamap,False)
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

   

  

    statis = {
         'rerank_list_len': rerank_list_len,
         'attribute_ft_num': datamap['attribute_ft_num'],
         'rating_threshold': rating_threshold,
         'item_num': len(datamap['id2item']),
         'attribute_num': len(datamap['id2attribute']),
         'rating_num': 5,
         'dense_dim': 0,
     }
    save_json(statis, PROCESSED_DIR + '/stat.json')

   





  


 
