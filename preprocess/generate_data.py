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
rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10
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

    #rlmrec的emb
    # rlmRec_user_raw2emb = generate_embeddings_dict(rlmRec_amz_user_id2raw,rlmRec_amz_user_pkl)
    # rlmRec_item_raw2emb = generate_embeddings_dict(rlmRec_amz_item_id2raw,rlmRec_amz_item_pkl)


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
            
            #我们的方法
            #idx表示cur_idx
            hist_item_seq = item_seq[:idx] if idx<=ctr_hist_len else item_seq[idx-ctr_hist_len:idx]
            hist_rating_seq = rating_seq[:idx] if idx<=ctr_hist_len else rating_seq[idx-ctr_hist_len:idx]
            hist_summary_seq = summary_seq[:idx] if idx<=ctr_hist_len else summary_seq[idx-ctr_hist_len:idx]
            
            history_texts = []
            for iid, rating,summary in zip(hist_item_seq, hist_rating_seq,hist_summary_seq):
                # tmp = '{}, {} stars, {}; '.format(itemid2title[str(iid)], int(rating),summary)
                brand,rank,price,cate,description=item2attribute[str(iid)]
                
                brand_name = attrid2name[str(brand)]
                cate_name = attrid2name[str(cate)]
                rank = attrid2name[str(rank)]
                price = attrid2name[str(price)]
                #summary = summary_seq[idx]
                brand_name_len = brand_name.split(' ')
                if brand_name_len<=3:
                    print()
                else:
                    print(brand_name_len)
                tmp = '{}#{} stars#{}#{}#{}; '.format(itemid2title[str(iid)], int(rating),summary,brand_name,cate_name.replace(';',','))
                history_texts.append(tmp.replace("{", ""))
            title = itemid2title[str(iid)]
            # brand, cate ,description,rank,price= item2attribute[str(iid)]
            if len(item2attribute[str(iid)])<5:
                brand,rank,price,cate=item2attribute[str(iid)]
                description = "None"
                no_des+=1
            else:
                brand,rank,price,cate,description=item2attribute[str(iid)]
                description = attrid2name[str(description)]

            brand_name = attrid2name[str(brand)]
            cate_name = attrid2name[str(cate)]
            rank = attrid2name[str(rank)]
            price = attrid2name[str(price)]
            summary = summary_seq[idx]
            review = review_seq[idx]
            

            question1 = "I will provide you with some  users' history of browsing books and current book information." + 'Given the user\'s browsing history of books, provide the title, rating,summary of reviews,brand,category(separate these with symbol \'#\') :' + ''.join(history_texts) 
    
            
            question2 = "The current book's  description is:"+description.replace("{", "") 
            question3 = ",the category is :"+cate_name.replace("{", "")+ \
            ", the brand is :"+brand_name.replace("{", "")[:30]+ \
            ", the rank is :"+rank.replace("{", "")[:30]+ \
            ", the title is :"+title.replace("{", "")[:30] + \
            ", and the price is :"+price.replace("{", "")[:30]+ \
            ", please predict the summary of review of the book the user is currently interacting with based on the above information as :"

            #", please predict whether the user will click on the current book :"
            answer = review.replace("{", "")+"}."
            
            
          
            
            if is_train:
                input_id =[151644]+ auto_tokenizer(question1).input_ids[:300] +auto_tokenizer(question2).input_ids[:40]+auto_tokenizer(question3).input_ids+ auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids[:70] + [151645]
            else:
                input_id = [151644]+auto_tokenizer(question1).input_ids[:300] +auto_tokenizer(question2).input_ids[:40]+auto_tokenizer(question3).input_ids + auto_tokenizer('{').input_ids
            

           
            if len(input_id)>max_padding_len:
                print()
            assert len(input_id)<=max_padding_len
            input_id += [auto_tokenizer.pad_token_id] * (max_padding_len - len(input_id))

       

            input_ids.append(input_id[:max_padding_len])
            
            
            print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
                  sum(total_label) / len(total_label))
        
            
            input_ids = torch.tensor(input_ids, dtype=torch.int)
            return full_data,prompts,input_ids,input_ids.ne(auto_tokenizer.pad_token_id)







if __name__ == '__main__':
    random.seed(12345)
    #DATA_DIR = '../data/'
    DATA_DIR = '/mnt/nasSave/user/asjdiasosd/data/amz/qwen_1.5_7_amz_user_item_w_brand_cate_to_review_100w/'
    # DATA_SET_NAME = 'amz'
    DATA_SET_NAME = ''
    # DATA_SET_NAME = 'ml-1m'
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
    train_ctr,train_prompts,train_input_ids,train_attention_masks= generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'],datamap,True)
    print('generating ctr test dataset')
    test_ctr,test_prompts,test_input_ids,test_attention_masks = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'],datamap,False)
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

    # print('generating reranking train dataset')
    # train_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
    #                                     train_test_split['train'], item_set)
    # print('generating reranking test dataset')
    # test_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
    #                                    train_test_split['test'], item_set)
    # print('save reranking data')
    # save_pickle(train_rerank, PROCESSED_DIR + '/rerank.train')
    # save_pickle(test_rerank, PROCESSED_DIR + '/rerank.test')
    # train_rerank, test_rerank = None, None

  

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

    #print('generating item prompt')
    #prompt = generate_item_prompt(item2attribute, datamap, DATA_SET_NAME)
    # print('generating history prompt')
    # hist_prompt = generate_hist_prompt(sequence_data, item2attribute, datamap,
    #                                    train_test_split['lm_hist_idx'], DATA_SET_NAME)
    print('save prompt data')
    save_json(train_prompts, PROCESSED_DIR + '/train_prompts.json')
    save_json(test_prompts, PROCESSED_DIR + '/test_prompts.json')
    torch.save(train_input_ids, PROCESSED_DIR +'/train_input_ids.pt')
    torch.save(train_attention_masks, PROCESSED_DIR +'/train_attention_masks.pt')
    torch.save(test_input_ids, PROCESSED_DIR +'/test_input_ids.pt')
    torch.save(test_attention_masks, PROCESSED_DIR +'/test_attention_masks.pt')
    # print('save rlmrec data')
    # torch.save(train_rlmRec_user_emb,PROCESSED_DIR +'/train_rlmRec_user_emb.pt')
    # torch.save(train_rlmRec_item_emb,PROCESSED_DIR +'/train_rlmRec_item_emb.pt')
    # torch.save(test_rlmRec_user_emb,PROCESSED_DIR +'/test_rlmRec_user_emb.pt')
    # torch.save(test_rlmRec_item_emb,PROCESSED_DIR +'/test_rlmRec_test_emb.pt')

 
    # print('generating ctr test dataset left padding')
    # test_ctr,test_prompts,test_input_ids,test_attention_masks = generate_ctr_test_left_padding_data(sequence_data, train_test_split['lm_hist_idx'],
    #                              train_test_split['test'],datamap,False)





  


    # print('save prompt data')

    # torch.save(test_input_ids, PROCESSED_DIR +'/test_input_ids_leftPadding.pt')
    # torch.save(test_attention_masks, PROCESSED_DIR +'/test_attention_masks_leftPadding.pt')

    prompt = None

