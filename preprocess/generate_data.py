import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING
from alps.pytorch.modelhub.hub_layer import TorchHubLayer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
# model_name = 'bigscience/bloomz-1b7'
#model_name = 'Qwen/Qwen1.5-0.5B'
# model_name = 'bert-base-uncased'
model_name = '/mnt/nas/alps/modelhub/layer/Qwen.Qwen1.5-7B/qwen1.5-7B'
max_padding_len = 512
rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10
bert =False
os.chdir('/ossfs/workspace/Open-World-Knowledge-Augmented-Recommendation-main/preprocess')
# rlmRec_path = '/ossfs/workspace/RLMRec-main/data/amazon/'
# rlmRec_amz_user_id2raw = rlmRec_path + 'amazon_user.json'
# rlmRec_amz_item_id2raw = rlmRec_path + 'amazon_item.json'

# rlmRec_amz_user_pkl = rlmRec_path + 'usr_emb_np.pkl'
# rlmRec_amz_item_pkl = rlmRec_path + 'itm_emb_np.pkl'

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

    item_have = 0
    item_no = 0
    if '7B' not in model_name:
        auto_tokenizer = TorchHubLayer.restore_from_modelhub(
            name=model_name,
            hf_loader=AutoTokenizer,
            from_huggingface=True,
            hf_args=None,
            hf_kwargs={"trust_remote_code": True},
        )
    else:
        auto_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            #让bloomz来猜title
            
            # question1 = "I will provide you with some interaction information about the user and the current book, as well as information about the book itself." \
            # "The user's summary of reviews on the current book is :"+summary.replace("{", "")
            # question2 = ", the reviews are :"+review.replace("{", "")
            # question3 = "; the current book's category is :"+cate_name.replace("{", "")+ \
            # ", and the brand is :"+brand_name.replace("{", "")+ \
            # ", please predict the title of the book the user is currently interacting with based on the above information as :"
            # answer = title.replace("{", "")+"}."

            #让bloomz来猜title,仅仅输入user review侧信息
            
            # question1 = "I will provide you with some interaction information about the user and the current book." \
            # "The user's summary of reviews on the current book is :"+summary.replace("{", "")
            # question2 = ", the reviews are :"+review.replace("{", "")
            # question3 = "; please predict the title of the book the user is currently interacting with based on the above information as :"
            # answer = title.replace("{", "")+"}."

            #让bloomz来猜title,仅仅输入item侧信息
            
            #question1 = "I will provide you with some information about the current book." 
            #question2 = "The current book's  description is:"+description.replace("{", "") 
            #question3 = ",the category is :"+cate_name.replace("{", "")+ \
            #", the brand is :"+brand_name.replace("{", "")+ \
            #", the rank is :"+rank.replace("{", "")+ \
            #", and the price is :"+price.replace("{", "")[:30]+ \
            #", please predict the title of the book the user is currently interacting with based on the above information as :"
            #answer = title.replace("{", "")+"}."


            #让bloomz来猜category
            # question1 = "I will provide you with some interaction information about the user and the current book, as well as information about the book itself." \
            # "The user's summary of reviews on the current book is :"+summary.replace("{", "")
            # question2 = ", the reviews are :"+review.replace("{", "")
            # question3 = "; the current book's title is :"+title.replace("{", "")+ \
            # ", and the brand is :"+brand_name.replace("{", "")+ \
            # ", please predict the category of the book the user is currently interacting with based on the above information as :"
            # answer = cate_name.replace("{", "")+"}."

            
            #仅有历史序列信息，来推理分类
            
            # question1 = "I will provide you with some some users' history of browsing books and their ratings." 
            # question2 = "The current book's  description is:"+description.replace("{", "") 
            # question3 = ",Please predict which book the user wants to browse next is :"
            # answer = title.replace("{", "")+"}."
                
            # prompt = "please predict the category of the book the user is next interacting with based on the above information as "
            
            # question1 = 'Given user\'s book rating history: ' 
            # question2 = ''.join(history_texts) 
            # question3 = prompt
            # answer = "cate_name"+"}."
            #history_texts = None
            #加入用户浏览历史和当前book的特征来预测review

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
            
            # if label == 1:
            #     answer="Yes}."
            # else:
            #     answer="No}."
            
            #加入用户浏览历史和当前book的特征来预测summay

            # question1 = "I will provide you with some  users' history of browsing books and current book information." + 'Given user\'s book rating and summary of review history: ' + ''.join(history_texts) 
    
            
            # question2 = "The current book's  description is:"+description.replace("{", "") 
            # question3 = ",the category is :"+cate_name.replace("{", "")+ \
            # ", the brand is :"+brand_name.replace("{", "")[:30]+ \
            # ", the rank is :"+rank.replace("{", "")[:30]+ \
            # ", the title is :"+title.replace("{", "")[:30] + \
            # ", and the price is :"+price.replace("{", "")[:30]+ \
            # ", please predict the summary of review of the book the user is currently interacting with based on the above information as :"

            # answer = summary.replace("{", "")+"}."
            #加入用户浏览历史和当前book的特征来预测label

            # question1 = "I will provide you with some  users' history of browsing books and current book information." + 'Given the user\'s browsing history of books, provide the title, rating, and summary of reviews (separate these three with commas) :' + ''.join(history_texts) 
    
            
            # question2 = "The current book's  description is:"+description.replace("{", "") 
            # question3 = ",the category is :"+cate_name.replace("{", "")+ \
            # ", the brand is :"+brand_name.replace("{", "")[:30]+ \
            # ", the rank is :"+rank.replace("{", "")+ \
            # ", the title is :"+title.replace("{", "")[:30] + \
            # ", and the price is :"+price.replace("{", "")[:30]+ \
            # ", please predict whether the user will click on the current book :"
            # if label == 1:
            #     answer="Yes}."
            # else:
            #     answer="No}."
            

             #仅加入当前book的特征来预测review

            # question1 = "I will provide you with some  current book information." 
    
            
            # question2 = "The current book's  description is:"+description.replace("{", "") 
            # question3 = ",the category is :"+cate_name.replace("{", "")+ \
            # ", the brand is :"+brand_name.replace("{", "")[:30]+ \
            # ", the rank is :"+rank.replace("{", "")+ \
            # ", and the price is :"+price.replace("{", "")[:30]+ \
            # ", please predict the review of the book the user is currently interacting with based on the above information as :"
            # answer = review.replace("{", "")+"}."

            #加入用户浏览历史来预测review

            # question1 = "I will provide you with some  users' history of browsing books." + 'Given the user\'s browsing history of books, provide the title, rating, and summary of reviews (separate these three with commas) :' + ''.join(history_texts) 
    
            
            # question2 = ""
            # question3 = ", please predict the review of the book the user is currently interacting with based on the above information as :"
            # answer = review.replace("{", "")+"}."

            # question1 = 'Given user\'s book rating history: ' 
            # question2 = ''.join(history_texts) 
            # question3 = prompt
            # answer = "cate_name"+"}."
            #让bloomz来猜category
            #代码掠过
          
            prompts.append(question1+question2+question3+'{'+answer)
            #prompts.append(question+answer)  
            # if bert:
            #     question1_input_ids = auto_tokenizer(question1).input_ids[:100] if len(auto_tokenizer(question1).input_ids) >=101 else auto_tokenizer(question1).input_ids[0:-1]
            #     question2_input_ids = auto_tokenizer(question2).input_ids[1:300] if len(auto_tokenizer(question2).input_ids) >=301 else auto_tokenizer(question2).input_ids[1:-1]
            #     question3_input_ids =  auto_tokenizer(question3).input_ids[1:-1]
            #     if is_train:
            #         input_id = question1_input_ids+question2_input_ids+question3_input_ids+ auto_tokenizer('{').input_ids[1:-1] + auto_tokenizer(answer).input_ids[1:]
            #     else:
            #         input_id = question1_input_ids+question2_input_ids+question3_input_ids+ auto_tokenizer('{').input_ids[1:-1] 
            #[94]代表{
            # if is_train:
            #     input_id = auto_tokenizer(question1).input_ids[:100] +auto_tokenizer(question2).input_ids[:300]+auto_tokenizer(question3).input_ids+ auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids + [2]
            # else:
            #     input_id = auto_tokenizer(question1).input_ids[:100] +auto_tokenizer(question2).input_ids[:300]+auto_tokenizer(question3).input_ids + auto_tokenizer('{').input_ids
            
            if is_train:
                input_id =[151644]+ auto_tokenizer(question1).input_ids[:300] +auto_tokenizer(question2).input_ids[:40]+auto_tokenizer(question3).input_ids+ auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids[:70] + [151645]
            else:
                input_id = [151644]+auto_tokenizer(question1).input_ids[:300] +auto_tokenizer(question2).input_ids[:40]+auto_tokenizer(question3).input_ids + auto_tokenizer('{').input_ids
            

            # if is_train:
            #     input_id = auto_tokenizer(question1).input_ids[:100] +auto_tokenizer(question3).input_ids+ auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids + [2]
            # else:
            #     input_id = auto_tokenizer(question1).input_ids[:100] +auto_tokenizer(question3).input_ids + auto_tokenizer('{').input_ids
            
            # if is_train:
            #     input_id = auto_tokenizer(question).input_ids + auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids + [2]
            # else:
            #     input_id = auto_tokenizer(question).input_ids  + auto_tokenizer('{').input_ids

            #target = len(auto_tokenizer(question).input_ids) * [auto_tokenizer.pad_token_id] + [94] + auto_tokenizer(answer).input_ids+ [2]
            if len(input_id)>max_padding_len:
                print()
            assert len(input_id)<=max_padding_len
            input_id += [auto_tokenizer.pad_token_id] * (max_padding_len - len(input_id))

            # if(len(torch.where(input_id == 94)[1])!=1):
            #     print(question)
            #     print("error!!!")
            
            # if(len(torch.where(input_id == 2)[1])!=1):
            #     print(question)
            #     print("no </s> error!!!")

            input_ids.append(input_id[:max_padding_len])
            
            #rlmRec的方法
            # try:
            #     rlmRec_user_emb.append(rlmRec_user_raw2emb[id2user[str(uid)]])
            #     user_have+=1
            # except KeyError:
            #     rlmRec_user_emb.append(None)
            #     user_no+=1

            # try:
            #     rlmRec_item_emb.append(rlmRec_item_raw2emb[id2item[str(iid)]])
            #     item_have+=1
            # except KeyError:
            #     rlmRec_user_emb.append(None)
            #     item_no+=1
    # print(no_des)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label))

    # print('user_have',user_have)
    # print('user_no',user_no)
    # print('item_have',item_have)
    # print('item_no',item_no)
    # print(full_data[:5])
    # print(prompts[:5])
  
    
    # 使用tokenizer处理句子，并将结果保存为tensor
    # tokenized_data111 = [
    #     auto_tokenizer(
    #         sentence,
    #         return_tensors="pt",
    #     )
    #     for sentence in prompts
    # ]
    # for index in range(len(tokenized_data111)):
    #     print(len(tokenized_data111[index]))
    # 使用tokenizer处理句子，并将结果保存为tensor
    # tokenized_data = [
    #     auto_tokenizer(
    #         sentence,
    #         max_length=max_padding_len,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     for sentence in prompts
    # ]
    #tokenized_data = [tokenizer.encode_plus(json_obj["sentence"], add_special_tokens=True, max_length=128, pad_to_max_length=True, return_tensors="pt") for json_obj in json_data]
    # 提取input_ids和attention_mask，准备作为模型的输入
    # input_ids = torch.cat([entry["input_ids"] for entry in tokenized_data], dim=0)
    # attention_masks = torch.cat([entry["attention_mask"] for entry in tokenized_data], dim=0)
    #return full_data,prompts,input_ids,attention_masks,rlmRec_user_emb,rlmRec_item_emb
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    return full_data,prompts,input_ids,input_ids.ne(auto_tokenizer.pad_token_id)

def generate_ctr_test_left_padding_data(sequence_data, lm_hist_idx, uid_set,datamap,is_train=False):
    # print(list(lm_hist_idx.values())[:10])
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    id2user = datamap['id2user']




    full_data = []
    total_label = []
    prompts= [] 


    user_have = 0
    user_no = 0

    item_have = 0
    item_no = 0
    auto_tokenizer = TorchHubLayer.restore_from_modelhub(
        name=model_name,
        hf_loader=AutoTokenizer,
        from_huggingface=True,
        hf_args=None,
        hf_kwargs={"trust_remote_code": True},
    )
    auto_tokenizer.padding_side = "left"
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
                tmp = '{}, {} stars,{}; '.format(itemid2title[str(iid)], int(rating),summary)
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
            


            
           
    
            question1 = "I will provide you with some  users' history of browsing books and current book information." + 'Given the user\'s browsing history of books, provide the title, rating,summary of reviews(separate these three with commas) :' + ''.join(history_texts) 
            question2 = "The current book's  description is:"+description.replace("{", "") 
            question3 = ",the category is :"+cate_name.replace("{", "")+ \
            ", the brand is :"+brand_name.replace("{", "")[:30]+ \
            ", the rank is :"+rank.replace("{", "")[:30]+ \
            ", and the price is :"+price.replace("{", "")[:30]+ \
            ", the title is :"+title.replace("{", "")[:30] + \
            ", please predict the review of the book the user is currently interacting with based on the above information as :"

            answer = review.replace("{", "")+"}."
            

            

            
            
            if is_train:
                input_id =[151644]+ auto_tokenizer(question1).input_ids[:400] +auto_tokenizer(question2).input_ids[:40]+auto_tokenizer(question3).input_ids+ auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids[:70] + [151645]
            else:
                input_id = [151644]+auto_tokenizer(question1).input_ids[:400] +auto_tokenizer(question2).input_ids[:40]+auto_tokenizer(question3).input_ids + auto_tokenizer('{').input_ids
            # if is_train:
            #     input_id = auto_tokenizer(question1).input_ids[:100] +auto_tokenizer(question3).input_ids+ auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids + [2]
            # else:
            #     input_id = auto_tokenizer(question1).input_ids[:100] +auto_tokenizer(question3).input_ids + auto_tokenizer('{').input_ids
            
            # if is_train:
            #     input_id = auto_tokenizer(question).input_ids + auto_tokenizer('{').input_ids + auto_tokenizer(answer).input_ids + [2]
            # else:
            #     input_id = auto_tokenizer(question).input_ids  + auto_tokenizer('{').input_ids

            #target = len(auto_tokenizer(question).input_ids) * [auto_tokenizer.pad_token_id] + [94] + auto_tokenizer(answer).input_ids+ [2]
            if len(input_id)>max_padding_len:
                print()
            assert len(input_id)<=max_padding_len
            new_input_id = [auto_tokenizer.pad_token_id] * (max_padding_len - len(input_id))
            new_input_id += input_id
            # if(len(torch.where(input_id == 94)[1])!=1):
            #     print(question)
            #     print("error!!!")
            
            # if(len(torch.where(input_id == 2)[1])!=1):
            #     print(question)
            #     print("no </s> error!!!")

            input_ids.append(new_input_id[:max_padding_len])
            
            #rlmRec的方法
            # try:
            #     rlmRec_user_emb.append(rlmRec_user_raw2emb[id2user[str(uid)]])
            #     user_have+=1
            # except KeyError:
            #     rlmRec_user_emb.append(None)
            #     user_no+=1

            # try:
            #     rlmRec_item_emb.append(rlmRec_item_raw2emb[id2item[str(iid)]])
            #     item_have+=1
            # except KeyError:
            #     rlmRec_user_emb.append(None)
            #     item_no+=1
    print(no_des)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label))

    # print('user_have',user_have)
    # print('user_no',user_no)
    # print('item_have',item_have)
    # print('item_no',item_no)
    # print(full_data[:5])
    # print(prompts[:5])
  
    
    # 使用tokenizer处理句子，并将结果保存为tensor
    # tokenized_data111 = [
    #     auto_tokenizer(
    #         sentence,
    #         return_tensors="pt",
    #     )
    #     for sentence in prompts
    # ]
    # for index in range(len(tokenized_data111)):
    #     print(len(tokenized_data111[index]))
    # 使用tokenizer处理句子，并将结果保存为tensor
    # tokenized_data = [
    #     auto_tokenizer(
    #         sentence,
    #         max_length=max_padding_len,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     for sentence in prompts
    # ]
    #tokenized_data = [tokenizer.encode_plus(json_obj["sentence"], add_special_tokens=True, max_length=128, pad_to_max_length=True, return_tensors="pt") for json_obj in json_data]
    # 提取input_ids和attention_mask，准备作为模型的输入
    # input_ids = torch.cat([entry["input_ids"] for entry in tokenized_data], dim=0)
    # attention_masks = torch.cat([entry["attention_mask"] for entry in tokenized_data], dim=0)
    #return full_data,prompts,input_ids,attention_masks,rlmRec_user_emb,rlmRec_item_emb
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    return full_data,prompts,input_ids,input_ids.ne(auto_tokenizer.pad_token_id)

def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    full_data = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('user num', len(uid_set), 'data num', len(full_data))
    print(full_data[:5])
    return full_data


def generate_hist_prompt(sequence_data, item2attribute, datamap, lm_hist_idx, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2user = datamap['id2user']
    if dataset_name == 'ml-1m':
        user2attribute = datamap['user2attribute']
    hist_prompts = {}
    print('item2attribute', list(item2attribute.items())[:10])
    for uid, item_rating in sequence_data.items():
        user = id2user[uid]
        item_seq, rating_seq,review_seq,summary_seq = item_rating
        cur_idx = lm_hist_idx[uid]
        hist_item_seq = item_seq[:cur_idx]
        hist_rating_seq = rating_seq[:cur_idx]
        history_texts = []
        for iid, rating in zip(hist_item_seq, hist_rating_seq):
            tmp = '"{}", {} stars; '.format(itemid2title[str(iid)], int(rating))
            history_texts.append(tmp)
        if dataset_name == 'amz':
            # prompt = 'Analyze user\'s preferences on product (consider factors like genre, functionality, quality, ' \
            #          'price, design, reputation. Provide clear explanations based on ' \
            #          'relevant details from the user\'s product viewing history and other pertinent factors.'
            # hist_prompts[uid] = 'Given user\'s product rating history: ' + ''.join(history_texts) + prompt
            prompt = 'Analyze user\'s preferences on books about factors like genre, author, writing style, theme, ' \
                     'setting, length and complexity, time period, literary quality, critical acclaim (Provide ' \
                     'clear explanations based on relevant details from the user\'s book viewing history and other ' \
                     'pertinent factors.'
            hist_prompts[user] = 'Given user\'s book rating history: ' + ''.join(history_texts) + prompt
            print()
        elif dataset_name == 'ml-1m':
            gender, age, occupation = user2attribute[uid]
            user_text = 'Given a {} user who is aged {} and {}, this user\'s movie viewing history over time' \
                        ' is listed below. '.format(GENDER_MAPPING[gender], AGE_MAPPING[age],
                                                    OCCUPATION_MAPPING[occupation])
            question = 'Analyze user\'s preferences on movies (consider factors like genre, director/actors, time ' \
                       'period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, ' \
                       'and soundtrack). Provide clear explanations based on relevant details from the user\'s movie ' \
                       'viewing history and other pertinent factors.'
            hist_prompts[user] = user_text + ''.join(history_texts) + question
        else:
            raise NotImplementedError
    print('data num', len(hist_prompts))
    print(list(hist_prompts.items())[0])
    return hist_prompts


def generate_item_prompt(item2attribute, datamap, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    item_prompts = {}
    for iid, title in itemid2title.items():
        item = id2item[iid]
        if dataset_name == 'amz':
            brand, cate = item2attribute[str(iid)]
            brand_name = attrid2name[str(brand)]
            # cate_name = attrid2name[cate]
            item_prompts[item] = 'Introduce book {}, which is from brand {} and describe its attributes including but' \
                                ' not limited to genre, author, writing style, theme, setting, length and complexity, ' \
                                'time period, literary quality, critical acclaim.'.format(title, brand_name)
            # item_prompts[iid] = 'Introduce product {}, which is from brand {} and describe its attributes (including but' \
            #                     ' not limited to genre, functionality, quality, price, design, reputation).'.format(title, brand_name)
        elif dataset_name == 'ml-1m':
            item_prompts[item] = 'Introduce movie {} and describe its attributes (including but not limited to genre, ' \
                                'director/cast, country, character, plot/theme, mood/tone, critical ' \
                                'acclaim/award, production quality, and soundtrack).'.format(title)
        else:
            raise NotImplementedError
    print('data num', len(item_prompts))
    print(list(item_prompts.items())[0])
    return item_prompts


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

    # print('generating ctr train dataset')
    # train_ctr,train_prompts,train_input_ids,train_attention_masks,train_rlmRec_user_emb,train_rlmRec_item_emb = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
    #                               train_test_split['train'],datamap,True)
    # print('generating ctr test dataset')
    # test_ctr,test_prompts,test_input_ids,test_attention_masks,test_rlmRec_user_emb,test_rlmRec_item_emb= generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
    #                              train_test_split['test'],datamap,False)
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

