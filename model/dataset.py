import torch
import torch.utils.data as Data
import pickle
from utils import load_json, load_pickle
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import pandas
import numpy
class AmzDataset(Data.Dataset):
    def __init__(self, args,data_path, set='train', max_hist_len=5,  llm_emb=False,data=None):

        self.max_hist_len = max_hist_len

        self.llm_emb = llm_emb

        self.set = set
  

        self.data = load_pickle(data_path + f'/ctr.{set}')
 
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']

        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']

        if llm_emb:
            #self.llm_token = load_json(data_path+f'/{set}_prompts.json')
            self.input_ids = torch.load(data_path + f'/{set}_input_ids.pt')
            self.attention_masks = torch.load(data_path + f'/{set}_attention_masks.pt')
           
       

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
 
        uid, seq_idx, lb = self.data[_id]
        item_seq, rating_seq,_,_= self.sequential_data[str(uid)]
        iid = item_seq[seq_idx]
        hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
        attri_id = self.item2attribution[str(iid)]
        # attri_id = attri_id[:2]
        hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
        # hist_attri_seq = [hist_attri[:2] for hist_attri in hist_attri_seq]
        out_dict = {
            'user_id':self.id2user[str(uid)],
            'item_id':self.id2item[str(iid)],
            'iid': torch.tensor(iid).long(),
            'aid': torch.tensor(attri_id).long(),
            'lb': torch.tensor(lb).long(),
            'hist_iid_seq': torch.tensor(hist_item_seq).long(),
            'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
            'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
            'hist_seq_len': torch.tensor(hist_seq_len).long()
        }
        if self.llm_emb:
            input_ids=self.input_ids[_id]
            attention_masks=self.attention_masks[_id]
            out_dict['input_ids'] = torch.tensor(input_ids).long()
            out_dict['attention_masks'] = torch.tensor(attention_masks).long()
        return out_dict


