'''
-*- coding: utf-8 -*-
@File  : models.py
'''
import numpy as np
import torch
import torch.nn as nn
from layers import AttentionPoolingLayer, MLP, CrossNet, ConvertNet, CIN, MultiHeadSelfAttention, \
    SqueezeExtractionLayer, BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, InterestExtractor, \
    InterestEvolving, SLAttention
from layers import Phi_function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss,BCELoss
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM,AutoModelForCausalLM,AutoModelForPreTraining,BertModel,BertTokenizer
from alps.pytorch.modelhub.hub_layer import TorchHubLayer
import os



device = 'cuda'


def tau_function(x):
    return torch.where(x > 0, torch.exp(x), torch.zeros_like(x))


def attention_score(x, temperature=1.0):
    return tau_function(x / temperature) / (tau_function(x / temperature).sum(dim=1, keepdim=True) + 1e-20)


class BaseModel(nn.Module):
    def __init__(self, args, dataset):
        super(BaseModel, self).__init__()

        self.args = args
        self.device = args.device
        self.kd_temperature = args.kd_temperature



        self.gpus = args.gpus
        self.item_num = dataset.item_num
        self.attr_num = dataset.attr_num
        self.attr_fnum = dataset.attr_ft_num
        # self.attr_fnum = 2
        self.rating_num = dataset.rating_num

        self.max_hist_len = args.max_hist_len
        
        self.embed_dim = args.embed_dim
        self.final_mlp_arch = args.final_mlp_arch
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size

        self.output_dim = args.output_dim

        self.llm_emb = args.llm_emb
        #self.item_fnum = 1 
        self.item_fnum = 1 + self.attr_fnum
        self.hist_fnum = 2 + self.attr_fnum
        self.itm_emb_dim = self.item_fnum * self.embed_dim
        self.hist_emb_dim = self.hist_fnum * self.embed_dim


        self.item_embedding = nn.Embedding(self.item_num + 1, self.embed_dim)
        self.attr_embedding = nn.Embedding(self.attr_num + 1, self.embed_dim)
        self.rating_embedding = nn.Embedding(self.rating_num + 1, self.embed_dim)
   
        self.module_inp_dim = self.get_input_dim()
        self.field_num = self.get_field_num()

        #self.regularization_weight = []

    def gather_indexes(self, output, gather_index):
        """
        Extract the corresponding tensor from output according to the index of gather_index and return it.

        Parameters:
        output (torch.Tensor): input tensor, shape is (batch_size, seq_length, hidden_size).
        gather_index (torch.Tensor): index used to extract tensor, shape is (batch_size, num_indexes).

        Output:
        torch.Tensor: extracted tensor, shape is (batch_size, num_indexes, hidden_size).

        """
        # Convert gather_index to (batch_size, 1, 1) and expand the last dimension to match the shape of output
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])

        # Use the gather function to index according to the specified dimension and extract the corresponding tensor
        output_tensor = output.gather(dim=1, index=gather_index)

        # Remove the squeeze(1) dimension and return the extracted tensor with shape (batch_size, num_indexes, hidden_size)
        return output_tensor.squeeze(1)

    def cal_infonce_loss(self,embeds1, embeds2, all_embeds2, temp=1.0):
        # Normalize embeds1 to ensure that the L2 norm of each vector is 1, which increases stability
        normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
        # Normalized embeds2, same as above
        normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
        # Normalized all_embeds2, containing multiple embedding vectors for negative samples
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
        # Calculate the dot product between embeds1 and embeds2 and divide by the temperature parameter to calculate the numerator term
        nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
        # Compute the dot product of embeds1 and all_embeds2 to construct the denominator, then sum and take the logarithm
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
        # Calculate the contrast loss, which is the sum of the numerator and denominator
        cl_loss = (nume_term + deno_term).sum()
        return cl_loss

    def process_input(self, inp):
        #cpu running
        if self.device == 'cpu':
            device = next(self.parameters()).device
        else:
            #Single GPU running
            if len(self.gpus) ==1:
                device = f'cuda:0'
            #multiple GPUs running
            else:
                device = inp['hist_iid_seq'].device
        
        hist_item_emb = self.item_embedding(inp['hist_iid_seq'].to(device)).view(-1, self.max_hist_len, self.embed_dim)
        hist_attr_emb = self.attr_embedding(inp['hist_aid_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                 self.embed_dim * self.attr_fnum)
        hist_rating_emb = self.rating_embedding(inp['hist_rate_seq'].to(device)).view(-1, self.max_hist_len,
                                                                                      self.embed_dim)
        hist_emb = torch.cat([hist_item_emb, hist_attr_emb, hist_rating_emb], dim=-1)
        hist_len = inp['hist_seq_len'].to(device)

        
        iid_emb = self.item_embedding(inp['iid'].to(device))
        attr_emb = self.attr_embedding(inp['aid'].to(device)).view(-1, self.embed_dim * self.attr_fnum)
        #选择加入
        item_emb = torch.cat([iid_emb, attr_emb], dim=-1)
        #这里我们选择先不加入item侧其他信息
        #item_emb = iid_emb
        # item_emb = item_emb.view(-1, self.itm_emb_dim)
        labels = inp['lb'].to(device)

        if self.llm_emb:
            input_ids = inp['input_ids'].to(device)
            attention_masks = inp['attention_masks'].to(device)
            return item_emb, hist_emb, hist_len, labels,input_ids,attention_masks
        else:
            return item_emb, hist_emb, hist_len, labels
            



    def get_input_dim(self):

        return self.hist_emb_dim + self.itm_emb_dim 



    def get_field_num(self):
        return self.item_fnum  + self.hist_fnum
        #6+7

    def get_filed_input(self, inp):
        item_embedding, user_behavior, hist_len, labels = self.process_input(inp)
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)

        
        inp = torch.cat([item_embedding, user_behavior], dim=1)
        out = inp.view(-1, self.field_num, self.embed_dim)
        return out, labels

    def get_filed_input_EGRec(self, inp):
        item_embedding, user_behavior, hist_len, labels,llm_input_ids,llm_attention_mask= self.process_input(inp)
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        inp = torch.cat([item_embedding, user_behavior], dim=1)
        out = inp.view(-1, self.field_num, self.embed_dim)
        return out, labels,llm_input_ids,llm_attention_mask
    

    def get_ctr_output(self, logits, labels=None):
        
        if labels is not None:
            if self.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.output_dim)), labels.float())
            else:
                loss_fct = BCELoss()
                logits = torch.clamp(torch.sigmoid(logits), min=1e-7, max=1-1e-7)
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            
        outputs = {
            'logits': logits,
            'labels': labels,
        }
        outputs['loss'] = loss 
        return outputs

    
    def llm_module(self,llm_input_ids,llm_attention_mask):

        lm_labels = llm_input_ids.clone().detach().to(llm_input_ids.device)
        lm_labels[llm_input_ids == self.tokenizer.pad_token_id] = torch.tensor(-100).to(llm_input_ids.device)

        #Here we use '{' as a delimiter, which is the token corresponding to the current item in the paper.
        braces_id = self.tokenizer('{').input_ids[0]
        # Keep the value of the position corresponding to ‘{’ in llm_input_ids in llm_hiddens_mask, and set the values ​​of other positions to 0
        llm_hiddens_mask = torch.where(llm_input_ids == braces_id, llm_attention_mask, torch.tensor(0).to(llm_input_ids.device)) 

        llm_hiddens_index, _ = torch.max(llm_hiddens_mask, dim=-1)
        
        llm_hiddens_index = torch.where(llm_input_ids == braces_id)[1].view(llm_input_ids.shape[0], -1).max(dim=1).values

        #Set other non-generated parts to -100 to avoid participating in the generation loss calculation
        for i in range(lm_labels.size(0)): 
            X_label = llm_hiddens_index[i].item()       
            lm_labels[i, :X_label] = -100  
        assert self.llm_model.device == llm_input_ids.device ==llm_attention_mask.device
        llm_outputs = self.llm_model(
            input_ids=llm_input_ids,
            attention_mask=llm_attention_mask,
            output_hidden_states=True,
            labels=lm_labels,
        )

        # len(llm_hiddens)=25, llm_hiddens[-1].shape=[bsz, padding_len, 1024]
        llm_hiddens = llm_outputs['hidden_states']
        llm_hiddens_last_layer = llm_hiddens[-1]

        # find last pos where ids=94, 94 means '{'
        # llm_hiddens_mask=[bs, padding_length], llm_hiddens_index.shape=[bs]
        # Create a tensor llm_hiddens_mask with a shape of (N, L), where N is the number of samples of llm_hiddens_last_layer, L is the sequence length of llm_hiddens_last_layer, and the value of each element is the corresponding index value
        llm_hiddens_mask = torch.arange(llm_hiddens_last_layer.shape[1]).unsqueeze(0).repeat(
            llm_hiddens_last_layer.shape[0], 1).to(llm_input_ids.device)  
        
        loss_generation = llm_outputs[0]

        llm_hiddens_output = self.gather_indexes(llm_hiddens_last_layer, llm_hiddens_index)

  
        llm_fe = self.llm_linear(llm_hiddens_output)
        
        return loss_generation,llm_hiddens_output

class DCN(BaseModel):
    '''
    DCNv1
    '''
    def __init__(self, args, mode, dataset):
        super(DCN, self).__init__(args, dataset)
        self.deep_arch = args.dcn_deep_arch
        self.cross_net = CrossNet(self.module_inp_dim, args.dcn_cross_num, mode)
        self.deep_net = MLP(self.deep_arch, self.module_inp_dim, self.dropout)
        final_inp_dim = self.module_inp_dim + self.deep_arch[-1]
        self.final_mlp = MLP(self.final_mlp_arch, final_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)


    def forward(self, inp):
        '''
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        '''
        item_embedding, user_behavior, hist_len, labels = self.process_input(inp)

        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)

        inp = torch.cat([item_embedding, user_behavior], dim=1)

        deep_part = self.deep_net(inp)
        cross_part = self.cross_net(inp)

        final_inp = torch.cat([deep_part, cross_part], dim=1)
        mlp_out = self.final_mlp(final_inp)
        logits = self.final_fc(mlp_out)
        outputs = self.get_ctr_output(logits, labels)
        return outputs

class WideDeep(BaseModel):
    def __init__(self, args, dataset):
        super(WideDeep, self).__init__(args, dataset)

        #初始化wide侧部分
        self.fm_first_iid_emb = nn.Embedding(self.item_num + 1, 1)
        self.fm_first_aid_emb = nn.Embedding(self.attr_num + 1, 1)

        # 初始化DNN部分
        self.deep_part = MLP(args.dnn_deep_arch, self.module_inp_dim, self.dropout)
        self.dnn_fc_out = nn.Linear(args.dnn_deep_arch[-1], 1)

            
    def forward(self, inp):

        item_embedding, user_behavior, hist_len, labels = self.process_input(inp)

        # 将用户行为的嵌入取均值，以便和商品嵌入进行拼接
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        
        #wide侧
        iid_first = self.fm_first_iid_emb(inp['iid'].to(device)).view(-1, 1)
        aid_first = self.fm_first_aid_emb(inp['aid'].to(device)).view(-1, self.attr_fnum)
        wide_logit = torch.sum(torch.cat([iid_first, aid_first], dim=1), dim=1).view(-1, 1)

        # 根据是否有额外特征来决定输入DNN的特征
      
      
        dnn_inp = torch.cat([item_embedding, user_behavior], dim=1)

        # DNN部分的处理
        deep_out = self.deep_part(dnn_inp)

        logits = wide_logit+self.dnn_fc_out(deep_out)  # [bs, 1]
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class WideDeepEGRec(BaseModel):
    def __init__(self, args, dataset):
        super(WideDeepEGRec, self).__init__(args, dataset)
        llm_dim = args.llm_input_dim

        #Initialize the wide side part
        self.fm_first_iid_emb = nn.Embedding(self.item_num + 1, 1)
        self.fm_first_aid_emb = nn.Embedding(self.attr_num + 1, 1)
        
        # Initialize the DNN part
        self.deep_part = MLP(args.widedeep_deep_arch, self.module_inp_dim+llm_dim, self.dropout)
        self.dnn_fc_out = nn.Linear(args.widedeep_deep_arch[-1], 1)
        # Initialize the LLM part
        self.llm_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float32,
                ).to(device)
        auto_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        hidden_dim = 1024
        auto_tokenizer.padding_side = "right"
        self.tokenizer = auto_tokenizer
        #Mapping of llm modules
        self.llm_linear = nn.Linear(hidden_dim, llm_dim, bias=False).to(device)
        #Mapping of CL modules
        self.llm_to_ctr_mlp = nn.Sequential(
            nn.Linear(hidden_dim, (hidden_dim + 192) // 2),
            nn.LeakyReLU(),
            nn.Linear((hidden_dim +192) // 2, 192) )

    def forward(self, inp):

        item_embedding, user_behavior, hist_len, labels,llm_input_ids,llm_attention_mask = self.process_input(inp)

        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)

        dnn_inp = torch.cat([item_embedding, user_behavior], dim=1)

        
        #wide
        iid_first = self.fm_first_iid_emb(inp['iid']).view(-1, 1)
        aid_first = self.fm_first_aid_emb(inp['aid']).view(-1, self.attr_fnum)
        wide_logit = torch.sum(torch.cat([iid_first, aid_first], dim=1), dim=1).view(-1, 1)

        #LLM module

        loss_generation,llm_hiddens_output =self.llm_module(llm_input_ids,llm_attention_mask)

        llm_fe = self.llm_linear(llm_hiddens_output)


        inp_concat = torch.cat((dnn_inp, llm_fe), -1)

        align_input = inp_concat[:,:192]
        #CL module 
        #Only for itemid and item_attri, a total of 32*(1+6)=192 dimensions
        llm_hiddens_output_toCtr = self.llm_to_ctr_mlp(llm_hiddens_output)
        loss_align = self.cal_infonce_loss(align_input,llm_hiddens_output_toCtr,llm_hiddens_output_toCtr,self.kd_temperature)
    
        # DNN processing
        deep_out = self.deep_part(inp_concat)
        logits = wide_logit+self.dnn_fc_out(deep_out)  # [bs, 1]
        outputs = self.get_ctr_output(logits, labels)
        return outputs,loss_generation,loss_align


class DCNEGRec(BaseModel):
    '''
    DCNv1 or DCNv2 are determined by the variable mode
    '''
    def __init__(self, args, mode, dataset):
        super(DCNEGRec, self).__init__(args, dataset)
        
        llm_dim=args.llm_input_dim
        # Initialize the cross part
        self.deep_arch = args.dcn_deep_arch
        self.cross_net = CrossNet(self.module_inp_dim, args.dcn_cross_num, mode)
        # Initialize the DNN part
        self.deep_net = MLP(self.deep_arch, self.module_inp_dim+llm_dim, self.dropout)
        final_inp_dim = self.module_inp_dim + self.deep_arch[-1]
        self.final_mlp = MLP(self.final_mlp_arch, final_inp_dim, self.dropout)
        self.final_fc = nn.Linear(self.final_mlp_arch[-1], 1)
        # Initialize the LLM part
        self.llm_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float32,
                ).to(device)
        auto_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        hidden_dim = 1024
        auto_tokenizer.padding_side = "right"
        self.tokenizer = auto_tokenizer
        #Mapping of llm modules
        self.llm_linear = nn.Linear(hidden_dim, llm_dim, bias=False).to(device)
        #Mapping of CL modules
        self.llm_to_ctr_mlp = nn.Sequential(
            nn.Linear(hidden_dim, (hidden_dim + 192) // 2),
            nn.LeakyReLU(),
            nn.Linear((hidden_dim +192) // 2, 192) )

        
        #添加正则化
    def forward(self, inp):
        '''
            :param behaviors (bs, hist_len, hist_fnum)
            :param item_ft (bs, itm_fnum)
            :param user_ft (bs, usr_fnum)
            :return score (bs)
        '''
        item_embedding, user_behavior, hist_len, labels ,llm_input_ids,llm_attention_mask= self.process_input(inp)

        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
      
        inp = torch.cat([item_embedding, user_behavior], dim=1)

        #LLM module

        loss_generation,llm_hiddens_output =self.llm_module(llm_input_ids,llm_attention_mask)

        llm_fe = self.llm_linear(llm_hiddens_output)


        inp_concat = torch.cat((inp, llm_fe), -1)
        #仅针对itemid和item_attri，一共32*（1+6）=192个维度
        align_input = inp_concat[:,:192]

        #CL module 
        #Only for itemid and item_attri, a total of 32*(1+6)=192 dimensions
        llm_hiddens_output_toCtr = self.llm_to_ctr_mlp(llm_hiddens_output)
        loss_align = self.cal_infonce_loss(align_input,llm_hiddens_output_toCtr,llm_hiddens_output_toCtr,self.kd_temperature)
            

        deep_part = self.deep_net(inp_concat)
        cross_part = self.cross_net(inp)

        final_inp = torch.cat([deep_part, cross_part], dim=1)
        mlp_out = self.final_mlp(final_inp)
        logits = self.final_fc(mlp_out)
        outputs = self.get_ctr_output(logits, labels)

        return outputs,loss_generation,loss_align





class DNN(BaseModel):
    def __init__(self, args, dataset):
        super(DNN, self).__init__(args, dataset)
        # 初始化DNN部分
        self.deep_part = MLP(args.dnn_deep_arch, self.module_inp_dim, self.dropout)
        self.dnn_fc_out = nn.Linear(args.dnn_deep_arch[-1], 1)

    def forward(self, inp):
        item_embedding, user_behavior, hist_len, labels = self.process_input(inp)

        # 将用户行为的嵌入取均值，以便和商品嵌入进行拼接
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        
   

        dnn_inp = torch.cat([item_embedding, user_behavior], dim=1)

        # DNN部分的处理
        deep_out = self.deep_part(dnn_inp)

        logits = self.dnn_fc_out(deep_out)  # [bs, 1]
        outputs = self.get_ctr_output(logits, labels)
        return outputs




class DNNEGRec(BaseModel):
    def __init__(self, args, dataset):
        super(DNNEGRec, self).__init__(args, dataset)

        llm_dim = args.llm_input_dim

        # Initialize the DNN part
        self.deep_part = MLP(args.dnn_deep_arch, self.module_inp_dim+llm_dim, self.dropout)
        self.dnn_fc_out = nn.Linear(args.dnn_deep_arch[-1], 1)
        

        # Initialize the LLM part

        self.llm_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float32,
                ).to(device)
        auto_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        hidden_dim = 1024
        auto_tokenizer.padding_side = "right"
        self.tokenizer = auto_tokenizer
        #Mapping of llm modules
        self.llm_linear = nn.Linear(hidden_dim, llm_dim, bias=False).to(device)
        #Mapping of CL modules
        self.llm_to_ctr_mlp = nn.Sequential(
            nn.Linear(hidden_dim, (hidden_dim + 192) // 2),
            nn.LeakyReLU(),
            nn.Linear((hidden_dim +192) // 2, 192) )


    def forward(self, inp):
        item_embedding, user_behavior, hist_len, labels,llm_input_ids,llm_attention_mask = self.process_input(inp)

        
        user_behavior = torch.mean(user_behavior, dim=1).view(-1, self.hist_emb_dim)
        
 
        dnn_inp = torch.cat([item_embedding, user_behavior], dim=1)
        
        #LLM module

        loss_generation,llm_hiddens_output =self.llm_module(llm_input_ids,llm_attention_mask)

        llm_fe = self.llm_linear(llm_hiddens_output)

        inp_concat = torch.cat((dnn_inp, llm_fe), -1)
        align_input = inp_concat[:,:192]
        #CL module 
        #Only for itemid and item_attri, a total of 32*(1+6)=192 dimensions
        llm_hiddens_output_toCtr = self.llm_to_ctr_mlp(llm_hiddens_output)
        loss_align = self.cal_infonce_loss(align_input,llm_hiddens_output_toCtr,llm_hiddens_output_toCtr,self.kd_temperature)
    
        deep_out = self.deep_part(inp_concat)
        logits = self.dnn_fc_out(deep_out)  # [bs, 1]
        outputs = self.get_ctr_output(logits, labels)
        return outputs,loss_generation,loss_align



class xDeepFM(BaseModel):
    def __init__(self, args, dataset):
        super(xDeepFM, self).__init__(args, dataset)
        input_dim = self.field_num * args.embed_dim
        cin_layer_units = args.cin_layer_units
        self.cin = CIN(self.field_num, cin_layer_units)
        self.dnn = MLP(args.final_mlp_arch, input_dim, self.dropout)
        final_dim = sum(cin_layer_units) + args.final_mlp_arch[-1]
        self.final_fc = nn.Linear(final_dim, args.output_dim)

    def forward(self, inp):
        inp, labels = self.get_filed_input(inp)

        final_vec = self.cin(inp)
        dnn_vec = self.dnn(inp.flatten(start_dim=1))
        final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        logits = self.final_fc(final_vec)
        outputs = self.get_ctr_output(logits, labels)
        return outputs


class xDeepFMEGRec(BaseModel):
    def __init__(self, args, dataset):
        super(xDeepFMEGRec, self).__init__(args, dataset)
        llm_dim=args.llm_input_dim
        input_dim = self.field_num * args.embed_dim
        cin_layer_units = args.cin_layer_units
        self.cin = CIN(self.field_num, cin_layer_units)
        self.dnn = MLP(args.final_mlp_arch, input_dim+llm_dim, self.dropout)
        final_dim = sum(cin_layer_units) + args.final_mlp_arch[-1]
        self.final_fc = nn.Linear(final_dim, args.output_dim)
        # Initialize the LLM part
        self.llm_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float32,
                ).to(device)
        auto_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        hidden_dim = 1024
        auto_tokenizer.padding_side = "right"
        self.tokenizer = auto_tokenizer
        #Mapping of llm modules
        self.llm_linear = nn.Linear(hidden_dim, llm_dim, bias=False).to(device)
        #Mapping of CL modules
        self.llm_to_ctr_mlp = nn.Sequential(
            nn.Linear(hidden_dim, (hidden_dim + 192) // 2),
            nn.LeakyReLU(),
            nn.Linear((hidden_dim +192) // 2, 192) )

    def forward(self, inp):
        inp, labels,llm_input_ids,llm_attention_mask = self.get_filed_input_EGRec(inp)
        
        #LLM module

        loss_generation,llm_hiddens_output =self.llm_module(llm_input_ids,llm_attention_mask)

        llm_fe = self.llm_linear(llm_hiddens_output)


        #xDeepFM 
        inp_concat=torch.cat((inp.flatten(start_dim=1), llm_fe), -1)
        final_vec = self.cin(inp)
        dnn_vec = self.dnn(inp_concat)
        #CL module
        align_input = inp_concat[:,:192]
        llm_hiddens_output_toCtr = self.llm_to_ctr_mlp(llm_hiddens_output)
        loss_align = self.cal_infonce_loss(align_input,llm_hiddens_output_toCtr,llm_hiddens_output_toCtr,self.kd_temperature)

        final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        logits = self.final_fc(final_vec)
        outputs = self.get_ctr_output(logits, labels)
        return outputs,loss_generation,loss_align
    

