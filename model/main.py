'''
-*- coding: utf-8 -*-
@File  : main_ctr.py
'''
# 1.python
import os
import time

import numpy as np
import json
import argparse
import datetime
# 2.pytorch
import torch
import torch.utils.data as Data
# 3.sklearn
from sklearn.metrics import roc_auc_score, log_loss

from utils import load_parse_from_json, setup_seed, load_data, weight_init, str2list
from models import *
from dataset import AmzDataset
from optimization import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup


def eval(args,model, test_loader):
    model.eval()
    losses = []
    preds = []
    labels = []
    t = time.time()
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            if args.llm_emb:
                outputs,loss_generation,loss_align = model(data)
                loss = outputs['loss']+loss_generation*args.llm_loss_weight+loss_align*args.llm_align_loss_weight
            else:
                outputs = model(data)
                loss = outputs['loss']
            loss=torch.sum(loss)
            logits = outputs['logits']
            preds.extend(logits.detach().cpu().tolist())
            labels.extend(outputs['labels'].detach().cpu().tolist())
            losses.append(loss.item())
    eval_time = time.time() - t
    auc = roc_auc_score(y_true=labels, y_score=preds)
    ll = log_loss(y_true=labels, y_pred=preds)
    return auc, ll, np.mean(losses), eval_time


def test(args):
    model = torch.load(args.reload_path)
    test_set = AmzDataset(args,args.data_dir, 'test', args.max_hist_len, args.augment, args.llm_emb)
    test_loader = Data.DataLoader(args,dataset=test_set, batch_size=args.batch_size, shuffle=False)
    print('Test data size:', len(test_set))
    auc, ll, loss, eval_time = eval(model, test_loader)
    print("test loss: %.5f, test time: %.5f, auc: %.5f, logloss: %.5f" % (loss, eval_time, auc, ll))


def load_model(args, dataset):
    algo = args.algo
    device = args.device
    if algo == 'DCN':
        model = DCN(args, 'v1', dataset).to(device)
    elif algo == 'DCNv2':
        model = DCN(args, 'v2', dataset).to(device)
    elif algo == 'xDeepFM':
        model = xDeepFM(args, dataset).to(device)
    elif algo == 'WideDeep':
        model = xDeepFM(args, dataset).to(device)
    elif algo == 'DNN' :
        model = DNN(args, dataset).to(device)
    elif algo == 'DNNEGRec' :
        model = DNNEGRec(args, dataset).to(device)
    elif algo == 'DCNEGRec' :
        model = DCNEGRec(args, 'v1',dataset).to(device)
    elif algo == 'DCNv2EGRec' :
        model = DCNEGRec(args, 'v2',dataset).to(device)
    elif algo == 'WideDeepEGRec' :
        model = WideDeepEGRec(args, dataset).to(device)
    elif algo == 'xDeepFMEGRec' :
        model = xDeepFMEGRec(args,dataset).to(device)
    else:
        print('No Such Model')
        exit()
    model.apply(weight_init)
    return model


def get_optimizer(args, model, train_data_num):
    no_decay = ['bias', 'LayerNorm.weight']
    # no_decay = []
    named_params = [(k, v) for k, v in model.named_parameters()]
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    beta1, beta2 = args.adam_betas.split(',')
    beta1, beta2 = float(beta1), float(beta2)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon,
                          betas=(beta1, beta2))
    t_total = int(train_data_num * args.epoch_num)
    t_warmup = int(t_total * args.warmup_ratio)
    if args.lr_sched.lower() == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=t_warmup,
                                                    num_training_steps=t_total)
    elif args.lr_sched.lower() == 'const':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=t_warmup)
    else:
        raise NotImplementedError
    return optimizer, scheduler


def train(args):

    train_set = AmzDataset(args,args.data_dir, 'train',  args.max_hist_len, args.llm_emb)
    test_set = AmzDataset(args,args.data_dir, 'test',  args.max_hist_len,args.llm_emb)

    
    print('Train data size:', len(train_set), 'Test data size:', len(test_set))

    model = load_model(args, test_set)

    optimizer, scheduler = get_optimizer(args, model, len(train_set))

    save_path = os.path.join(args.save_dir, args.algo + f'{args.print_model_name}_loss_{args.llm_loss_weight}_guesstitle_8gpus.pt')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    
    batch_size = args.batch_size
    if args.gpus and len(args.gpus)>1:
            print('parallel running on these gpus:', args.gpus)
            model = torch.nn.DataParallel(model, device_ids=args.gpus)
            batch_size *= len(args.gpus)  # input `batch_size` is batch_size per gpu
    else:
        print(args.device)
    
    train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    best_auc = 0
    global_step = 0
    patience = 0

    for epoch in range(args.epoch_num):
        t = time.time()
        train_loss = []
        model.train()
        for _, data in enumerate(train_loader):

            if args.llm_emb:
                outputs,loss_generation,loss_align = model(data)
                all_loss = outputs['loss']
                print(f'step:{global_step}')
                print(f'loss:{all_loss},loss_generation:{loss_generation},loss_align:{loss_align}')
                loss = outputs['loss']+loss_generation*args.llm_loss_weight+loss_align*args.llm_align_loss_weight
            else:
                outputs = model(data)
                all_loss = outputs['loss']
                print(f'step:{global_step}')
                print(f'loss:{all_loss}')
                loss = outputs['loss']
            optimizer.zero_grad()

            loss=torch.sum(loss)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            global_step += 1
        train_time = time.time() - t
        torch.save(model, save_path)
        eval_auc, eval_ll, eval_loss, eval_time = eval(args,model, test_loader)
        print("EPOCH %d  STEP %d train loss: %.5f, train time: %.5f, test loss: %.5f, test time: %.5f, auc: %.5f, "
              "logloss: %.5f" % (epoch, global_step, np.mean(train_loss), train_time, eval_loss,
                                 eval_time, eval_auc, eval_ll))
        
        



def parse_args():
    # The optimal learning rate for each backbone
    model2lr_dict = {
                    'DCN':0.001,
                    'DCNv2':0.001,
                    'DNN':0.0001,
                    'WideDeep':0.0001,
                    'xDeepFM':0.0001}
    which_model='xDeepFM'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/amz/proc_data/',help='your data dir')
    parser.add_argument('--save_dir', default='../model/amz/',help='model save dir')
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='', help='setting dir')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--output_dim', default=1, type=int, help='output_dim')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--epoch_num', default=1, type=int, help='epochs of each iteration.') #
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--lr', default=model2lr_dict[which_model], type=float, help='learning rate')  #1e-3
    parser.add_argument('--weight_decay', default=0, type=float, help='l2 loss scale')  #0
    parser.add_argument('--adam_betas', default='0.9,0.999', type=str, help='beta1 and beta2 for Adam optimizer.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=str, help='Epsilon for Adam optimizer.')
    parser.add_argument('--lr_sched', default='cosine', type=str, help='Type of LR schedule method')
    parser.add_argument('--warmup_ratio', default=0.0, type=float, help='inear warmup over warmup_ratio if warmup_steps not set')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')  #0
    parser.add_argument('--convert_dropout', default=0.0, type=float, help='dropout rate of convert module')  # 0
    parser.add_argument('--grad_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--patience', default=3, type=int, help='The patience for early stop')
    parser.add_argument('--algo', default=f'{which_model}EGRec', type=str, help='model name')
    parser.add_argument('--llm_emb', default='true', type=str, help='whether to use EGRec feature')
    parser.add_argument('--model_name_or_path', default='/mnt/nas/alps/modelhub/layer/Qwen.Qwen1.5-0.5B/main/20240228142240/hf_model', type=str, help='your llm path')
    parser.add_argument('--convert_type', default='HEA', type=str, help='type of convert module')
    parser.add_argument('--max_hist_len', default=5, type=int, help='the max length of user history')
    parser.add_argument('--embed_dim', default=32, type=int, help='size of embedding')  #32
    parser.add_argument('--final_mlp_arch', default='200,80', type=str2list, help='size of final layer')
    parser.add_argument('--llm_loss_weight', default=0.1, type=float, help='loss for LLM module')
    parser.add_argument('--llm_align_loss_weight', default=0.01, type=float, help='loss for CL module')
    parser.add_argument('--kd_temperature', default=1.0, type=float, help='loss for load balance in expert')
    parser.add_argument('--hidden_size', default=64, type=int, help='size of hidden size')
    parser.add_argument('--rnn_dp', default=0.0, type=float, help='dropout rate in RNN')
    parser.add_argument('--dnn_deep_arch', default='200,80', type=str2list, help='size of deep net in DNN')
    parser.add_argument('--dcn_deep_arch', default='200,80', type=str2list, help='size of deep net in DCN')
    parser.add_argument('--widedeep_deep_arch', default='200,80', type=str2list, help='size of deep net in WideDeep')
    parser.add_argument('--dcn_cross_num', default=3, type=int, help='num of cross layer in DCN')
    parser.add_argument('--cin_layer_units', default='50,50', type=str2list, help='CIN layer in xDeepFM')
    parser.add_argument('--print_model_name', default=f'{which_model}AlignJointqwen_1.5_0.5_amz_user_item_brand_cate_side_to_review_real_next_token_title_130w_align_len_5', type=str, help='print  model name')
    parser.add_argument('--gpus', type=str, default='0,1', help='GPUs to use')
    parser.add_argument('--llm_input_dim', default=16, type=int, help='llm passes through a linear layer and is concatenated to the dimension of ctr')
    args, _ = parser.parse_known_args()

    args.llm_emb =  True if args.llm_emb.lower() == 'true' else False

    gpu_list = args.gpus.split(',')
    args.gpus = [int(gpu) for gpu in gpu_list]
    print('max hist len', args.max_hist_len)
    print(args.print_model_name)

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args.timestamp)
    if args.setting_path:
        args = load_parse_from_json(args, args.setting_path)
    setup_seed(args.seed)

    print('parameters', args)
    train(args)

