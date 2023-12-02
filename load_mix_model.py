# import logging
# from pathlib import Path
# import argparse
import torch as t
import torch.nn as nn
# import yaml

# import process
# import quan
# import util
# from model.my_model import create_model
# import os
# from torchvision._internally_replaced_utils import load_state_dict_from_url
# p = argparse.ArgumentParser(description='Learned Step Size Quantization')
# p.add_argument('--arch', type=str, default='mix_s_vgg16', help='config file')
# p.add_argument('--pre_trained', type=bool, default=True, help='config file')
# # p.dataloader.add_argument('dataset', type=str, default='imagenet', help='dataset')
# args = p.parse_args()
# # setattr(args.dataloader, 'dataset', 'imagenet')
# model = create_model(args)
# pre_location = "/home/dingchenchen/test_quant/lsq_sparse/out/VGG16_ImageNet_bmix_s0_20230227-211651/VGG16_ImageNet_bmix_s0_best.pth.tar"

# def load_model_from_fp(model, fp_dict):
#     state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth',progress=progress)
#     model.load_state_dict(state_dict)
#     return model
# model.cuda()
# model = t.nn.DataParallel(model)
def load_mix_from_pretrained(model, pre_location:str):
    state_dict = t.load(pre_location,map_location='cpu')['state_dict']
    print(state_dict.keys())
    # model.load_state_dict(state_dict)
    for name, param in model.named_parameters():
        if ('alpha' in name):
            pass
        elif ("op_list.0" in name):
            print(name)
            name_org = name.replace("op_list.0.", "")
            param.data.copy_(state_dict[name_org])
        elif ("op_list.1" in name):
            name_org = name.replace("op_list.1.", "")
            param.data.copy_(state_dict[name_org])
        elif ("op_list.2" in name):
            name_org = name.replace("op_list.2.", "")
            param.data.copy_(state_dict[name_org])
def load_mix_from_t2pretrained(model, pre_location:str):
    state_dict = t.load(pre_location,map_location='cpu')['state_dict']
    print(state_dict.keys())
    # model.load_state_dict(state_dict)
    for name, param in model.named_parameters():
        if ('alpha' in name):
            pass
        elif ("op_list.0" in name):
            name_org = '_orig_mod.'+name.replace("op_list.0.", "")
            try:
                param.data.copy_(state_dict[name_org])    
            
            except(RuntimeError):
                print(name+" cannot pre-load")
        elif ("op_list.1" in name):
            name_org = '_orig_mod.'+name.replace("op_list.1.", "")
            try:
                param.data.copy_(state_dict[name_org])
            except(RuntimeError):
                print(name+" cannot pre-load")    
        elif ("op_list.2" in name):
            name_org = '_orig_mod.'+name.replace("op_list.2.", "")
            try:
                param.data.copy_(state_dict[name_org])
            except(RuntimeError):
                print(name+" cannot pre-load") 
    # return model

# load_mix_from_pretrained(model, pre_location)
# model.module.init_weight()

import copy
def get_module_alpha(model):
    state_dict = copy.deepcopy(model.module.state_dict())
    alpha_dict = {k: nn.functional.softmax(v.detach()) for k, v in state_dict.items() if "alpha" in k}
    return alpha_dict
def get_alpha(model):
    state_dict = copy.deepcopy(model.state_dict())
    alpha_dict = {k: nn.functional.softmax(v.detach()) for k, v in state_dict.items() if "alpha" in k}
    return alpha_dict
    # keys=alpha_dict.keys()
    # append_cnt=0
    # blk_cnt=0
    # RANK_LIST = [
    #     (4, 4, 4),
    #     (6, 6, 6),
    #     (8, 8, 8),
    # ]
    # if "module" in list(keys)[0]:
    #     append_cnt=1
    # out_dict={}
    # out_index={}
    # out_dict.update({str(blk_cnt): []})
    # out_index.update({str(blk_cnt): []})
    # for key in keys:
    #     if(int(key.split(".")[1+append_cnt])!=blk_cnt):
    #         blk_cnt=int(key.split(".")[1+append_cnt])
    #         out_dict.update({str(blk_cnt):[]})
    #         out_index.update({str(blk_cnt): []})

    #     out_dict[str(blk_cnt)].append(RANK_LIST[int(alpha_dict[key].argmax())])
    #     out_index[str(blk_cnt)].append(int(alpha_dict[key].argmax()))
    # print(out_dict)
    # print(alpha_dict)
    # return out_index,out_dict
def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path
import os
def make_alpha_log(output_dir:str,alpha_dict):
    alpha_folder = os.path.join(output_dir,'alpha_log')
    if not os.path.exists(alpha_folder):
        os.mkdir(alpha_folder)
    for k in alpha_dict:
        create_csv(alpha_folder,k+'.csv',['epoch','alpha_0','alpha_1','alpha_2'])


def save_state(alpha_folder_dir, *entries, filename='alpha.csv'):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(alpha_folder_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))

def update_alpha_log(output_dir:str,epoch,alpha_dict):
    alpha_folder = os.path.join(output_dir,'alpha_log')
    for k in alpha_dict:
        save_state(alpha_folder,epoch,float(alpha_dict[k][0]),float(alpha_dict[k][1]),float(alpha_dict[k][2]),filename=k+'.csv')


def get_bitwidth_from_alpha_dict(alpha_dict):
    BITS = [2,4,8]
    # bit_dict = {}
    bit_list = []
    for k in alpha_dict.keys():
        m,index = t.max(alpha_dict[k].cpu(),dim=0)
        # name = k.replace(".alpha","")
        # bit_dict[name] = BITS[index]
        bit_list.append(BITS[index])
    # return bit_dict,bit_list
    return bit_list
def get_sparsity_from_alpha_dict(alpha_dict):
    SPARSITY = [0.875,0.75,0.5]
    # bit_dict = {}
    sparsity_list = []
    for k in alpha_dict.keys():
        m,index = t.max(alpha_dict[k].cpu(),dim=0)
        # name = k.replace(".alpha","")
        # bit_dict[name] = BITS[index]
        sparsity_list.append(SPARSITY[index])
    # return bit_dict,bit_list
    return sparsity_list


def load_mix_from_t2pretrained_expect_head(model, pre_location:str):
    state_dict = t.load(pre_location,map_location='cpu')['state_dict']
    print(state_dict.keys())
    # model.load_state_dict(state_dict)
    for name, param in model.named_parameters():
        if ('alpha' in name or 'head' in name):
            pass
        elif ("op_list.0" in name):
            name_org = '_orig_mod.'+name.replace("op_list.0.", "")
            try:
                param.data.copy_(state_dict[name_org])    
            
            except(RuntimeError):
                print(name+" cannot pre-load")
        elif ("op_list.1" in name):
            name_org = '_orig_mod.'+name.replace("op_list.1.", "")
            try:
                param.data.copy_(state_dict[name_org])
            except(RuntimeError):
                print(name+" cannot pre-load")    
        elif ("op_list.2" in name):
            name_org = '_orig_mod.'+name.replace("op_list.2.", "")
            try:
                param.data.copy_(state_dict[name_org])
            except(RuntimeError):
                print(name+" cannot pre-load") 
        else:
            try:
                param.data.copy_(state_dict[name])
            except(RuntimeError):
                print(name+" cannot pre-load")


def load_mix_from_t2pretrained(model, pre_location:str):
    state_dict = t.load(pre_location,map_location='cpu')['state_dict']
    print(state_dict.keys())
    # model.load_state_dict(state_dict)
    for name, param in model.named_parameters():
        if ('alpha' in name):
            pass
        elif ("op_list.0" in name):
            name_org = '_orig_mod.'+name.replace("op_list.0.", "")
            try:
                param.data.copy_(state_dict[name_org])    
            
            except(RuntimeError):
                print(name+" cannot pre-load")
        elif ("op_list.1" in name):
            name_org = '_orig_mod.'+name.replace("op_list.1.", "")
            try:
                param.data.copy_(state_dict[name_org])
            except(RuntimeError):
                print(name+" cannot pre-load")    
        elif ("op_list.2" in name):
            name_org = '_orig_mod.'+name.replace("op_list.2.", "")
            try:
                param.data.copy_(state_dict[name_org])
            except(RuntimeError):
                print(name+" cannot pre-load") 
        else:
            try:
                param.data.copy_(state_dict[name])
            except(RuntimeError):
                print(name+" cannot pre-load")


def load_mix_from_t2pretrained_mix(model, pre_location:str):
    state_dict = t.load(pre_location,map_location='cpu')['state_dict']
    print(state_dict.keys())
    # model.load_state_dict(state_dict)
    for name, param in model.named_parameters():
        if ('alpha' in name):
            # pass
            name_org = '_orig_mod.'+name
            param.data.copy_(state_dict[name_org])
        # elif ("op_list.0" in name):
        #     name_org = '_orig_mod.'+name.replace("op_list.0.", "")
        #     try:
        #         param.data.copy_(state_dict[name_org])    
            
        #     except(RuntimeError):
        #         print(name+" cannot pre-load")
        # elif ("op_list.1" in name):
        #     name_org = '_orig_mod.'+name.replace("op_list.1.", "")
        #     try:
        #         param.data.copy_(state_dict[name_org])
        #     except(RuntimeError):
        #         print(name+" cannot pre-load")    
        # elif ("op_list.2" in name):
        #     name_org = '_orig_mod.'+name.replace("op_list.2.", "")
        #     try:
        #         param.data.copy_(state_dict[name_org])
        #     except(RuntimeError):
        #         print(name+" cannot pre-load") 
        else:
            try:
                name_org = '_orig_mod.'+name
                param.data.copy_(state_dict[name_org])
            except(RuntimeError):
                print(name+" cannot pre-load")
