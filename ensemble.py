import argparse
import csv
import torch
import os
import re
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

def preprocess_code(code):
    code = re.sub('\/\/.+', '', code)
    code = re.sub('#include.+', '', code)
    code = re.sub('using namespace.+', '', code)
    code = re.sub('\/\*.*\*\/', '', code, flags=re.DOTALL)
    code = re.sub('\n+', '\n', code)
    return code

class TrainDataset(Dataset):
    def __init__(self) -> None:
        super(type(self), self).__init__()
        self.df = pd.read_csv('sample_train.csv')

    def __getitem__(self, i):
        code1 = preprocess_code(self.df.iloc[i]['code1'])
        code2 = preprocess_code(self.df.iloc[i]['code2'])
        label = self.df.iloc[i]['similar']
        return code1, code2, label

    def __len__(self):
        return len(self.df)
    
class TestDataset(Dataset):
    def __init__(self) -> None:
        super(type(self), self).__init__()
        self.df = pd.read_csv('test.csv')

    def __getitem__(self, i):
        code1 = preprocess_code(self.df.iloc[i]['code1'])
        code2 = preprocess_code(self.df.iloc[i]['code2'])
        return code1, code2

    def __len__(self):
        return len(self.df)

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_0 = AutoTokenizer.from_pretrained(opt.model_path_0, truncation_side='left')
    model_0 = AutoModelForSequenceClassification.from_pretrained(opt.model_path_0).to(device)

    tokenizer_1 = AutoTokenizer.from_pretrained(opt.model_path_1, truncation_side='left')
    model_1 = AutoModelForSequenceClassification.from_pretrained(opt.model_path_1).to(device)
            
    assert os.path.exists(opt.pt_path_0) and os.path.exists(opt.pt_path_1)
    model_0.eval()
    model_1.eval()
    checkpoint = torch.load(opt.pt_path_0)
    model_0.load_state_dict(checkpoint['model'])
    checkpoint = torch.load(opt.pt_path_1)
    model_1.load_state_dict(checkpoint['model'])
    print('test model loaded...')

    dataloader = DataLoader(TestDataset(), batch_size=opt.batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        tq = tqdm.tqdm(dataloader)
        output_list = []
        for i, (code1, code2) in enumerate(tq):
            encode = tokenizer_0(code1, code2, padding=True, truncation=True, return_tensors='pt').to(device)
            output_0 = model_0(**encode)
            encode = tokenizer_1(code1, code2, padding=True, truncation=True, return_tensors='pt').to(device)
            output_1 = model_1(**encode)
            output_list.append(torch.cat([output_0['logits'], output_1['logits']], 1))

            if (i+1)%100==1 or (i+1)==len(tq):
                output = torch.cat(output_list, dim=0).cpu().detach()
                index = np.array([f"TEST_{i:06d}" for i in range(output.size(0))])
                output = np.hstack((index.reshape(-1,1), output.reshape(-1,4)))
                df = pd.DataFrame(data=output, columns=["pair_id", "logits_0_0", "logits_0_1", "logits_1_0", "logits_1_1"])
                df.to_csv(f"logits.csv", index=False, encoding='utf-8')


def test_weight_sum(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_0 = AutoTokenizer.from_pretrained(opt.model_path_0, truncation_side='left')
    model_0 = AutoModelForSequenceClassification.from_pretrained(opt.model_path_0).to(device)

    tokenizer_1 = AutoTokenizer.from_pretrained(opt.model_path_1, truncation_side='left')
    model_1 = AutoModelForSequenceClassification.from_pretrained(opt.model_path_1).to(device)
            
    assert os.path.exists(opt.pt_path_0) and os.path.exists(opt.pt_path_1)
    model_0.eval()
    model_1.eval()
    checkpoint = torch.load(opt.pt_path_0)
    model_0.load_state_dict(checkpoint['model'])
    checkpoint = torch.load(opt.pt_path_1)
    model_1.load_state_dict(checkpoint['model'])
    print('test model loaded...')

    dataloader = DataLoader(TrainDataset(), batch_size=opt.batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        tq = tqdm.tqdm(dataloader)
        acc_list = np.zeros((101))
        n_div = 0
        for i, (code1, code2, label) in enumerate(tq):
            encode = tokenizer_0(code1, code2, padding=True, truncation=True, return_tensors='pt').to(device)
            output_0 = model_0(**encode)
            encode = tokenizer_1(code1, code2, padding=True, truncation=True, return_tensors='pt').to(device)
            output_1 = model_1(**encode)
            for j, w in enumerate(range(101)):
                w = w / 100
                w_sum = output_0['logits'] * w + output_1['logits'] * (1-w)
                pred = 1 - torch.argmax(w_sum, 1)
                acc_list[j] += torch.sum(pred == label.to(device))
            n_div += output_0['logits'].size(0)
            tq.set_postfix_str(f'optimal w:{np.argmax(acc_list) * 0.01:.2f}')


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path_0', type=str, default='neulab/codebert-cpp')
    parser.add_argument('--model_path_1', type=str, default='neulab/codebert-c')
    parser.add_argument('--pt_path_0', type=str, default='codebert-cpp.pt')
    parser.add_argument('--pt_path_1', type=str, default='codebert-c.pt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=43)

    
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_arg()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.set_verbosity_error()

    # main(opt)
    test_weight_sum(opt)
