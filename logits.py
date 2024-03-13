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
    model_name = opt.model_path.split('/')[-1]
    pt_path = model_name + '.pt'

    tokenizer = AutoTokenizer.from_pretrained(opt.model_path, truncation_side='left')
    model = AutoModelForSequenceClassification.from_pretrained(opt.model_path).to(device)
            
    assert os.path.exists(pt_path)
    checkpoint = torch.load(pt_path)
    model.load_state_dict(checkpoint['model'])
    print('test model loaded...')

    dataloader = DataLoader(TestDataset(), batch_size=opt.batch_size, shuffle=False, drop_last=False)

    model.eval()
    with torch.no_grad():
        tq = tqdm.tqdm(dataloader)
        output_list = []

        for i, (code1, code2) in enumerate(tq):
            encode = tokenizer(code1, code2, padding=True, truncation=True, return_tensors='pt').to(device)
            output = model(**encode)
            output_list.append(output['logits'])

            if (i+1) % 100 == 1 or (i+1)==len(dataloader):
                output = torch.cat(output_list, dim=0).cpu().detach()
                df = pd.DataFrame(data=output, columns=["P", "N"])
                df.to_csv(f"test_{model_name}_logits.csv", index=False, encoding='utf-8')

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='MickyMike/graphcodebert-c')
    parser.add_argument('--batch_size', type=int, default=256)

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_arg()

    main(opt)
