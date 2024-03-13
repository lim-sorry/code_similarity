import argparse
import csv
import torch
import os
import re
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
    def __init__(self):
        super(type(self), self).__init__()
        self.idx = np.arange(500*500)
        np.random.shuffle(self.idx)
        self.ref = np.random.randint(0,499*499,500*500)

    def __getitem__(self, i):
        prob = (self.idx[i] // 500) + 1
        num = (self.idx[i] % 500) + 1

        with open(f'train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code = ''.join(fs.readlines())
            code = preprocess_code(code)

        num += self.ref[i] % 499 + 1
        num = num - 500 if num > 500 else num
        with open(f'train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code_same = ''.join(fs.readlines())
            code_same = preprocess_code(code_same)

        prob += self.ref[i] // 499 + 1
        prob = prob - 500 if prob > 500 else prob
        with open(f'train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code_diff = ''.join(fs.readlines())
            code_diff = preprocess_code(code_diff)

        return code, code_same, code_diff

    def __len__(self):
        return 500*500


def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(opt.model_path, truncation_side='left')
    model = AutoModelForSequenceClassification.from_pretrained(opt.model_path).to(device)
    pt_path = opt.model_path.split('/')[-1]+'.pt'

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    epoch = 1
    if os.path.exists(pt_path):
        checkpoint = torch.load(pt_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('checkpoint load...')

    model.train()
    for ep in range(opt.epochs+1):
        # Synchronize training sample while epochs
        dataset = TrainDataset()
        if ep < epoch: continue

        print(f'epoch {ep} train start...')
        
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)

        n_div, train_loss, train_acc = 0, 0.0, 0.0
        for i, (code_org, code_same, code_diff) in enumerate(dataloader):
            encode = tokenizer(code_org, code_same, padding=True, truncation=True, return_tensors='pt').to(device)
            encode['labels'] = torch.tensor([[1, 0]], dtype=torch.float32).repeat(encode['input_ids'].size(0),1).to(device)

            output = model(**encode)
            loss = output['loss']
            train_loss += loss.item()
            train_acc += torch.sum(1 - torch.argmax(output['logits'], 1)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            encode = tokenizer(code_org, code_diff, padding=True, truncation=True, return_tensors='pt').to(device)
            encode['labels'] = torch.tensor([[0, 1]], dtype=torch.float32).repeat(encode['input_ids'].size(0),1).to(device)

            output = model(**encode)
            loss = output['loss']
            train_loss += loss.item()
            train_acc += torch.sum(torch.argmax(output['logits'], 1)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_div += encode['input_ids'].size(0)

            if (i+1) % 1000 == 0 or (i+1)==len(dataloader):
                train_loss /= n_div
                train_acc /= n_div*2
                print(f"iter {i+1}: train_loss: {train_loss:3.5f}, train_acc: {train_acc:.3f}")
                n_div, train_loss, train_acc = 0, 0.0, 0.0

        checkpoint = {
            'epoch': ep+1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, pt_path)
        print('checkpoint saved...')

            
def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='MickyMike/graphcodebert-c')
    # parser.add_argument('--model_path', type=str, default='neulab/codebert-c')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)

    
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

    main(opt)
