import torch
import torch.nn as nn
import os
import glob
import re
from random import randint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, RobertaConfig, AutoModelForSequenceClassification, RobertaTokenizer, logging





class CustomDataset(Dataset):
    def __init__(self):
        super(type(self), self).__init__()
        self.codes = glob.glob('/home/code_similarity/train_code/**/*.cpp', recursive=True)

    def __getitem__(self, i):
        prob = (i // 500) + 1
        num = (i % 500) + 1

        with open(f'/home/code_similarity/train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code = ''.join(fs.readlines())
            code = self._preprocess_code(code)

        num = num+randint(1,499)
        num = num - 500 if num > 500 else num
        with open(f'/home/code_similarity/train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code_same = ''.join(fs.readlines())

        prob = prob+randint(1,499)
        prob = prob - 500 if prob > 500 else prob
        with open(f'/home/code_similarity/train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code_diff = ''.join(fs.readlines())
        

        return code, code_same, code_diff

    def __len__(self):
        return 500*500
    
    def _preprocess_code(self, code):
        code = re.sub('//.+', '', code)
        code = re.sub('#include.+', '', code)
        code = re.sub('using.+', '', code)
        code = re.sub('\n+', '\n', code)
        return code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 2e-5
weight_decay = 1e-2
logging.set_verbosity_error()

def main():
    tokenizer:RobertaTokenizer = RobertaTokenizer.from_pretrained("neulab/codebert-cpp", truncation_side='left')
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained("neulab/codebert-cpp").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataloader = DataLoader(CustomDataset(), batch_size=batch_size, shuffle=True, drop_last=True)

    labels_same = torch.tensor([[1, 0]], dtype=torch.float32).repeat(batch_size,1).to(device)
    labels_diff = torch.tensor([[0, 1]], dtype=torch.float32).repeat(batch_size,1).to(device)

    history = {'loss':[], 'acc_same':[], 'acc_diff':[]}
    for i, (code, code_same, code_diff) in enumerate(dataloader):
        total_loss = 0.0
        encode = tokenizer(code, code_same, padding=True, truncation=True, return_tensors='pt').to(device)
        encode['labels'] = labels_same

        output = model(**encode)
        print()
        loss = output['loss']
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history['acc_same'].append(torch.sum(1 - torch.argmax(output['logits'], 1)).item())

        encode = tokenizer(code, code_diff, padding=True, truncation=True, return_tensors='pt').to(device)
        encode['labels'] = labels_diff

        output = model(**encode)
        loss = output['loss']
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['acc_diff'].append(torch.sum(torch.argmax(output['logits'], 1)).item())

        history['loss'].append(loss.item())

        if i+1 % 100 == 0:
            print(history['loss'][i], history['acc_same'][i], history['acc_diff'][i])

        

if __name__ == '__main__':
    main()