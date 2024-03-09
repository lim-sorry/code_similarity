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

    def __getitem__(self, i):
        prob = (i // 500) + 1
        num = (i % 500) + 1

        with open(f'train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code = ''.join(fs.readlines())
            code = preprocess_code(code)

        num += random.randint(1,499)
        num = num - 500 if num > 500 else num
        with open(f'train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code_same = ''.join(fs.readlines())
            code_same = preprocess_code(code_same)

        prob += random.randint(1,499)
        prob = prob - 500 if prob > 500 else prob
        with open(f'train_code/problem{prob:03d}/problem{prob:03d}_{num}.cpp') as fs:
            code_diff = ''.join(fs.readlines())
            code_diff = preprocess_code(code_diff)

        return code, code_same, code_diff

    def __len__(self):
        return 500*500


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

    tokenizer = AutoTokenizer.from_pretrained('neulab/codebert-cpp', truncation_side='left')
    model = AutoModelForSequenceClassification.from_pretrained(opt.model_path).to(device)

    if opt.mode.lower() == 'train':
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        dataloader = DataLoader(TrainDataset(), batch_size=opt.batch_size, shuffle=True, drop_last=False)

        if os.path.exists(opt.pt_path) and True:
            checkpoint = torch.load(opt.pt_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('checkpoint loaded...')

        n_div, train_loss, train_acc = 0, 0.0, 0.0
        for i, (code_org, code_same, code_diff) in enumerate(dataloader):
            optimizer.zero_grad()
            encode = tokenizer(code_org, code_same, padding=True, truncation=True, return_tensors='pt').to(device)
            encode['labels'] = torch.tensor([[1, 0]], dtype=torch.float32).repeat(encode['input_ids'].size(0),1).to(device)

            output = model(**encode)
            loss = output['loss']
            train_loss += loss.item()
            train_acc += torch.sum(1 - torch.argmax(output['logits'], 1)).item()

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            encode = tokenizer(code_org, code_diff, padding=True, truncation=True, return_tensors='pt').to(device)
            encode['labels'] = torch.tensor([[0, 1]], dtype=torch.float32).repeat(encode['input_ids'].size(0),1).to(device)

            output = model(**encode)
            loss = output['loss']
            train_loss += loss.item()
            train_acc += torch.sum(torch.argmax(output['logits'], 1)).item()

            loss.backward()
            optimizer.step()

            n_div += encode['input_ids'].size(0)

            if (i+1) % 1000 == 0 or (i+1)==len(dataloader):
                train_loss /= n_div
                train_acc /= n_div*2
                print(f"iter {i+1}: train_loss: {train_loss:3.5f}, train_acc: {train_acc:3.2f}")
                n_div, train_loss, train_acc = 0, 0.0, 0.0
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, opt.pt_path)
            
    if opt.mode.lower() == 'test':
        assert os.path.exists(opt.pt_path)
        model.eval()
        checkpoint = torch.load(opt.pt_path)
        model.load_state_dict(checkpoint['model'])
        print('test model loaded...')

        dataloader = DataLoader(TestDataset(), batch_size=opt.batch_size, shuffle=False, drop_last=False)

        with torch.no_grad():
            tq = tqdm.tqdm(dataloader)
            output_list = []
            for code1, code2 in tq:
                encode = tokenizer(code1, code2, padding=True, truncation=True, return_tensors='pt').to(device)

                output = model(**encode)
                output_list.append(1 - torch.argmax(output['logits'], 1))

            output = torch.cat(output_list, dim=0).cpu().detach()
            index = np.array([f"TEST_{i:06d}" for i in range(output.size(0))])
            output = np.hstack((index.reshape(-1,1), output.reshape(-1,1)))
            df = pd.DataFrame(data=output, columns=["pair_id", "similar"])
            df.to_csv(f"{opt.model_name}_submission.csv", index=False, encoding='utf-8')

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='neulab/codebert-cpp')
    parser.add_argument('--model_name', type=str, default='codebert-cpp')
    parser.add_argument('--pt_path', type=str, default='codebert-cpp.pt')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
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

    main(opt)
