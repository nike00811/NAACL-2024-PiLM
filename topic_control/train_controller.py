import sys
sys.path.append('../src')
from method import set_seed

import torch
import torch.nn as nn
import pickle
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tqdm.notebook import tqdm
import numpy as np
import argparse
import json
import os




def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed",                 type=int,   default=20)
    parser.add_argument("--device",               type=str,   default='cuda')
    parser.add_argument("--learning_rate",        type=float, default=3e-5)
    parser.add_argument("--num_epochs",           type=int,   default=100)
    parser.add_argument("--batch_size",           type=int,   default=64)
    
    parser.add_argument("--output_dir",           type=str,   required=False)
    
    parser.add_argument("--train_dir",   type=str,   required=True)
    parser.add_argument("--eval_dir",   type=str,   required=True)
    parser.add_argument("--attribute",   type=str,   required=True)
    
    parser.add_argument("--logfile",   type=str,   default='training_log.json')
    

    
    args = parser.parse_args()
#     args = parser.parse_args([
#                               '--device', 'cuda',
#                               '--batch_size', '64',
#                               '--learning_rate', '3e-5',
#                               '--attribute', 'Space',
#                               '--num_epochs', '3',

#                               '--train_dir', 'find_parameter/M3_3-seed*',
#                               '--eval_dir', 'find_parameter/M3_3-seed*',
        
# #                               '--output_dir', 'test3',
#                              ])

    if args.output_dir is not None:
        if os.path.exists(args.output_dir):
            print('[INFO] {} already exist'.format(args.output_dir))
            assert os.path.exists(args.output_dir) ==  False
            exit()
        else:
            os.makedirs(args.output_dir)
    
    return args




class Regression_Model(nn.Module):
    def __init__(self, input_dim):
        super(Regression_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_dim),
        )

    def forward(self, x):
        input_shape = x.shape
        # [batch_size, 24, 2, 1, 16, 1, 64]

        x = x.flatten(1)
        x = self.layers(x)
        x = x.reshape(input_shape)
        return x

    @property
    def device(self):
        return next(self.parameters()).device




class loadDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
        self.dataset = []
        
        for file in tqdm(file_list, bar_format='{l_bar}{bar:50}{r_bar}'):
            try:
                with open(file, mode='rb') as fp:
                    pairs = pickle.load(fp)
                self.build(pairs)
            except Exception as e:
                print('[ERROR] {}'.format(e))
                print(file)
                print('-'*50)

    
    def build(self, pairs):
        for pair in pairs:
            length = pair[0].shape[4]
            X = pair[0]
            Y = pair[1]
            # [24, 2, 1, 16, length, 64]
            for i in range(length):
                sample = {'x': X[:, :, :, :, i:i+1, :],
                          'y': Y[:, :, :, :, i:i+1, :]}
                self.dataset.append(sample)
        
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]




def train_section(model, criterion, optimizer, train_loader):
    train_loss = 0
    for batch in tqdm(train_loader, leave=False, bar_format='{l_bar}{bar:30}{r_bar}'):
        outputs = model(batch['x'].to(args.device))
        loss = criterion(outputs, batch['y'].to(args.device))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss
    train_loss /= len(train_loader)
    return train_loss

@torch.no_grad()
def eval_section(model, eval_loader):
    eval_loss = 0
    for batch in eval_loader:
        outputs = model(batch['x'].to(args.device))
        loss = criterion(outputs, batch['y'].to(args.device))

        eval_loss += loss
    eval_loss /= len(eval_loader)
    return eval_loss




def get_file_list(data_dir):
    file_list = []
    file_path = glob.glob('{}/*.pickle'.format(data_dir))
#     file_path = file_path[:10]
    for file in file_path:
        attribute = os.path.basename(file).split('+')[0]
        if attribute == args.attribute:
            file_list.append(file)
    return file_list




if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seed(args.seed)


    train_file_list = get_file_list(args.train_dir)
    training_set = loadDataset(train_file_list)
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    
    
    eval_file_list = get_file_list(args.eval_dir)
    evaluation_set = loadDataset(eval_file_list)
    eval_loader = DataLoader(evaluation_set, batch_size=args.batch_size, shuffle=False)
    
    print('len(training_set.dataset) = {}'.format(len(training_set.dataset)))
    print('len(evaluation_set.dataset) = {}'.format(len(evaluation_set.dataset)))
    
    print('len(train_loader) = {}'.format(len(train_loader)))
    print('len(eval_loader) = {}'.format(len(eval_loader)))

    input_dim = 24 * 2 * 1 * 16 * 1 * 64
    model = Regression_Model(input_dim=input_dim)
    model.to(args.device)
#     model = nn.parallel.DataParallel(model)
#     model = nn.parallel.DistributedDataParallel(model)
    model.train()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    record = {
              'train_loss': [],
              'eval_loss': [],
             }
    
    print('start training')
    min_error = 1e20
    best_epoch = -1
    for epoch in tqdm(range(1, args.num_epochs+1), bar_format='{l_bar}{bar:30}{r_bar}', desc='training'):
        model.train()
        train_loss = train_section(model, criterion, optimizer, train_loader)
#         scheduler.step()
        model.eval()
        eval_loss = eval_section(model, eval_loader)

        record['train_loss'].append(train_loss.item())
        record['eval_loss'].append(eval_loss.item())
        
        if eval_loss < min_error:
            min_error = eval_loss
            best_epoch = epoch
            print('epoch {:3d}: training loss = {:.4f}, eval loss = {:.4f}**'.format(epoch, train_loss, eval_loss))
        
        else:
            print('epoch {:3d}: training loss = {:.4f}, eval loss = {:.4f}'.format(epoch, train_loss, eval_loss))

    
    if args.output_dir is not None:
        torch.save(model, '{}/final_model.pt'.format(args.output_dir))

        record['args'] = vars(args)
        with open('{}/{}'.format(args.output_dir, args.logfile), mode='w', encoding='utf-8') as fp:
            json.dump(obj=record, fp=fp, indent=4)
