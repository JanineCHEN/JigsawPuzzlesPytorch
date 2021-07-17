import argparse
import torch
import os
import numpy as np
import random
import pandas as pd
import shutil
import torch.nn as nn
from Dataset import FoldDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from JigsawNet import JigsawNet
from tqdm import tqdm


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def save_checkpoint(net, path, global_step, accuracy=None, info=''):
    try:
        os.makedirs(path)
        print('Created checkpoint directory')
    except OSError:
        pass

    if accuracy:
        checkpoint_name = f'CP_%d_%.4f%s.pth' % (global_step, accuracy, info)
    else :
        checkpoint_name = f'CP_{global_step}{info}.pth'
    torch.save(net.state_dict(),
               os.path.join(path, checkpoint_name))

    print(f'Checkpoint {checkpoint_name} saved !')


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32, dest='batch_size')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, dest='lr')
    parser.add_argument('-n', '--exp_name', type=str, default='exp', dest='exp_name')
    parser.add_argument('-e', '--epochs', type=int, default=100, dest='epochs')
    parser.add_argument('-s', '--seed', type=int, default=1, dest='seed')

    return parser.parse_args()

def evaluate(model, test_loader, device):
    model.eval()
    all = 0
    p = 0
    for batch in test_loader:
        _, clips, labels = batch
        clips = clips.to(device)
        labels = labels.to(device, dtype=torch.long).squeeze() # B * 1

        # ---- forward ----
        pred = model(clips) # B * 1000

        pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1).long()

        p += (pred_label == labels).sum().item()
        all += labels.size(0)
    model.train()

    return p/all


def train(train_loader, test_loader, model, optimizer, epochs, device, writer):
    # ----prepare ----
    model.to(device)
    model.train()
    total_step = 0
    criterion = nn.CrossEntropyLoss()

    # ---- training ----
    for epoch in range(1, epochs+1):
        with tqdm(total=len(train_loader), desc=f'epoch[{epoch}/{epochs+1}]:') as bar:
            for batch in train_loader:

                # ---- data prepare ----
                _, clips, labels = batch
                clips = clips.to(device)
                labels = labels.to(device, dtype=torch.long)

                # ---- forward ----
                preds = model(clips) # B * 1000

                # ---- loss ----
                loss = criterion(preds, labels)

                # ---- backward ----
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_step += 1

                # lr_ = args.lr * max(1.0 - total_step / (len(train_loader) * epochs), 1e-7) ** 0.9
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_

                # ---- log ----
                writer.add_scalar('info/loss', loss, total_step)
                bar.set_postfix(**{'loss (batch)': loss.item()})
                bar.update(1)

            # ---- validation ----
            accuracy = evaluate(model, test_loader, device)

            writer.add_scalar('eval/ac', accuracy, total_step)
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], total_step)
            print(f"""
                performance {accuracy}
            """)

    print('training finish')
    return accuracy

if __name__ == '__main__':

    # ---- init ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    imgs_dir = ''
    log_path = 'log/'
    train_folds = ''
    val_folds = ''

    try:
        os.makedirs(log_path)
    except:
        pass

    # ---- random seed ----
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    # ---- log & dataset ----
    if not os.path.exists(os.path.join(log_path, args.exp_name)):
        os.makedirs(os.path.join(log_path, args.exp_name))

    if os.path.exists(os.path.join(log_path, args.exp_name, 'log')):
        shutil.rmtree(os.path.join(log_path, args.exp_name, 'log'))
    writer = SummaryWriter(os.path.join(log_path, args.exp_name, 'log'))

    csv = pd.read_csv(train_folds)
    train_pool = [item[0] for item in csv.values]
    csv = pd.read_csv(val_folds)
    test_pool = [item[0] for item in csv.values]
    permutations = np.load('permutations.npy').tolist()

    train_set = FoldDataset(imgs_dir, train_pool, permutations, in_channels=1)
    test_set = FoldDataset(imgs_dir, test_pool, permutations, in_channels=1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True,
                              worker_init_fn=_init_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    # ---- model ----
    model = JigsawNet(1, 1000)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    epochs = args.epochs

    # train
    print(f'''
            training start! 
            train set num: {len(train_set)} 
            val set num: {len(test_set)}
            
            ''')

    ac = train(train_loader, test_loader, model, optimizer, epochs, device, writer)

    save_checkpoint(model, os.path.join(log_path, args.exp_name, 'checkpoints'), 0, ac)





