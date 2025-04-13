import os
import PIL
import time
import pickle
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm

from config import *
from train_utils import frozen, free
from models.resnet import ResNet18
from models.lossnet import LossNet
from data_sampler import SubsetSequentialSampler

from kcenterGreedy import kCenterGreedy
from scal import scal

import medmnist
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms


ADDENDUM_SIZE = {'pathmnist': 2500, 'dermamnist': 140, 'octmnist': 2000,
                 'pneumoniamnist': 100, 'bloodmnist': 240, 'tissuemnist': 3300,
                 'organamnist': 700, 'organcmnist': 250, 'organsmnist': 280}
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[.5], std=[.5])
    ])


def loss_pred_loss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss


def ft_epoch(models, criterion, optimizers, dataloaders, info_task):
    models['backbone'].eval()
    models['module'].train()

    free(models['module'])
    frozen(models['backbone'])

    for data in tqdm(dataloaders['ft'], leave=False, total=len(dataloaders['ft'])):
        inputs = data[0].cuda()
        if info_task == 'multi-label, binary-class':
            labels = data[1].to(torch.float32).cuda()
        else:
            labels = torch.squeeze(data[1], 1).long().cuda()

        optimizers['ft'].zero_grad()

        scores, features, _ = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        if info_task == 'multi-label, binary-class':
            target_loss = target_loss.sum(dim=1)

        features[0] = features[0].detach()
        features[1] = features[1].detach()
        features[2] = features[2].detach()
        features[3] = features[3].detach()

        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        loss = loss_pred_loss(pred_loss, target_loss, margin=MARGIN)

        loss.backward()
        optimizers['ft'].step()


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, method, info_task):
    if method == 'll4al' or method == 'scal':
        free(models['module'])
        models['module'].train()

    free(models['backbone'])
    models['backbone'].train()

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        if info_task == 'multi-label, binary-class':
            labels = data[1].to(torch.float32).cuda()
        else:
            labels = torch.squeeze(data[1], 1).long().cuda()

        optimizers['backbone'].zero_grad()

        if method == 'll4al' or method == 'scal':
            optimizers['module'].zero_grad()

        scores, features, _ = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        if info_task == 'multi-label, binary-class':
            target_loss = target_loss.sum(dim=1)
        loss = torch.sum(target_loss) / target_loss.size(0)

        if method == 'll4al' or method == 'scal':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_module_loss = loss_pred_loss(pred_loss, target_loss, margin=MARGIN)
            loss += WEIGHT * m_module_loss

        loss.backward()
        if method == 'll4al' or method == 'scal':
            optimizers['module'].step()
        optimizers['backbone'].step()


def test(models, evaluator, data_loader, info_task, criterion, run, save_folder=None):
    if method == 'll4al' or method == 'scal':
        models['module'].eval()
    models['backbone'].eval()

    total_loss = []
    y_score = torch.tensor([]).cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.cuda()
            outputs, _, _ = models['backbone'](inputs)
            
            if info_task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).cuda()
                loss = criterion(outputs, targets)
                loss = torch.sum(loss) / loss.size(0)
                m = nn.Sigmoid()
                outputs = m(outputs).cuda()
            else:
                targets = torch.squeeze(targets, 1).long().cuda()
                loss = criterion(outputs, targets)
                loss = torch.sum(loss) / loss.size(0)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).cuda()
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, method, info_task):
    print('>> Train a Model.')

    for epoch in range(num_epochs):
        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, method, info_task)

        if method == 'll4al' or method == 'scal':
            schedulers['module'].step()
        schedulers['backbone'].step()

    if method == 'scal':
        for epoch in range(20):
            ft_epoch(models, criterion, optimizers, dataloaders, info_task)

    print('>> Finished.')


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            scores, features, _ = models['backbone'](inputs)
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()


def get_kcg(models, labeled_data_size, unlabeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            inputs = inputs.cuda()
            _, _, features_batch = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)

        feat = features.detach().cpu().numpy()

        new_av_idx = np.arange(unlabeled_data_size, (unlabeled_data_size + labeled_data_size))
        sampling = kCenterGreedy(feat)
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)

        other_idx = [x for x in range(unlabeled_data_size) if x not in batch]

    return other_idx + batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--method', type=str, help='[scal, ll4al, coreset, random]')
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='pathmnist')
    args = parser.parse_args()
    method = args.method

    for trial in range(TRIALS):
        random.seed(trial)
        np.random.seed(trial)
        torch.manual_seed(trial)
        torch.cuda.manual_seed(trial)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        info = INFO[args.dataset]
        ADDENDUM = ADDENDUM_SIZE[args.dataset]
        SUBSET = ADDENDUM * 10

        DataClass = getattr(medmnist, info['python_class'])

        CLS_CNT = len(info['label'])
        print(info['n_samples'])

        tmp_idx = [i for i in range(info['n_samples']['train'])]
        random.shuffle(tmp_idx)
        labeled_set, unlabeled_set = tmp_idx[:ADDENDUM], tmp_idx[ADDENDUM:]
        print(labeled_set[:10], len(labeled_set))
        fp = open(f'record_{trial + 1}.txt', 'w')

        removal_size = max(len(labeled_set) // 100, 2)
        removal_size = removal_size - 1 if removal_size % 2 else removal_size
        train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=True)
        test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=True)

        test_loader = DataLoader(test_dataset, batch_size=BATCH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader}

        if method == 'scal':
            WEIGHT = args.weight
            SUBSET = ADDENDUM * 10

            scal_train_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set[:-removal_size]), pin_memory=True)
            dataloaders = {'train': scal_train_loader, 'test': test_loader, 'ft': train_loader}
        
        resnet18 = ResNet18(num_classes=CLS_CNT, channel_size=3).cuda()
        models = {'backbone': resnet18}
        if method == 'll4al' or method == 'scal':
            loss_module = LossNet().cuda()
            models['module'] = loss_module

        train_evaluator = medmnist.Evaluator(args.dataset, 'train')
        test_evaluator = medmnist.Evaluator(args.dataset, 'test')

        for cycle in range(CYCLES):
            print(f'cycle {cycle + 1} start -  labeled data size: {len(labeled_set)} / unlabeled data size: {len(unlabeled_set)}')
            if info['task'] == "multi-label, binary-class":
                criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
            else:
                criterion = nn.CrossEntropyLoss(reduction='none').cuda()

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers = {'backbone': optim_backbone}

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            schedulers = {'backbone': sched_backbone}

            if method == 'll4al' or method == 'scal':
                optim_module = optim.SGD(models['module'].parameters(), lr=0.001, momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers['module'] = optim_module
                schedulers['module'] = sched_module

                optim_ft = optim.SGD(models['module'].parameters(), lr=0.001, momentum=MOMENTUM, weight_decay=WDECAY)
                optimizers['ft'] = optim_ft
                

            print(f'labeled: {len(labeled_set)} / unlabeled: {len(unlabeled_set)}')
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, method, info['task'])
            test_eval = test(models, test_evaluator, dataloaders['test'], info['task'], criterion, 'model1')
            
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(set(labeled_set)), test_eval))

            start_time = time.process_time()
            if method == 'scal' and cycle < CYCLES - 1:
                print(ADDENDUM, SUBSET)
                labeled_set, unlabeled_set = scal(ADDENDUM, SUBSET, BATCH, labeled_set, unlabeled_set,
                                                  models, criterion, train_dataset)

            else:
                SUBSET = min(20000, max(len(unlabeled_set) // 2, ADDENDUM * 10))
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]
                if method == 'll4al':
                    unlabeled_loader = DataLoader(train_dataset, batch_size=BATCH,
                                                  sampler=SubsetSequentialSampler(subset),
                                                  pin_memory=True)

                    uncertainty = get_uncertainty(models, unlabeled_loader)

                    arg = np.argsort(uncertainty)

                    labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                    unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

                elif method == 'coreset':
                    unlabeled_loader = DataLoader(train_dataset, batch_size=BATCH,
                                                  sampler=SubsetSequentialSampler(subset + labeled_set),
                                                  pin_memory=True)

                    arg = get_kcg(models, len(labeled_set), len(subset), unlabeled_loader)

                    labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                    unlabeled_set = list(set(unlabeled_set) - set(labeled_set))

                elif method == 'random':
                    labeled_set += unlabeled_set[:ADDENDUM]
                    unlabeled_set = list(set(unlabeled_set) - set(labeled_set))
            end_time = time.process_time()
            
            fp.write(f'{test_eval} / {end_time - start_time}\n')
            
            if method == 'scal':
                if cycle == CYCLES - 2:
                    dataloaders['train'] = DataLoader(train_dataset, batch_size=BATCH,
                                                        sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
                else:
                    _size = removal_size // 2
                    dataloaders['train'] = DataLoader(train_dataset, batch_size=BATCH,
                                                        sampler=SubsetRandomSampler(labeled_set[_size:-_size]),
                                                        pin_memory=True)

                dataloaders['ft'] = DataLoader(train_dataset, batch_size=BATCH,
                                               sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

            else:
                dataloaders['train'] = DataLoader(train_dataset, batch_size=BATCH,
                                                  sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

        fp.close()
