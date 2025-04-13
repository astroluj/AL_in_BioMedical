from sklearn.cluster import AgglomerativeClustering

import torch
import numpy as np

from torch.utils.data import DataLoader

from data_sampler import SubsetSequentialSampler


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            scores, features, _ = models['backbone'](inputs)
            pred_loss = models['module'](features[-4:])
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


def get_real_loss(models, data_loader, criterion):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels = labels.view(labels.size(0))

            scores, features, _ = models['backbone'](inputs)
            target_loss = criterion(scores, labels)

            uncertainty = torch.cat((uncertainty, target_loss), 0)

    return uncertainty.cpu()


def clustering(model, cluster_size, data_loader):
    model.eval()

    features = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.cuda()

            _, _, feature = model(inputs)

            features = torch.cat((features, feature), 0)
    features = features.cpu().numpy()

    return AgglomerativeClustering(n_clusters=cluster_size, linkage='complete').fit_predict(features)


def sampling(cluster_dict):
    sampled = []

    for key in cluster_dict:
        sampled.append(cluster_dict[key][-1])

    return sampled


def scal(addendnum, subset, batch, labeled_set, unlabeled_set, models, criterion, evaluate_transform_data):
    unlabeled_loader = DataLoader(evaluate_transform_data, batch_size=batch,
                                  sampler=SubsetSequentialSampler(unlabeled_set),
                                  pin_memory=True)

    uncertainty = get_uncertainty(models, unlabeled_loader)

    arg = np.argsort(uncertainty)

    subset = list(torch.tensor(unlabeled_set)[arg][-subset:].numpy())

    subset_label = clustering(models['backbone'], addendnum,
                              DataLoader(evaluate_transform_data, batch_size=batch,
                                         sampler=SubsetSequentialSampler(subset),
                                         pin_memory=True))

    subset_cluster = {}
    for i, idx in enumerate(subset):
        if subset_label[i] not in subset_cluster:
            subset_cluster[subset_label[i]] = [idx]
        else:
            subset_cluster[subset_label[i]].append(idx)

    sampled_data = sampling(subset_cluster)
    sampled_loader = DataLoader(evaluate_transform_data, batch_size=batch,
                                sampler=SubsetSequentialSampler(sampled_data),
                                pin_memory=True)
    sampled_real_loss = get_real_loss(models, sampled_loader, criterion)
    sampled_arg = np.argsort(sampled_real_loss)
    sampled_data = list(torch.tensor(sampled_data)[sampled_arg].numpy())[::-1]

    labeled_loader = DataLoader(evaluate_transform_data, batch_size=batch,
                                sampler=SubsetSequentialSampler(labeled_set),
                                pin_memory=True)
    labeled_real_loss = get_real_loss(models, labeled_loader, criterion)
    labeled_arg = np.argsort(labeled_real_loss)
    labeled_set = list(torch.tensor(labeled_set)[labeled_arg].numpy())

    labeled_set += sampled_data
    unlabeled_set = list(set(unlabeled_set) - set(labeled_set))

    return labeled_set, unlabeled_set
