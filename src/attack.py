import logging
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
import copy
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
import torch_geometric
import random
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models import GIN, GCN, GraphSAGE
from data_loader import inject_sub_trigger, S2VGraph
from graphcnn import GraphCNN
import argparse


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')


def prepare_dataset_x(dataset, max_degree=0):
    if dataset[0].x is None:
        if max_degree == 0:
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())
                data.num_nodes = int(torch.max(data.edge_index)) + 1
        else:
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                data.num_nodes = int(torch.max(data.edge_index)) + 1
            max_degree = max_degree
        if max_degree < 10000:
            # dataset.transform = T.OneHotDegree(max_degree)

            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree + 1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ((degs - mean) / std).view(-1, 1)
    return dataset, max_degree


def prepare_dataset_onehot_y(dataset, num_classes=2):
    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))

    for idx, data in enumerate(dataset):
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]

    return dataset


def train(args, model, criterion, train_loader, epoch):
    model.train()
    loss_all = 0
    correct = 0
    graph_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y.view(-1, args.num_classes)
        loss = criterion(output, y).to(device)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    loss = loss_all / graph_all

    if epoch % 10 == 0:
        logger.info(
            'Epoch: {:03d}, Train Loss: {:.6f}'.format(
                epoch, loss))
    return model, loss


def test(args, model, criterion, loader, epoch, type="Train"):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        y = data.y.view(-1, args.num_classes)
        loss += criterion(output, y).item() * data.num_graphs
        y = y.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    acc = correct / total
    loss = loss / total

    if epoch % 10 == 0:
        logger.info(
                '[{}] Epoch: {:03d}, Test Loss: {:.6f} Test Accuracy {:.6f}'.format(str(type),
                    epoch, loss, acc))
    return acc, loss


def preprocess_dataset(dataset, num_classes=2, max_degree=0):
    for graph in dataset:
        graph.y = graph.y.view(-1)

    # dataset = prepare_dataset_onehot_y(dataset, num_classes=num_classes)
    dataset, max_degree = prepare_dataset_x(dataset, max_degree=max_degree)
    return dataset, max_degree


def train_(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    logger.info(
        'Epoch: {:03d}, Train Loss: {:.6f}'.format(
            epoch, average_loss))

    return average_loss
###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test_(model, device, test_graphs, type="backdoor"):
    model.eval()

    output = pass_data_iteratively(model, test_graphs)


    pred = output.max(1, keepdim=True)[1]

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # print(labels)
    loss = criterion(output, labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    logger.info(
        '[{}] Epoch: {:03d}, Test Loss: {:.6f} Test Accuracy {:.6f}'.format(str(type),
                                                                            epoch, loss, acc_test))

    return acc_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="MUTAG")
    parser.add_argument('--model', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--gnn', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_screen', type=str, default="True")

    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')

    parser.add_argument("--injection_ratio", default=0.1, type=float, help="the number of injected samples to the training dataset")
    parser.add_argument("--split_ratio", default=0.9, type=float, help="train/test split ratio")
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
    parser.add_argument("--trigger_size", default=3, type=int, help="# of Nodes to be poisoned")
    parser.add_argument("--trigger_density", default=0.8, type=float, help="Density of Subgraph Triggers")
    parser.add_argument("--device", default="0", type=str, help="GPU device index")

    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    log_screen = eval(args.log_screen)
    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    model = args.model

    handler = logging.FileHandler(
        "./logs/LR{}+Dataset{}+Injection_ratio{}.txt".format(args.lr, args.dataset, args.injection_ratio),
        encoding='utf-8', mode='a')

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(handler)

    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))

    torch.manual_seed(seed)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    logger.info(f"runing device: {device}")

    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name)
    dataset = list(dataset)
    random.seed(seed)
    random.shuffle(dataset)

    train_nums = int(len(dataset) * args.split_ratio)
    test_nums = int(len(dataset) * (1-args.split_ratio))

    train_dataset = dataset[:train_nums]
    test_dataset = dataset[train_nums:]
    train_dataset, injected_graph_idx, backdoor_train_dataset, clean_train_dataset = inject_sub_trigger(copy.deepcopy(train_dataset),
                                                                                                        inject_ratio=args.injection_ratio,
                                                                                                        target_label=0, backdoor_num=args.trigger_size,
                                                                                                        density=args.trigger_density)

    logger.info("# Train Dataset {}".format(len(train_dataset)))
    logger.info("# Backdoor Train Dataset {}".format(len(backdoor_train_dataset)))
    logger.info("# Clean Train Dataset {}".format(len(clean_train_dataset)))

    _, _, backdoor_test_dataset, _ = inject_sub_trigger(copy.deepcopy(test_dataset), inject_ratio=1, target_label=0, density=args.trigger_density,
                                                        backdoor_num=args.trigger_size)

    _, _, _, clean_test_dataset = inject_sub_trigger(copy.deepcopy(test_dataset), inject_ratio=0, target_label=0)


    logger.info("# Test Dataset {}".format(len(test_dataset)))
    logger.info("# Backdoor Test Dataset {}".format(len(backdoor_test_dataset)))
    logger.info("# Clean Test Dataset {}".format(len(clean_test_dataset)))

    # After splitting, preprocess all the dataset
    train_dataset, max_degree = preprocess_dataset(train_dataset, num_classes=args.num_classes)
    backdoor_test_dataset, _ = preprocess_dataset(backdoor_test_dataset, num_classes=args.num_classes, max_degree=max_degree)
    clean_test_dataset, _ = preprocess_dataset(clean_test_dataset, num_classes=args.num_classes, max_degree=max_degree)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=batch_size)
    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=batch_size)

    num_features = train_dataset[0].x.shape[1]

    # preprocess the batch data
    train_backdoor = []
    test_graphs = []
    test_backdoor = []

    if model == "GIN":
        model = GIN(num_features=num_features, num_classes=args.num_classes, num_hidden=num_hidden).to(device)
    elif model =="GCN":
        model = GCN(num_features=num_features, num_classes=args.num_classes, num_hidden=num_hidden).to(device)
    elif model == "sage":
        model = GraphSAGE(num_features=num_features, num_classes=args.num_classes, num_hidden=num_hidden).to(device)
    elif model == "GraphCNN":
        # implemented by https://github.com/zaixizhang/graphbackdoor
        for graph in train_dataset:
            net = torch_geometric.utils.to_networkx(graph)
            graph_s2v = S2VGraph(net, graph.y.item(), None, graph.x)
            graph_s2v.edge_mat = graph.edge_index
            train_backdoor.append(graph_s2v)

        for graph in clean_test_dataset:
            net = torch_geometric.utils.to_networkx(graph)
            graph_s2v = S2VGraph(net, graph.y.item(), None, graph.x)
            graph_s2v.edge_mat = graph.edge_index
            test_graphs.append(graph_s2v)

        for graph in backdoor_test_dataset:
            net = torch_geometric.utils.to_networkx(graph)
            graph_s2v = S2VGraph(net, graph.y.item(), None, graph.x)
            graph_s2v.edge_mat = graph.edge_index
            test_backdoor.append(graph_s2v)

        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_backdoor[0].node_features.shape[1],
                         args.hidden_dim, args.num_classes, args.final_dropout, args.lr, args.graph_pooling_type,
                         args.neighbor_pooling_type, device).to(device)

    else:
        logger.info(f"No model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    if args.model == "GraphCNN":
        # implemented by https://github.com/zaixizhang/graphbackdoor
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.epoch + 1):
            scheduler.step()
            avg_loss = train_(args, model, device, train_backdoor, optimizer, epoch)
            acc_train=test_(model, device, train_backdoor, type="Trainset")
            acc_test_clean = test_(model, device, test_graphs, type="Clean")
            acc_test_backdoor = test_(model, device, test_backdoor, type="Backdoor")
    else:
        for epoch in range(1, num_epochs):
            criterion = nn.CrossEntropyLoss()
            model, _ = train(args, model, criterion, train_loader, epoch)
            train_acc, _ = test(args, model, criterion, train_loader, epoch, type="Trainset")
            test_acc, test_loss = test(args, model, criterion, clean_test_loader, epoch, type="Clean Test")
            backdoor_test_acc, backdoor_test_loss = test(args, model, criterion, backdoor_test_loader, epoch, type="Backdoor Test")
            scheduler.step()
            if epoch % 5 == 0:
                logger.info("=================Epoch {}=================".format(epoch))



