import argparse
from itertools import product

import torch
from datasets import get_dataset
from gcn import GCN
from gin import GIN
from graph_sage import GraphSAGE
from train_eval import eval_acc, train

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_stats_summary, profileit, timeit
from torch.profiler import profile, record_function, ProfilerActivity
# from torch_geometric.data import DataLoader
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler

seed_everything(0)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--warmup_profile', type=int, default=1,
                    help='Skip the first few runs')
parser.add_argument('--goal_accuracy', type=int, default=1,
                    help='The goal test accuracy')
args = parser.parse_args()

# layers = [1]
# hiddens = [16]
layers = [1, 2, 3]
hiddens = [16, 32]

datasets = ['REDDIT-BINARY']
# datasets = ['OGBN-PRODUCTS']
#datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']

nets = [
    GCN,
    #GraphSAGE,
    #GIN,
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Decorate train and eval functions:
# train = profileit()(train)
#train = profileit(print_layer_stats=False)(train)
# eval_acc = timeit()(eval_acc)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import os
    import pathlib
    profile_dir = str(pathlib.Path.cwd()) + '/'
    profile_file = profile_dir + 'profile-' + \
        str(dataset_name) + '-' + str(Net.__name__) + '-' + str(num_layers) + '-' + str(hidden) + '.log'
    print(profile_file)
    with open(profile_file, 'w') as f:
        f.write(output)
        f.close()
    timeline_file = profile_dir + 'profile-' + \
        str(dataset_name) + '-' + str(Net.__name__) + '-' + str(num_layers) + '-' + str(hidden) + '.json'
    p.export_chrome_trace(timeline_file)

for dataset_name, Net in product(datasets, nets):
    print(dataset_name)
    if dataset_name == "OGBN-PRODUCTS":
        dataset = PygNodePropPredDataset(name = 'ogbn-products', root = 'dataset/')
        print("dataset done")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        evaluator = Evaluator(name='ogbn-products')
        data = dataset[0]
        train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[10, 10, 10], batch_size=512,
                               shuffle=True, num_workers=12)
        # train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_idx, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_idx, batch_size=args.batch_size, shuffle=False)
    else:
        dataset = get_dataset(dataset_name, sparse=True)
        num_train = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)

        train_dataset = dataset[:num_train]
        val_dataset = dataset[num_train:num_train + num_val]
        test_dataset = dataset[num_train + num_val:]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False)

    for num_layers, hidden in product(layers, hiddens):
        print(f'--\n{dataset_name} - {Net.__name__} - {num_layers} - {hidden}')

        model = Net(dataset, num_layers, hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        stats_list = []
        # warmup (args.epochs - 1) times
        for epoch in range(1, args.epochs):
            train(model, optimizer, train_loader)
            eval_acc(model, val_loader)
            eval_acc(model, test_loader)
        # run the args.epochs time
        train(model, optimizer, train_loader)
        eval_acc(model, val_loader)

        # Profile test in the args.epochs time
        with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
            on_trace_ready=trace_handler) as p:
                eval_acc(model, test_loader)
                p.step()

        # if epoch >= args.warmup_profile:
        #     stats_list.append(stats)

        # stats_summary = get_stats_summary(stats_list)
        # print(stats_summary)
