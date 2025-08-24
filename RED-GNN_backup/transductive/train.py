import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import select_gpu

parser = argparse.ArgumentParser(description="Parser for RED-GNN")
parser.add_argument('--data_path', type=str, default='data/family/')
parser.add_argument('--seed', type=str, default=1234)


args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]

    save_dir = '/content/drive/MyDrive/RED-GNN/transductive/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    results_dir = save_dir + 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    weights_dir = save_dir + "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    checkpoint_dir = save_dir + "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset + '_perf.txt')
    opts.best_weight_file = os.path.join(weights_dir, dataset + '_weight.pt')
    opts.checkpoint_file = os.path.join(checkpoint_dir, dataset + '_checkpoint.pt')

    # gpu = select_gpu()
    # torch.cuda.set_device(gpu)
    # print('gpu:', gpu)
    
    loader = DataLoader(args.data_path)
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel

    if dataset == 'family':
        opts.lr = 0.0036
        opts.decay_rate = 0.999
        opts.lamb = 0.000017
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.29
        opts.act = 'relu'
        opts.n_batch = 20
        opts.n_tbatch = 50
    elif dataset == 'umls':
        opts.lr = 0.0012
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.01
        opts.act = 'tanh'
        opts.n_batch = 10
        opts.n_tbatch = 50
    elif dataset == 'WN18RR':
        opts.lr = 0.0003
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 64
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.02
        opts.act = 'idd'
        opts.n_batch = 50
        opts.n_tbatch = 50
    elif dataset == 'fb15k-237':
        opts.lr = 0.0009
        opts.decay_rate = 0.9938
        opts.lamb = 0.000080
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 4
        opts.dropout = 0.0391
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1
    elif dataset == 'nell':
        opts.lr = 0.0011
        opts.decay_rate = 0.9938
        opts.lamb = 0.000089
        opts.hidden_dim = 48
        opts.attn_dim = 5
        opts.n_layer = 5
        opts.dropout = 0.2593
        opts.act = 'relu'
        opts.n_batch = 5
        opts.n_tbatch = 1



    config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str)
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)

    model = BaseModel(opts, loader)

    start_epoch, best_mrr = model.load_checkpoint(opts.checkpoint_file)
    end_epoch = 50
    if start_epoch is None:
        start_epoch = 0
    if best_mrr is None:
        best_mrr = 0

    for epoch in range(start_epoch, end_epoch):
        print(f"Epoch {epoch}")
        mrr, out_str = model.train_batch()

        with open(opts.perf_file, 'a+') as f:
            f.write(f'epoch {epoch}  ' + out_str)

        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            print(f"Best Epoch {epoch}:\t{best_str}")
            model.save_checkpoint(opts.best_weight_file, epoch + 1, best_mrr)
            print(f"Best model saved at epoch {epoch + 1}")

        if (epoch + 1) % 2 == 0:
            model.save_checkpoint(opts.checkpoint_file, epoch + 1, best_mrr)
            print(f"Checkpoint saved at epoch {epoch + 1}")

