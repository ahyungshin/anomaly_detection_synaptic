import os
import argparse
import time

from torch.backends import cudnn
from utils.utils import *

from solver import Solver

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    start = time.time()
    seed_fix(config.seed)

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
        
    end = time.time()-start
    print("TOTAL RUNNING TIME:", end)

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--cluster_t', type=float, default=1.0)
    parser.add_argument('--consist_coeff', type=float, default=1.0)
    parser.add_argument('--fe_epochs', type=int, default=1000)
    parser.add_argument('--ploss_coeff', type=float, default=100)
    parser.add_argument('--ae_lr',type=float,default=1e-4)
    parser.add_argument('--pool_size',type=int,default=10)
    parser.add_argument('--prompt_num',type=int,default=10)
    parser.add_argument('--use_p_noise',action='store_true')
    parser.add_argument('--plot',action='store_true')
    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
