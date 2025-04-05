import configargparse
import argparse
import copy
import numpy as np

def parse_args(args=None, dataset='disk'):

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='whether to use GPU')
    parser.add_argument('--NF-dyn', action='store_true',help='train using normalising flow')
    parser.add_argument('--NF-cond', action='store_true',help='train using conditional normalising flow')
    parser.add_argument('--measurement',type=str, default='cos', help='|CRNVP|cos|NN|CGLOW|gaussian|')
    parser.add_argument('--NF-lr', type=float, default=2.5,help='NF learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='num epochs')
    parser.add_argument('--num_exp', type=int, default=100, help='num exp')
    parser.add_argument('--trainType', type=str, default='dpf', choices=['dpf', 'dpf_pl', 'dpf_elbo'], help='train type: supervised, semi, unsupervised learning')
    parser.add_argument('--labelled_ratio', type=float, default=0.1, help='labelled_ratio for supervised learning')
    parser.add_argument('--elbo_ratio', type=float, default=1e-3, help='ratio for elbo in total loss')
    parser.add_argument('--pl_ratio', type=float, default=1e-2, help='ratio for pl in total loss')
    parser.add_argument('--rmse_ratio', type=float, default=1e-4, help='ratio for rmse in total loss')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon in OT resampling')
    parser.add_argument('--scaling', type=float, default=0.75, help='scaling in OT resampling')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameter for soft resampling')
    parser.add_argument('--threshold', type=float, default=1e-2, help='threshold in OT resampling')
    parser.add_argument('--max_iter', type=int, default=100, help='max iteration in OT resampling')
    parser.add_argument('--resampler_type',type=str, default='ot', help='|ot|soft|')
    parser.add_argument('--experiment', type=str, default='linear_gaussian', help='|linear_gaussian|position_tracking|')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    param = copy.deepcopy(args)
    for labeledRatio in param.labeledRatio:
        args.labeledRatio = labeledRatio
        print(args)