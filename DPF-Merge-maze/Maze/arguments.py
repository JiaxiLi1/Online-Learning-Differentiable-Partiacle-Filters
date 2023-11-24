# import configargparse
import argparse
import copy

def parse_args(args=None):

    # parser = configargparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps_path', type=str, default='./maze01.jpg', help='path to the map image')
    parser.add_argument('--gpu', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--gpu-index', type=int, default=0, help='index num of GPU to use')

    parser.add_argument('--dataset', dest='dataset', type=str,
                        default='maze', choices=['toy', 'maze', 'house3d', 'kitti'],
                        help='which problem to address, and choose which dataset')

    parser.add_argument('--trainType', dest='trainType', type=str,
                        default='semi', choices=['sup', 'semi', 'unsup'],
                        help='train type: supervised, semi, unsupervised learning')

    parser.add_argument('--pretrain', type=bool, default=False,
                        help='pretrain of model: AE')

    parser.add_argument('--batchsize', type=int, default=32, help='batch size') # origin: 32, 24
    parser.add_argument('--hiddensize', type=int, default=128, help='hidden size')  # origin: 32 for toy; 128 for maze
    parser.add_argument('--lr', type=float, default=1e-3, # one object: 1e-3
                        help='learning rate') # origin: 1e-3, 1e-4
    parser.add_argument('--optim', type=str, default='Adam',
                        help='type of optim')
    parser.add_argument('--epochs', type=int, default=300, help='num epochs') # origin: 20, 13
    parser.add_argument('--num-particles', type=int, default=100, help='num of particles')

    parser.add_argument('--mazeID', type=str, default='nav01', help='nav01 means Maze 1, navigation')

    parser.add_argument('--split-ratio', type=float, default=0.1, help='split training data')
    parser.add_argument('--labeledRatio', type=float, default=0.1, help='labeled training data') # 1.0 [0.1, 0.3, 0.5, 0.7, 0.9]
    parser.add_argument('--init-with-true-state', type=bool, default=True,help='init_with_true_state, default: false, uniform initialisation')
    parser.add_argument('--n_sequence', type = int, default=2, help='number of sequence in normalizing flows')

    parser.add_argument('--NF-dyn', action='store_true',help='train using normalising flow')
    parser.add_argument('--NF-cond', action='store_true',help='train using conditional normalising flow')
    parser.add_argument('--CNF-measurement', action='store_true', help='build measurement model with conditional normalising flow')
    parser.add_argument('--NF-lr', type=float, default=0.25, help='NF learning rate')

    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon in OT resampling')
    parser.add_argument('--scaling', type=float, default=0.75, help='scaling in OT resampling')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameter for soft resampling')
    parser.add_argument('--threshold', type=float, default=1e-3, help='threshold in OT resampling')
    parser.add_argument('--max_iter', type=int, default=100, help='max iterarion in OT resampling')
    parser.add_argument('--resampler_type',type=str, default='soft', help='|ot|soft|')

    parser.add_argument('--dropout-keep-ratio', type=float, default=0.3, help='1-dropout_ratio')
    parser.add_argument('--particle_std', type=float, default=0.2, help='particle std')
    parser.add_argument('--seed', type=int, default=10, help='random seed')

    parser.add_argument('--std-x', type=float, default=15.0, help='std_x')
    parser.add_argument('--std-y', type=float, default=15.0, help='std_y')
    parser.add_argument('--std-t', type=float, default=0.1, help='std_t')

    parser.add_argument('--sequence-length', dest='sequence_length', type=int,
                        default=50, help='length of the generated sequences')
    parser.add_argument('--width', dest='width', type=int, default=128, #120
                        help='width (= height) of the generated observations')

    parser.add_argument('--pos-noise', dest='pos_noise', type=float,
                        default=0.1,
                        help='sigma for the positional process noise')
    parser.add_argument('--vel-noise', dest='vel_noise', type=float,
                        default=2.0,
                        help='sigma for the velocity noise')

    parser.add_argument('--block-length', dest='block_length', type=int,
                        default=10, help='block length for pseudo-likelihood')  # dest: destination, origin=5

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    param = copy.deepcopy(args)
    for labeledRatio in param.labeledRatio:
        args.labeledRatio = labeledRatio
        print(args)

