import configargparse
import argparse
import copy
import numpy as np

def parse_args_disk(args=None, dataset='disk'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='whether to use GPU')
    parser.add_argument('--gpu-index', type=int, default=0, help='index num of GPU to use')

    parser.add_argument('--trainType', dest='trainType', type=str,
                        default='DPF', choices=['DPF', 'SDPF', 'UDPF'],
                        help='train type: supervised, semi, unsupervised learning')
    parser.add_argument('--dataset', dest='dataset', type=str,
                        default='disk', choices=['disk', 'maze','house3d'],
                        help='experiment dataset')

    parser.add_argument('--pretrain_ae', action='store_true',
                        help='pretrain of autoencoder model')
    parser.add_argument('--pretrain-NFcond', action='store_true',
                        help='pretrain of conditional normalising flow model')
    parser.add_argument('--e2e-train', action='store_false',
                        help='End to end training')
    parser.add_argument('--load-pretrainModel', action='store_true',
                        help='Load pretrain model')
    parser.add_argument('--NF-dyn', action='store_true',help='train using normalising flow')
    parser.add_argument('--NF-cond', action='store_true',help='train using conditional normalising flow')
    parser.add_argument('--measurement',type=str, default='cos', help='|CRNVP|cos|NN|CGLOW|gaussian|')
    parser.add_argument('--NF-lr', type=float, default=2.5,help='NF learning rate')

    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon in OT resampling')
    parser.add_argument('--scaling', type=float, default=0.75, help='scaling in OT resampling')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameter for soft resampling')
    parser.add_argument('--threshold', type=float, default=1e-3, help='threshold in OT resampling')
    parser.add_argument('--max_iter', type=int, default=100, help='max iteration in OT resampling')
    parser.add_argument('--resampler_type',type=str, default='ot', help='|ot|soft|')

    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint')

    parser.add_argument('--Dyn_nn', action='store_true',
                        help='learned dynamic model using neural network')
    parser.add_argument('--Obs_feature', action='store_false',
                        help='Compute likelihood using feature similarity')

    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--hiddensize', type=int, default=32, help='hidden size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='type of optim')
    parser.add_argument('--num-epochs', type=int, default=150, help='num epochs')
    parser.add_argument('--num-particles', type=int, default=100, help='num of particles')

    parser.add_argument('--split-ratio', type=float, default=0.1, help='split training data')
    parser.add_argument('--labeledRatio', type=float, default=0.1, help='labeled training data')
    parser.add_argument('--lamda', type=float, default=0.01, help='Labeled ratio of true label state')
    parser.add_argument('--elbo_ratio', type=float, default=10.0,
                        help='elbo_ratio in loss')  # 1.0 [0.1, 0.3, 0.5, 0.7, 0.9]
    parser.add_argument('--init-with-true-state', action='store_true',
                        help='init_with_true_state, default: false, uniform initialisation')

    parser.add_argument('--dropout-keep-ratio', type=float, default=0.3, help='1-dropout_ratio')
    parser.add_argument('--particle_std', type=float, default=0.2, help='particle std')
    parser.add_argument('--seed', type=int, default=2, help='random seed')

    parser.add_argument('--sequence-length', dest='sequence_length', type=int,
                        default=50, help='length of the generated sequences')
    parser.add_argument('--width', dest='width', type=int, default=128,
                        help='width (= height) of the generated observations')

    parser.add_argument('--pos-noise', dest='pos_noise', type=float,
                        default=20.0,
                        help='sigma for the positional process noise')
    parser.add_argument('--vel-noise', dest='vel_noise', type=float,
                        default=20.0,
                        help='sigma for the velocity noise')

    parser.add_argument('--true-pos-noise', dest='true_pos_noise', type=float,
                        default=2.0,
                        help='sigma for the positional process noise when generating datasets')
    parser.add_argument('--true-vel-noise', dest='true_vel_noise', type=float,
                        default=2.0,
                        help='sigma for the velocity noise when generating datasets')

    parser.add_argument('--block-length', dest='block_length', type=int,
                        default=10, help='block length for pseudo-likelihood')

    parser.add_argument('--testing', action='store_true',
                        help='Check testing performance')
    parser.add_argument('--model-path', type=str, default='./model/e2e_model_bestval_e2e.pth', help='path of saved model')


    parser.add_argument("--x_size", type=tuple, default=(3,8,8))
    parser.add_argument("--y_size", type=tuple, default=(3,8,8))
    parser.add_argument("--x_hidden_channels", type=int, default=8)
    parser.add_argument("--x_hidden_size", type=int, default=16)
    parser.add_argument("--y_hidden_channels", type=int, default=8)
    parser.add_argument("-K", "--flow_depth", type=int, default=1)
    parser.add_argument("-L", "--num_levels", type=int, default=1)
    parser.add_argument("--learn_top", type=bool, default=False)

    parser.add_argument("--x_bins", type=float, default=256.0)
    parser.add_argument("--y_bins", type=float, default=256.0)

    parser.add_argument("--individual", action='store_true',help='set individual opimizers for different units')
    args = parser.parse_args()

    return args

def parse_args_maze(args=None):

    # parser = configargparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps_path', type=str, default='./maps/maze01.jpg', help='path to the map image')
    parser.add_argument('--gpu', action='store_true', help='whether to use GPU')
    parser.add_argument('--gpu-index', type=int, default=0, help='index num of GPU to use')

    parser.add_argument('--dataset', dest='dataset', type=str,
                        default='maze', choices=['toy', 'maze', 'house3d'],
                        help='which problem to address, and choose which dataset')

    parser.add_argument('--trainType', dest='trainType', type=str,
                        default='DPF', choices=['DPF', 'SDPF', 'UDPF'],
                        help='train type: supervised, semi, unsupervised learning')

    parser.add_argument('--learnType', dest='learnType', type=str,
                        default='offline', choices=['offline', 'online'],
                        help='learn type: offline learning or online learning')
    parser.add_argument('--onlineType', dest='onlineType', type=str,
                        default='fix', choices=['elbo', 'fix', 'rmse'],
                        help='onlineType: elbo or fix or rmse')
    parser.add_argument('--slice_size', type=int, default=10, help='slice_size')

    parser.add_argument('--pretrain', type=bool, default=False,
                        help='pretrain of model: AE')

    parser.add_argument('--batchsize', type=int, default=32, help='batch size') # origin: 32, 24
    parser.add_argument('--hiddensize', type=int, default=128, help='hidden size')  # origin: 32 for toy; 128 for maze
    parser.add_argument('--lr', type=float, default=1e-3, # one object: 1e-3
                        help='learning rate') # origin: 1e-3, 1e-4
    parser.add_argument('--optim', type=str, default='Adam',
                        help='type of optim')
    parser.add_argument('--num_epochs', type=int, default=100, help='num epochs') # origin: 20, 13
    parser.add_argument('--num-particles', type=int, default=100, help='num of particles')

    parser.add_argument('--mazeID', type=str, default='nav01', help='nav01 means Maze 1, navigation')

    parser.add_argument('--split-ratio', type=float, default=0.9, help='split training data')
    parser.add_argument('--lamda', type=float, default=0.01, help='Labeled ratio of true label state')
    parser.add_argument('--labeledRatio', type=float, default=1.0, help='labeled training data')
    parser.add_argument('--elbo_ratio', type=float, default=10.0, help='elbo_ratio in loss')# 1.0 [0.1, 0.3, 0.5, 0.7, 0.9]
    parser.add_argument('--init-with-true-state', type=bool, default=False,help='init_with_true_state, default: true')
    parser.add_argument('--n_sequence', type = int, default=2, help='number of sequence in normalizing flows')

    parser.add_argument('--NF-dyn', action='store_true',help='train using normalising flow')
    parser.add_argument('--NF-cond', action='store_true',help='train using conditional normalising flow')
    parser.add_argument('--measurement',type=str, default='cos', help='|CRNVP|cos|NN|CGLOW|gaussian|')
    # parser.add_argument('--CNF-measurement', action='store_true', help='build measurement model with conditional normalising flow')
    parser.add_argument('--NF-lr', type=float, default=0.25, help='NF learning rate')

    parser.add_argument('--epsilon', type=float, default=0.5, help='epsilon in OT resampling')
    parser.add_argument('--scaling', type=float, default=0.75, help='scaling in OT resampling')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameter for soft resampling')
    parser.add_argument('--threshold', type=float, default=1e-3, help='threshold in OT resampling')
    parser.add_argument('--max_iter', type=int, default=100, help='max iterarion in OT resampling')
    parser.add_argument('--resampler_type',type=str, default='ot', help='|ot|soft|')

    parser.add_argument('--dropout-keep-ratio', type=float, default=0.3, help='1-dropout_ratio')
    parser.add_argument('--particle_std', type=float, default=0.2, help='particle std')
    parser.add_argument('--seed', type=int, default=10, help='random seed')

    parser.add_argument('--std-x', type=float, default=0.1, help='std_x')
    parser.add_argument('--std-y', type=float, default=0.1, help='std_y')
    parser.add_argument('--std-t', type=float, default=0.1, help='std_t')

    parser.add_argument('--sequence-length', dest='sequence_length', type=int,
                        default=99, help='length of the generated sequences')
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
    parser.add_argument('--testing', action='store_true',
                        help='Check testing performance')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint')
    parser.add_argument('--pretrain_ae', action='store_true',
                        help='pretrain of autoencoder model')
    parser.add_argument('--pretrain-NFcond', action='store_true',
                        help='pretrain of conditional normalising flow model')
    parser.add_argument('--e2e-train', action='store_false',
                        help='End to end training')
    parser.add_argument('--load-pretrainModel', action='store_true',
                        help='Load pretrain model')
    parser.add_argument('--model-path', type=str, default='./model/e2e_model_bestval_e2e.pth', help='path of saved model')

    args = parser.parse_args()

    return args


def parse_args_house3d(args=None):
    """
    Parse command line arguments
    :param args: command line arguments or None (default)
    :return: dictionary of parameters
    """

    p = configargparse.ArgParser(default_config_files=[])

    p.add('--dataset', dest='dataset', type=str,
                        default='maze', choices=['toy', 'maze', 'house3d'],
                        help='which problem to address, and choose which dataset')

    p.add('-c', '--config', required=True, is_config_file=True,
          help='Config file. use ./config/train.conf for training')

    p.add('--trainfiles', nargs='*', help='Data file(s) for training (tfrecord).')
    p.add('--valfiles', nargs='*', help='Data file(s) for validation (tfrecord).')
    p.add('--testfiles', nargs='*', help='Data file(s) for testing (tfrecord).')
    p.add('--lamda', type=float, default=0.01, help='Labeled ratio of true label state')
    # input configuration
    p.add('--obsmode', type=str, default='rgb',
          help='Observation input type. Possible values: rgb / depth / rgb-depth / vrf.')
    p.add('--mapmode', type=str, default='wall',
          help='Map input type with different (semantic) channels. ' +
               'Possible values: wall / wall-door / wall-roomtype / wall-door-roomtype')
    p.add('--sequence-length', dest='sequence_length', type=int,
                        default=23, help='length of the generated sequences')
    p.add('--map_pixel_in_meters', type=float, default=0.02,
          help='The width (and height) of a pixel of the map in meters.')

    p.add('--init_particles_distr', type=str, default='tracking',
          help='Distribution of initial particles. Possible values: tracking / one-room / two-rooms / all-rooms')
    p.add('--init_particles_std', nargs='*', default=["0.3", "0.523599"],  # tracking setting, 30cm, 30deg
          help='Standard deviations for generated initial particles. Only applies to the tracking setting.' +
               'Expects two float values: translation std (meters), rotation std (radians)')
    p.add('--trajlen', type=int, default=24,
          help='Length of trajectories. Assumes lower or equal to the trajectory length in the input data.')

    # PF-net configuration
    p.add('--num_particles', type=int, default=30, help='Number of particles in PF-net.')
    p.add('--resample', type=str, default='false',
          help='Resample particles in PF-net. Possible values: true / false.')
    p.add('--alpha_resample_ratio', type=float, default=1.0,
          help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. '
               'Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')
    p.add('--transition_std', nargs='*', default=["0.0", "0.0"],
                help='Standard deviations for transition model. Expects two float values: ' +
                     'translation std (meters), rotatation std (radians). Defaults to zeros.')

    # training configuration
    p.add('--batchsize', type=int, default=24, help='Minibatch size for training. Must be 1 for evaluation.')
    p.add('--bptt_steps', type=int, default=4,
          help='Number of backpropagation steps for training with backpropagation through time (BPTT). '
               'Assumed to be an integer divisor of the trajectory length (--trajlen).')
    p.add('--learningrate', type=float, default=0.0025, help='Initial learning rate for training.')
    p.add('--l2scale', type=float, default=4e-6, help='Scaling term for the L2 regularization loss.')
    p.add('--num_epochs', metavar='num_epochs', type=int, default=1, help='Number of epochs for training.')
    p.add('--decaystep', type=int, default=4, help='Decay the learning rate after every N epochs.')
    p.add('--decayrate', type=float, help='Rate of decaying the learning rate.')

    p.add('--load', type=str, default="", help='Load a previously trained model from a checkpoint file.')
    p.add('--logpath', type=str, default='',
          help='Specify path for logs. Makes a new directory under ./log/ if empty (default).')
    p.add('--seed', type=int, help='Fix the random seed of numpy and tensorflow if set to larger than zero.')
    p.add('--validseed', type=int,
          help='Fix the random seed for validation if set to larger than zero. ' +
               'Useful to evaluate with a fixed set of initial particles, which reduces the validation error variance.')
    p.add('--gpu', action='store_true', help='whether to use GPU')

    p.add('--labeledRatio', type=float, default=1.0, help='Labeled ratio of true label state')
    p.add_argument('--elbo_ratio', type=float, default=10.0,
                        help='elbo_ratio in loss')  # 1.0 [0.1, 0.3, 0.5, 0.7, 0.9]
    p.add('--splitRatio', type=float, default=1.0, help='The percentage of training data over all data.')
    p.add('--semi', type=int, default=0, help='Train semi or supervised way.')
    p.add('--NF-dyn', action='store_true', help='train using normalising flow')
    p.add('--trainType', dest='trainType', type=str,
                        default='DPF', choices=['DPF', 'SDPF', 'UDPF'],
                        help='train type: supervised, semi, unsupervised learning')
    p.add('--block-length', dest='block_length', type=int,
                        default=10, help='block length for pseudo-likelihood')  # dest: destination, origin=5
    p.add('--pos-noise', dest='pos_noise', type=float,
                        default=20.0,
                        help='sigma for the positional process noise')
    p.add('--vel-noise', dest='vel_noise', type=float,
                        default=20.0,
                        help='sigma for the velocity noise')
    p.add('--NF-lr', type=float, default=2.5, help='NF learning rate')
    p.add('--lr', type=float, default=1e-4,
                        help='learning rate')
    p.add('--resampler_type', type=str, default='ot', help='|ot|soft|')
    p.add('--measurement', type=str, default='NN', help='|CRNVP|cos|NN|CGLOW|gaussian|')
    p.add('--testing', action='store_true',
                        help='Check testing performance')

    p.add('--resume', action='store_true',
                        help='resume training from checkpoint')
    p.add('--pretrain_ae', action='store_true',
                        help='pretrain of autoencoder model')
    p.add('--e2e-train', action='store_false',
                        help='End to end training')
    p.add('--load-pretrainModel', action='store_true',
                        help='Load pretrain model')
    p.add('--std_x', type=float, default=10.0, help='std_x')
    p.add('--std_y', type=float, default=10.0, help='std_y')
    p.add('--std_t', type=float, default=0.1, help='std_t')
    p.add('--spatial', action='store_false', help='spatial transformer')
    p.add('--hiddensize', type=int, default=128, help='hidden size')
    p.add('--n_sequence', type=int, default=2, help='number of sequence in normalizing flows')
    p.add('--NF-cond', action='store_true',help='train using conditional normalising flow')
    p.add('--dropout-keep-ratio', type=float, default=0.3, help='1-dropout_ratio')
    p.add('--epsilon', type=float, default=0.1, help='epsilon in OT resampling')
    p.add('--scaling', type=float, default=0.75, help='scaling in OT resampling')
    p.add('--alpha', type=float, default=0.5, help='hyperparameter for soft resampling')
    p.add('--threshold', type=float, default=1e-3, help='threshold in OT resampling')
    p.add('--max_iter', type=int, default=100, help='max iteration in OT resampling')
    p.add('--init-with-true-state', type=bool, default=True
          ,help='init_with_true_state, default: true')
    p.add('--model-path', type=str, default='./model/e2e_model_bestval_e2e.pth',
                        help='path of saved model')
    params = p.parse_args(args=args)

    # fix numpy seed if needed
    if params.seed is not None and params.seed >= 0:
        np.random.seed(params.seed)

    # convert multi-input fileds to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)

    # convert boolean fields
    if params.resample not in ['false', 'true']:
        print ("The value of resample must be either 'false' or 'true'")
        raise ValueError
    params.resample = (params.resample == 'true')

    return params

if __name__ == "__main__":
    args = parse_args_disk()
    param = copy.deepcopy(args)
    for labeledRatio in param.labeledRatio:
        args.labeledRatio = labeledRatio
        print(args)

