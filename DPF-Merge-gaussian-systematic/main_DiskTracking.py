import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import time
from dataset import ToyDiskDataset, MazeDataset, get_dataflow
import torch
import numpy as np
import train
import lgssm
from torch.utils.data import Dataset, DataLoader
from arguments import parse_args_disk, parse_args_maze, parse_args_house3d
import random
# from DPFs_unchanged_house import DPF_disk, DPF_maze, DPF_house3d
# from DPFs_house import DPF_disk, DPF_maze, DPF_house3d
from DPFs import DPF_disk, DPF_maze, DPF_house3d

import cv2
import argparse


start_time = time.time()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model(args):
    if args.dataset == 'disk':
        model = DPF_disk(args)
    elif args.dataset =='maze':
        model = DPF_maze(args)
    elif args.dataset =='house3d':
        model = DPF_house3d(args)
    else:
        raise ValueError('Please select a dataset from {disk, maze, house3d}')
    if torch.cuda.is_available() and args.gpu:
        model = model.to('cuda')
    return model

def get_data(args):
    # data_type:# whether training data or validation data
    if args.dataset == 'maze':
        filename = args.mazeID + '_train'
        maze_Train = MazeDataset(data_path='./data/100s', filename=filename, split_ratio=args.split_ratio,
                                 data_type=True)
        maze_Valid = MazeDataset(data_path='./data/100s', filename=filename, split_ratio=args.split_ratio,
                                 data_type=False)
        return maze_Train, maze_Valid, maze_Train.get_statistic()
    elif args.dataset == 'disk':
        Disk_Train = ToyDiskDataset(data_path='./data/disk/TwentyfiveDistractors/',
                                    filename='toy_pn={}_d=25_const'.format(args.true_pos_noise),
                                    datatype="train_data")
        Disk_Val = ToyDiskDataset(data_path='./data/disk/TwentyfiveDistractors/',
                                  filename='toy_pn={}_d=25_const'.format(args.true_pos_noise),
                                  datatype="val_data")
        return Disk_Train, Disk_Val, None
    elif args.dataset == 'house3d':
        return None, None, None
    else:
        raise ValueError('Please select a dataset from {disk, maze}')


def get_test_data(args):
    # data_type:# whether training data or validation data
    if args.dataset == 'maze':
        filename=args.mazeID+'_test'
        maze_Test = MazeDataset(data_path='./data/100s', filename=filename, split_ratio=1.0,
                                 data_type=True)
        return maze_Test, maze_Test.get_statistic()
    elif args.dataset == 'disk':
        Disk_Test = ToyDiskDataset(data_path='./data/disk/TwentyfiveDistractors/',
                                   filename='toy_pn={}_d=25_const'.format(args.true_pos_noise),
                                   # threeDistractors_400, toy_pn=0.1_d=3_const#
                                   datatype="test_data")
        return Disk_Test, None
    else:
        raise ValueError('Please select a dataset from {disk, maze}')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_run_id(args):
    if args.dataset == 'maze':
        cnt = 'linear_dim{}_{}_NF^{}_{}_{}_{}_resample^{}_{}_{}_{}_{}_{}_elbo_{}_split_{}_mode_{}_slice_size_{}_onlinetype_{}'.format(args.dim, args.seed, args.NF_dyn, args.trainType, args.NF_lr,
                                                                    args.lr, args.resampler_type, args.measurement,
                                                                    args.dataset, args.mazeID, args.labeledRatio,
                                                                    args.lamda, args.elbo_ratio, args.split_ratio, args.learnType, args.slice_size, args.onlineType)
    else:
        cnt = '{}_NF^{}_{}_{}_{}_resample^{}_{}_{}_{}_{}_elbo_{}_mix'.format(args.seed, args.NF_dyn, args.trainType, args.NF_lr,
                                                                 args.lr, args.resampler_type, args.measurement,
                                                                 args.dataset, args.labeledRatio, args.lamda, args.elbo_ratio)

    return cnt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str,default='maze',
                        choices=['disk', 'maze', 'house3d'], help='experiment dataset')
    parsed, unknown = parser.parse_known_args()
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    # configuration
    if parsed.dataset =='disk':
        args = parse_args_disk()
    elif parsed.dataset == 'maze':
        args = parse_args_maze()
    elif parsed.dataset == 'house3d':
        args = parse_args_house3d()

    # config_path = './configs'
    # config_filename = 'train.conf'
    #
    # # Check if the config directory exists, if not, create it
    # if not os.path.exists(config_path):
    #     os.makedirs(config_path)

    seed=args.seed
    setup_seed(seed)
    dim = 2
    args.dim = dim
    # print(args)
    run_id=get_run_id(args)
    print(run_id)

    logs_dir=os.path.join('logs', run_id)
    model_dir=os.path.join('logs', run_id, "models")
    data_dir=os.path.join('logs', run_id, "data")

    dirs=[logs_dir, model_dir, data_dir]
    flags=[os.path.isdir(dir) for dir in dirs]
    for i,flag in enumerate(flags):
        if not flag:
            os.mkdir(dirs[i])


    initial_loc = torch.zeros([dim]).to(device).squeeze()
    initial_scale = torch.eye(dim).to(device).squeeze()
    if dim > 1:
        true_transition_mult = torch.ones([dim, dim]).to(device).squeeze()
        true_transition_mult_online = torch.ones([dim, dim]).to(device).squeeze()
        init_transition_mult = torch.diag(0.1 * torch.ones([dim])).to(device).squeeze()
        for i in range(dim):
            for j in range(dim):
                true_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
                true_transition_mult_online[i, j] = 0.20 ** (abs(i - j) + 1)
                # init_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
    else:
        true_transition_mult = (0.5 * torch.ones(dim)).to(device).squeeze()
        true_transition_mult_online = (0.8 * torch.ones(dim)).to(device).squeeze()
        init_transition_mult = (1.0 * torch.ones(dim)).to(device).squeeze()

    transition_scale = torch.eye(dim).to(device).squeeze()
    true_emission_mult = (0.5 * torch.ones(dim)).to(device).squeeze()
    true_emission_mult_online = (0.8 * torch.ones(dim)).to(device).squeeze()
    init_emission_mult = (0.1 * torch.ones(dim)).to(device).squeeze()

    init_proposal_scale_0 = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()
    init_proposal_scale_t = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()

    emission_scale = ((0.1 ** 0.5) * torch.eye(dim)).to(device).squeeze()

    num_timesteps = 51
    num_timesteps_online = 100
    batch_size = 10
    batch_size_online = 1
    total_timestep_online = 5000

    dataloader = train.get_synthetic_dataloader(
        lgssm.Initial(initial_loc, initial_scale).to(device),
        lgssm.Transition(true_transition_mult, transition_scale).to(device),
        lgssm.Emission(true_emission_mult, emission_scale).to(device),
        num_timesteps, batch_size)

    dataloader_val = train.get_synthetic_dataloader(
        lgssm.Initial(initial_loc, initial_scale).to(device),
        lgssm.Transition(true_transition_mult, transition_scale).to(device),
        lgssm.Emission(true_emission_mult, emission_scale).to(device),
        num_timesteps, batch_size)

    dataloader_online = train.get_synthetic_dataloader(
        lgssm.Initial(initial_loc, initial_scale).to(device),
        lgssm.Transition(true_transition_mult_online, transition_scale).to(device),
        lgssm.Emission(true_emission_mult_online, emission_scale).to(device),
        num_timesteps_online, batch_size_online)

    dataloader_online_val = train.get_synthetic_dataloader(
        lgssm.Initial(initial_loc, initial_scale).to(device),
        lgssm.Transition(true_transition_mult, transition_scale).to(device),
        lgssm.Emission(true_emission_mult, emission_scale).to(device),
        num_timesteps_online, batch_size_online)

    # offline task dataset
    if args.dataset == 'house3d':
        train_loader, num_train_batch = get_dataflow(args.trainfiles, args, is_training=True)
        valid_loader, num_val_batch = get_dataflow(args.valfiles, args, is_training=False)
    else:
        train_dataset, valid_dataset, statistics = get_data(args)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, drop_last=True)
        num_train_batch, num_val_batch = 50, 5
    dpf = get_model(args)

    if args.dataset == 'maze':
        maps = cv2.imread(args.maps_path)
        maps = cv2.resize(maps, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
        maps = (maps - maps.mean()) / maps.std()
        maps = torch.tensor(maps).to(device).float()
        maps = maps.reshape([-1, 24, 24, 3]).permute(0, 3, 1, 2)
        environment_data = [maps, statistics]
    else:
        environment_data = None

    if not args.testing:
        dpf.train_val(dataloader, dataloader_val, run_id, environment_data=environment_data,
                      num_train_batch=num_train_batch, num_val_batch=num_val_batch)
        torch.save(dpf, './model/dpf.pkl')

    # online task dataset
    args.mazeID = 'nav02'
    args.batchsize = 1
    args.learnType = 'online'
    if args.dataset == 'house3d':
        train_loader, num_train_batch = get_dataflow(args.trainfiles, args, is_training=True)
        valid_loader, num_val_batch = get_dataflow(args.valfiles, args, is_training=False)
    else:
        train_dataset, valid_dataset, statistics = get_data(args)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, drop_last=True)
        num_train_batch, num_val_batch = 50, 50
    # dpf = get_model(args)

    if args.dataset == 'maze':
        maps = cv2.imread(args.maps_path)
        maps = cv2.resize(maps, dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
        maps = (maps - maps.mean()) / maps.std()
        maps = torch.tensor(maps).to(device).float()
        maps = maps.reshape([-1, 24, 24, 3]).permute(0, 3, 1, 2)
        environment_data = [maps, statistics]
    else:
        environment_data = None

    if not args.testing:
        dpf.train_val(dataloader_online, dataloader_online_val, run_id, environment_data=environment_data,
                      num_train_batch=num_train_batch, num_val_batch=num_val_batch)
        torch.save(dpf, './model/dpf.pkl')


    # if args.dataset =='maze':
    #     args.batchsize=20
    #     environment_data = [maps, statistics]
    # elif args.dataset == 'disk':
    #     args.batchsize=50
    #     environment_data = None
    # elif args.dataset == 'house3d':
    #     environment_data = None
    #
    # if args.dataset == 'house3d':
    #     test_loader, num_test_batch = get_dataflow(args.testfiles, args, is_training=False)
    # else:
    #     test_dataset, statistics = get_test_data(args)
    #     test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, drop_last=True)
    #     num_test_batch = 1e10
    # dpf.testing(test_loader, run_id=run_id, model_path=args.model_path,
    #             environment_data = environment_data, num_test_batch=num_test_batch)
    # print(run_id)
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_elapsed_time = "{:.2f}".format(elapsed_time)
    print(f"The process took {formatted_elapsed_time} seconds to complete.")


