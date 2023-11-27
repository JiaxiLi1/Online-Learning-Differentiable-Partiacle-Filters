import torch
from arguments import parse_args
import os
from torch.utils.tensorboard import SummaryWriter
from DPFs import DPF # ,AE, SWAE
from dataset import MazeDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import copy
from dataset import ToyDiskDataset
import sys
import cv2
from utils import compute_sq_distance

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not os.path.isdir('logs'):
    os.mkdir('logs')

def get_data(args):
    if args.dataset == 'maze':

        # data_type:# whether training data or validation data
        # maze_Train = MazeDataset(data_path='./data/100s', filename="nav01_train", split_ratio=args.split_ratio,
        #                          data_type=True)
        # maze_Valid = MazeDataset(data_path='./data/100s', filename="nav01_train", split_ratio=args.split_ratio,
        #                          data_type=False)

        maze_Test = MazeDataset(data_path='./data/100s', filename="nav01_test", split_ratio=1.0,
                                 data_type=True)

        # compute some statistics of the training data
        return maze_Test, maze_Test.get_statistic()
    elif args.dataset == 'toy':

        # toy_Train = ToyDiskDataset(data_path='./data/tr1200val150test150/', filename='toy_pn=0.1_d=5_const',
        #                            datatype="train_data")
        # toy_Val = ToyDiskDataset(data_path='./data/tr1200val150test150/', filename='toy_pn=0.1_d=5_const',
        #                          datatype="val_data")
        # toy_Train = ToyDiskDataset(data_path='./data/twoObj/', filename='toy_pn=0.1_d=1_const',
        #                            datatype="train_data")
        # toy_Val = ToyDiskDataset(data_path='./data/twoObj/', filename='toy_pn=0.1_d=1_const',
        #                          datatype="val_data")

        # toy_Train = ToyDiskDataset(data_path='./data/threeObj/', filename='toy_pn=0.1_d=3_const',
        #                            datatype="train_data")
        # toy_Val = ToyDiskDataset(data_path='./data/threeObj/', filename='toy_pn=0.1_d=3_const',
        #                          datatype="val_data")

        toy_Train = ToyDiskDataset(data_path='./data/threeObj2/', filename='toy_pn=0.1_d=3_const',
                                   datatype="train_data")
        toy_Val = ToyDiskDataset(data_path='./data/threeObj2/', filename='toy_pn=0.1_d=3_const',
                                 datatype="val_data")

        # toy_Train = ToyDiskDataset(data_path='./data/oneObj/', filename='toy_pn=0.1_d=0_const',
        #                            datatype="train_data")
        # toy_Val = ToyDiskDataset(data_path='./data/oneObj/', filename='toy_pn=0.1_d=0_const',
        #                          datatype="val_data")

        return toy_Train, toy_Val, None

def get_logger():
    root = './logs'
    existings = os.listdir(root)
    cnt = str(len(existings))
    logger = SummaryWriter(os.path.join(root, cnt, 'tflogs'))

    return logger, cnt

def save_args(args, run_id):
    ret = vars(args)
    path = os.path.join('logs', run_id, 'args.conf')
    import json
    with open(path, 'w') as fout:
        json.dump(ret, fout)

def get_model(args):

    if args.pretrain:
        print("Autoencoder model")
        # model = AE(args)

        # print("SWAW model")
        # model = SWAE(args)
    else:
        if args.dataset == 'maze':
            model = DPF(args)

    if torch.cuda.is_available() and args.gpu:
        model = model.to('cuda')
    return model

def get_optim(args, model):
    if args.optim == 'RMSProp':
        optim = torch.optim.RMSprop(model.parameters(), lr= args.lr)
    elif args.optim == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr= args.lr)
    else:
        raise NotImplementedError
    return optim


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_checkpoint(args):
    try:
        model_checkpoint = torch.load('./model/models/model_best1.0')
        optimizer_checkpoint = torch.load('./model/models/optim_best1.0')
    except:
        print("\n[Error] Please make sure you have trained the model using main.py. ")
        print("And set the correct model path. \n")

    return model_checkpoint, optimizer_checkpoint


def test(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    test_dataset, statistics = get_data(args)
    args.batchsize = 20 # batchsize=1
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, drop_last=True)

    maps = cv2.imread(args.maps_path)
    maps = cv2.resize(maps.cpu().numpy(), dsize=(24, 24), interpolation=cv2.INTER_CUBIC)
    maps = torch.tensor(maps / 255.0).to(device).float()
    maps = maps.reshape([-1, 24, 24, 3]).permute(0, 3, 1, 2)

    DPF = get_model(args)
    optimizer_DPF = get_optim(args, DPF)

    model_checkpoint, optimizer_checkpoint = get_checkpoint(args)
    DPF.load_state_dict(model_checkpoint)
    optimizer_DPF.load_state_dict(optimizer_checkpoint)
    DPF.eval()

    test_loss_all = []
    test_loss_last = []
    with torch.no_grad():
        for iteration, input in enumerate(test_loader):
            DPF.zero_grad()

            if args.dataset == 'maze':
                loss_all, loss_sup, loss_last, loss_pseud_lik, loss_ae, pred, particle_list, particle_weight_list, states = DPF.forward(
                    input, statistics, train=False, maps=maps)

                rmse = torch.sqrt(compute_sq_distance(pred[..., :2], states[..., :2], statistics[2]).mean())
                rmse_last = torch.sqrt(compute_sq_distance(pred[:,-1,:2], states[:,-1,:2], statistics[2]).mean())

                # rmse = torch.sqrt(((pred - states)**2).mean())
                # rmse_last = torch.sqrt(((pred[:,-1,:] - states[:,-1,:])**2).mean())

                test_loss_all.append(rmse.detach().cpu().numpy())
                test_loss_last.append(rmse_last.detach().cpu().numpy())

                print(f"it:{iteration}, test_loss: {rmse.detach().cpu().numpy()}, test_loss_last: {rmse_last.detach().cpu().numpy()}")
            elif args.dataset == 'toy':
                pass

    if args.dataset == 'maze':
        print(f"test_loss: {np.mean(test_loss_all)}, test_loss_last: {np.mean(test_loss_last)}")
    elif args.dataset == 'toy':
        pass


if __name__ == "__main__":
    args = parse_args()
    logger, run_id = get_logger()
    save_args(args, run_id)
    setup_seed(args.seed)
    test(args)
