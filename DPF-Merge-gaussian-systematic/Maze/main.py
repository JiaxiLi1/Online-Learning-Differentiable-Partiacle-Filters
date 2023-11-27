import torch
from arguments import parse_args
import os
from torch.utils.tensorboard import SummaryWriter
from DPFs import DPF # ,AE, SWAE
from dataset import MazeDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import cv2
from utils import compute_sq_distance

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not os.path.isdir('logs'):
    os.mkdir('logs')

def get_data(args):
    # data_type:# whether training data or validation data
    maze_Train = MazeDataset(data_path='./data/100s', filename="nav01_train", split_ratio=args.split_ratio,
                             data_type=True)
    maze_Valid = MazeDataset(data_path='./data/100s', filename="nav01_train", split_ratio=args.split_ratio,
                             data_type=False)

    # compute some statistics of the training data
    return maze_Train, maze_Valid, maze_Train.get_statistic()

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
    else:
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

def train(args, logger, run_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    param = copy.deepcopy(args)

    train_dataset, valid_dataset, statistics = get_data(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, drop_last=True)

    for labeledRatio in [param.labeledRatio]:

        args.labeledRatio = labeledRatio
        print(args)

        ## AutoEncoder
        if args.pretrain:
            model = get_model(args)
            optim = get_optim(args, model)

            os.mkdir(os.path.join('logs', run_id, 'models'))

            for epoch in range(args.epochs):
                total_loss = []
                model.train()

                for iteration, input in enumerate(train_loader):

                    model.zero_grad()

                    loss = model.forward(input)
                    loss.backward()
                    optim.step()
                    total_loss.append(loss.detach().cpu().numpy())

                    print(f'iteration: {iteration} ==> loss: {loss.detach().cpu().numpy()}')
                print(f'epoch: {epoch} ==> total loss: {np.mean(total_loss)}')

                model.eval()
                eval_loss_all = []
                with torch.no_grad():
                    for iteration, input in enumerate(valid_loader):
                        model.zero_grad()

                        eval_loss = model.forward(input)
                        eval_loss_all.append(eval_loss.detach().cpu().numpy())
                        # print(f'iteration: {iteration} ==> loss: {eval_loss.detach().cpu().numpy()}')
                    print(f'epoch: {epoch} ==> eval loss: {np.mean(eval_loss_all)}')

            torch.save(model.state_dict(), os.path.join(
                'logs', run_id, 'models', 'autoencoder_model'))
            torch.save(optim.state_dict(), os.path.join(
                'logs', run_id, 'models', 'autoencoder_optim'))

        else:

            DPF = get_model(args)
            optimizer_DPF = get_optim(args, DPF)

            os.mkdir(os.path.join('logs', run_id, 'models'))
            os.mkdir(os.path.join('logs', run_id, 'data'))

            cnt =0
            best_eval =10000

            maps = cv2.imread(args.maps_path)
            maps = cv2.resize(maps, dsize=(24,24), interpolation=cv2.INTER_CUBIC)
            maps = (maps-maps.mean())/maps.std()
            maps = torch.tensor(maps).to(device).float()
            maps = maps.reshape([-1, 24, 24, 3]).permute(0, 3, 1, 2)
            # for epoch in tqdm(range(args.epochs)):
            for epoch in range(args.epochs):

                total_loss = []
                total_loss_lass = []
                DPF.train()

                for iteration, input in enumerate(train_loader):
                    # if iteration!=0:
                    #     continue
                    input=[x[:,:50] for x in input]
                    cnt = cnt + 1

                    DPF.zero_grad()

                    loss_all, loss_sup, loss_last, loss_pseud_lik, loss_ae, pred, particle_list, particle_weight_list, states, obs_likelihood = DPF.forward(input, statistics, train=True, maps=maps)
                    loss_all.backward()
                    optimizer_DPF.step()
                    total_loss.append(loss_all.detach().cpu().numpy())
                    total_loss_lass.append(loss_last.detach().cpu().numpy())
                    logger.add_scalar('train/loss_last', loss_last, cnt)
                    logger.add_scalar('train/loss', loss_all, cnt)
                    # print(f'total loss: {loss_all.detach().cpu().numpy()}; sup loss: {loss_sup.detach().cpu().numpy()}; unsup loss: {loss_pseud_lik.detach().cpu().numpy()}; ae loss: {loss_ae.detach().cpu().numpy()}')

                    if args.trainType == 'sup':
                        print(
                            f'total loss: {loss_all.detach().cpu().numpy()}; loss_sup: {loss_sup.detach().cpu().numpy()}; obs_likelihood: {obs_likelihood.detach().cpu().numpy()}; loss_ae: {loss_ae.detach().cpu().numpy()};')
                    elif args.trainType == 'semi':
                        print(
                            f'total loss: {loss_all.detach().cpu().numpy()}; loss_sup: {loss_sup.detach().cpu().numpy()}; loss_pseud_lik: {loss_pseud_lik.detach().cpu().numpy()}; loss_ae: {loss_ae.detach().cpu().numpy()};')

                print(f"epoch: {epoch}, loss: {np.mean(total_loss)}")


                DPF.eval()
                eval_loss_all = []
                eval_loss_all_relative = []
                eval_loss_last = []
                eval_loss_last_relative = []
                with torch.no_grad():
                    for iteration, input in enumerate(valid_loader):
                        # if iteration!=0:
                        #     continue
                        input=[x[:,:50] for x in input]

                        DPF.zero_grad()

                        if args.dataset == 'maze':
                            loss_all, loss_sup, loss_last, loss_pseud_lik, loss_ae, pred, particle_list, particle_weight_list, states, obs_likelihood = DPF.forward(input, statistics, train=False, maps=maps)

                            rmse_relative = torch.sqrt(compute_sq_distance(pred[..., :2], states[..., :2], statistics[2]).mean())
                            rmse = torch.sqrt(torch.sum((pred-states)[:,:,:2]**2,dim=-1).mean())
                            rmse_last_relative = torch.sqrt(
                                compute_sq_distance(pred[:, -1, :2], states[:, -1, :2], statistics[2]).mean())
                            rmse_last = torch.sqrt(torch.sum((pred - states)[:, -1, :2] ** 2,dim=-1).mean())
                            eval_loss_all_relative.append(rmse_relative.detach().cpu().numpy())
                            eval_loss_all.append(rmse.detach().cpu().numpy())
                            eval_loss_last_relative.append(rmse_last_relative.detach().cpu().numpy())
                            eval_loss_last.append(rmse_last.detach().cpu().numpy())

                            # debug:
                            if iteration < 3:
                                if args.trainType == 'sup':
                                    print(f'total loss: {loss_all.detach().cpu().numpy()}; loss_sup: {loss_sup.detach().cpu().numpy()}; obs_likelihood: {obs_likelihood.detach().cpu().numpy()}; loss_ae: {loss_ae.detach().cpu().numpy()};')
                                elif args.trainType == 'semi':
                                    print(
                                        f'total loss: {loss_all.detach().cpu().numpy()}; loss_sup: {loss_sup.detach().cpu().numpy()}; loss_pseud_lik: {loss_pseud_lik.detach().cpu().numpy()}; loss_ae: {loss_ae.detach().cpu().numpy()};')

                            # eval_loss_all.append(loss_sup.detach().cpu().numpy())
                            # eval_loss_last.append(loss_last.detach().cpu().numpy())


                    print(f"epoch: {epoch}, val_rmse_relative: {np.mean(eval_loss_all_relative)}, val_rmse_last_relative: {np.mean(eval_loss_last_relative)},"
                          f"val_rmse: {np.mean(eval_loss_all)}, val_rmse_last: {np.mean(eval_loss_last)}")


                log_eval_all = np.mean(eval_loss_all_relative)
                logger.add_scalar('eval/loss', log_eval_all, cnt)
                log_eval_last = np.mean(eval_loss_last)
                logger.add_scalar('eval/loss_last', log_eval_last, cnt)


                if log_eval_all < best_eval:
                    best_eval = log_eval_all
                    torch.save(DPF.state_dict(), os.path.join(
                        'logs', run_id, 'models', 'model_best'+str(labeledRatio)))
                    torch.save(optimizer_DPF.state_dict(), os.path.join(
                        'logs', run_id, 'models', 'optim_best'+str(labeledRatio)))
                    #debug
                    print(f'Eval saving at {epoch}')
                    np.savez(os.path.join('logs',run_id, "data", "torch_maze_.npz"),
                             weight=particle_weight_list.detach().cpu().numpy(), particle=particle_list.detach().cpu().numpy(),
                             pred=pred.detach().cpu().numpy(), states=states.detach().cpu().numpy())

            torch.save(DPF.state_dict(), os.path.join(
                'logs', run_id, 'models', 'model_final'+str(labeledRatio)))
            torch.save(optimizer_DPF.state_dict(), os.path.join(
                'logs', run_id, 'models', 'optim_final'+str(labeledRatio)))

            run_id = str(int(run_id)+1)

if __name__ == "__main__":
    args = parse_args()
    logger, run_id = get_logger()
    save_args(args, run_id)
    setup_seed(args.seed)
    train(args, logger, run_id)
