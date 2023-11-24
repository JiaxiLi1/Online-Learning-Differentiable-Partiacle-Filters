import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from utils import wrap_angle, compute_statistics, noisyfy_data
import matplotlib.pyplot as plt


class MazeDataset(Dataset):

    def __init__(self, data_path, filename, split_ratio, data_type=True, steps_per_episode=100, transform=None):
        self.data_path = data_path
        self.filename = filename
        self.steps_per_episode = steps_per_episode
        self.data = dict(np.load(os.path.join(data_path, filename + '.npz'), allow_pickle=True))
        self.transform = transform

        for key in self.data.keys():
            self.data[key] = np.reshape(self.data[key],
                                        [-1, self.steps_per_episode] + list(self.data[key].shape[1:])).astype('float32')

        train_size= int(self.data['pose'].shape[0]*split_ratio)
        # convert degrees into radients and
        for key in ['pose', 'vel']:
            self.data[key][:, :, 2] *= np.pi / 180
        # angles should be between -pi and pi
        self.data['pose'][:, :, 2] = wrap_angle(self.data['pose'][:, :, 2])

        abs_d_x = (self.data['pose'][:, 1:, 0:1] - self.data['pose'][:, :-1, 0:1])
        abs_d_y = (self.data['pose'][:, 1:, 1:2] - self.data['pose'][:, :-1, 1:2])
        d_theta = wrap_angle(self.data['pose'][:, 1:, 2:3] - self.data['pose'][:, :-1, 2:3])
        s = np.sin(self.data['pose'][:, :-1, 2:3])
        c = np.cos(self.data['pose'][:, :-1, 2:3])
        rel_d_x = c * abs_d_x + s * abs_d_y
        rel_d_y = s * abs_d_x - c * abs_d_y

        self.data['rgbd'] = noisyfy_data(self.data['rgbd'])

        # data_type:# whether training data or validation data
        if data_type:
            sample = {'o': self.data['rgbd'][:train_size, :, :, :, :3], #.transpose((0, 1, 4, 2, 3)), # for pytorch, the channels are in front of width*height
                      's': self.data['pose'][:train_size, :, :],
                      # 'a': data['vel'][:, 1:, :],
                      'a': np.concatenate([rel_d_x[:train_size], rel_d_y[:train_size], d_theta[:train_size]], axis=-1),
                      # noisyfying action but not use in Semi_DPF, since it cannot compute the prior density
                      # 'a': np.concatenate([rel_d_x[:train_size], rel_d_y[:train_size], d_theta[:train_size]], axis=-1)* np.random.normal(1.0, 0.1 * 1.0, self.data['pose'][:train_size, 1:, :].shape)
                      }
            # noisify action
            #sample['a'] = sample['a'] * np.random.normal(1.0, 0.1 * 1.0, sample['a'].shape)
            # compute statistics
            self.statistics = compute_statistics(sample)
        else:
            sample = {'o': self.data['rgbd'][train_size:, :, :, :, :3],
                      # .transpose((0, 1, 4, 2, 3)), # for pytorch, the channels are in front of width*height
                      's': self.data['pose'][train_size:, :, :],
                      # 'a': data['vel'][:, 1:, :],
                      'a': np.concatenate([rel_d_x[train_size:], rel_d_y[train_size:], d_theta[train_size:]], axis=-1)
                      # 'a': np.concatenate([rel_d_x[train_size:], rel_d_y[train_size:], d_theta[train_size:]], axis=-1)* np.random.normal(1.0, 0.1 * 1.0, self.data['pose'][train_size:, 1:, :].shape)
                      }
            # noisify action
            #sample['a'] = sample['a'] * np.random.normal(1.0, 0.1 * 1.0, sample['a'].shape)
        self.sample = sample

    def __len__(self):
        o, s, a = self.sample.keys()
        return len(self.sample[s])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        o, s, a = self.sample.keys()
        data_idx = dict()
        data_idx[o] = self.sample[o][idx]
        data_idx[s] = self.sample[s][idx]
        data_idx[a] = self.sample[a][idx]

        if self.transform:
            data_idx = self.transform(data_idx)

        return (data_idx[s], data_idx[a], data_idx[o])

    def get_statistic(self):
        return self.statistics

class ToyDiskDataset(Dataset):
    def __init__(self, data_path, filename, datatype="train_data"):
        # datatype: train_data, val_data, test_data
        self.data_path=data_path
        self.filename=filename

        files = os.listdir(self.data_path)

        self.train_files = \
            [os.path.join(self.data_path, f) for f in files
             if f.startswith(self.filename) and
             'train' in f]
        self.val_files = \
            [os.path.join(self.data_path, f) for f in files
             if f.startswith(self.filename) and
             'val' in f]
        self.test_files = \
            [os.path.join(self.data_path, f) for f in files
             if f.startswith(self.filename) and
             'test' in f]

        self.train_data=sorted(self.train_files)
        self.val_data=sorted(self.val_files)
        self.test_data=sorted(self.test_files)

        if datatype=="train_data":
            loadData=self.train_data
        elif datatype=="val_data":
            loadData = self.val_data
        else:
            loadData = self.test_data

        for index in range(3): # for index in range(6):
            data = dict(np.load(loadData[index], allow_pickle=True))[datatype].item()
            if index == 0:
                self.start_image = data['start_image']
                self.start_state = data['start_state']
                self.image = data['image']
                self.state = data['state']
                self.q = data['q']
                self.visible = data['visible']
            else:
                self.start_image = np.concatenate((self.start_image, data['start_image']), axis=0)
                self.start_state = np.concatenate((self.start_state, data['start_state']), axis=0)
                self.image = np.concatenate((self.image, data['image']), axis=0)
                self.state = np.concatenate((self.state, data['state']), axis=0)
                self.q = np.concatenate((self.q, data['q']), axis=0)
                self.visible = np.concatenate((self.visible, data['visible']), axis=0)

        self.data_size = len(self.start_image)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.start_image[idx], self.start_state[idx], self.image[idx], self.state[idx], self.q[idx], self.visible[idx])


if __name__ == "__main__":
    data_path = './data/tr2400val300test300/'
    filename = 'toy_pn=0.1_d=5_const'

    dataset=ToyDiskDataset(data_path, filename)

    print(dataset.__len__())
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    i = 0
    for index, (start_image, start_state, image, state, q, visible) in enumerate(loader):
        print(image.numpy().shape)
        plt.imshow(image.numpy().astype(float)[0, 0])
        plt.show()
        i += 1
        if i > 10:
            break
    print("end")

    # # Dataset output is a torch.Tensor.
    # # tensor.numpy().astype(int)
    # dataset = MazeDataset(data_path='./data/100s', filename="nav01_train", split_ratio=0.9, data_type=False)
    # loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1)
    # i=0
    # for index, (states, actions, measurements) in enumerate(loader):
    #     print(states.numpy().shape)
    #     plt.imshow(measurements.numpy().astype(int)[0,0])
    #     plt.show()
    #     i+=1
    #     if i>10:
    #         break
    # print("end")

