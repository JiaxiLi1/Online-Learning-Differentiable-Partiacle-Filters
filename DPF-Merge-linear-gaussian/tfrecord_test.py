import torch
from tfrecord.torch.dataset import TFRecordDataset
import tensorflow as tf

tfrecord_path = "./data/train.tfrecords"
index_path = None
description = {'states': 'byte', 'depth': 'byte', 'roomID': 'byte', 'map_roomtype': 'byte', 'odometry': 'byte', 'rgb': 'byte', 'map_door': 'byte', 'map_roomid': 'byte', 'map_wall': 'byte', 'houseID': 'byte'}
dataset = TFRecordDataset(tfrecord_path, index_path, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=64)
data = next(iter(loader))

tf.train.Example.FromString