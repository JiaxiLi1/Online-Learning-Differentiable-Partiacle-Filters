import torch
import torch.nn as nn
import torch.nn.functional as F
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from nf.flows import  *
from utils import et_distance
import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def build_map_encoder_house3d(hidden_size, dropout_keep_ratio):
    encode=nn.Sequential( # input: 3*24*24
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1, bias=False), # 16*12*12
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 32*6*6
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 64*3*3
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Dropout2d(p=1-dropout_keep_ratio),
            nn.Linear(64*3*3, hidden_size),
            # nn.ReLU(True)
        )
    return encode

# def build_local_map_encoder_house3d(hidden_size, dropout_keep_ratio):
#     encode=nn.Sequential( # input: 3*24*24
#             nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, bias=False), # 16*12*12
#         )
#
#     return encode

class build_local_map_encoder_house3d(nn.Module):
    def __init__(self, hidden_size, dropout_keep_ratio):
        super(build_local_map_encoder_house3d, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv3 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv4 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=6, dilation=2, bias=True)
        self.conv5 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=9, dilation=3, bias=True)
        # self.dropout = nn.Dropout(p=1 - dropout_keep_ratio)

        # Initialize weights (He initialization)
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        # Initialize using variance scaling (He initialization)
        # nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')

        # First layer normalization
        self.layer_norm1 = nn.LayerNorm([64, 28, 28])
        # self.batch_norm1 = nn.BatchNorm2d(64)

        # Max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second set of convolutional layers
        self.conv6 = nn.Conv2d(64, 4, kernel_size=3, padding=1, bias=True)
        self.conv7 = nn.Conv2d(64, 4, kernel_size=5, padding=2, bias=True)

        # nn.init.kaiming_normal_(self.conv6.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='relu')
        # Initialize weights (He initialization)
        for conv in [self.conv6, self.conv7]:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

        # Second layer normalization
        self.layer_norm2 = nn.LayerNorm([8, 14, 14])
        # self.batch_norm2 = nn.BatchNorm2d(8)

    def forward(self, x):
        # start_time = time.time()
        out = [
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.conv4(x),
            self.conv5(x)
        ]
        # out = self.conv1(x)
        # end_time = time.time()
        #
        # # Compute the elapsed time
        # elapsed_time = end_time - start_time
        #
        # print(f"convs1 took {elapsed_time} seconds to run.")
        # # out1 = self.conv1(x)
        # # out2 = self.conv2(x)
        # # out3 = self.conv3(x)
        # # out4 = self.conv4(x)
        # # out5 = self.conv5(x)
        # start_time = time.time()
        out = torch.cat(out, dim=1)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.max_pool(out)
        # end_time = time.time()
        #
        # # Compute the elapsed time
        # elapsed_time = end_time - start_time
        #
        # print(f"convs2 took {elapsed_time} seconds to run.")
        # Second set of operations
        # out6 = self.conv6(out)
        # out7 = self.conv7(out)
        # start_time = time.time()
        out = [
            self.conv6(out),
            self.conv7(out)
        ]

        out = torch.cat(out, dim=1)
        out = self.layer_norm2(out)
        out = F.relu(out)
        # end_time = time.time()
        #
        # # Compute the elapsed time
        # elapsed_time = end_time - start_time
        #
        # print(f"convs3 took {elapsed_time} seconds to run.")
        # out = self.dropout(out)
        return out

# def build_local_map_encoder_house3d(hidden_size, dropout_keep_ratio):
#     encode=nn.Sequential( # input: 3*24*24
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False), # 16*12*12
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 32*6*6
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), # 64*3*3
#             # nn.ReLU(True),
#             nn.BatchNorm2d(32),
#             # nn.Flatten(),
#             # nn.Dropout2d(p=1-dropout_keep_ratio),
#             # nn.Linear(64*3*3, hidden_size),
#             # nn.ReLU(True)
#         )
#     return encode

class build_obs_feature_house3d(nn.Module):
    def __init__(self, hidden_size, dropout_keep_ratio):
        super(build_obs_feature_house3d, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv3 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=4, dilation=2, bias=True)
        self.conv4 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=8, dilation=4, bias=True)
        # self.dropout = nn.Dropout(p=1 - dropout_keep_ratio)

        # Initialize using variance scaling (He initialization)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')

        # Max pooling layer
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # First layer normalization
        self.layer_norm1 = nn.LayerNorm([384, 28, 28])

        # Second set of convolutional layers
        self.conv5 = nn.Conv2d(384, 16, kernel_size=3, padding=1, bias=True)

        # Initialize using variance scaling (He initialization)
        nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second layer normalization
        self.layer_norm2 = nn.LayerNorm([16, 14, 14])

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.max_pool1(out)
        out = self.layer_norm1(out)
        out = F.relu(out)

        # Second set of operations
        out5 = self.conv5(out)
        out = self.max_pool2(out5)
        out = self.layer_norm2(out)
        out = F.relu(out)
        # out = self.dropout(out)
        return out

# def build_obs_feature_house3d(hidden_size, dropout_keep_ratio):
#     encode=nn.Sequential( # input: 3*56*56
#             nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False), # 16*28*28
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 32*14*14
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), # 32*14*14
#             # nn.ReLU(True),
#             nn.BatchNorm2d(32),
#             # nn.Flatten(),
#             # nn.Dropout2d(p=1-dropout_keep_ratio),
#             # nn.Linear(64*7*7, hidden_size),
#             # nn.ReLU(True)
#         )
#     return encode

def build_encoder_house3d(hidden_size, dropout_keep_ratio):
    encode=nn.Sequential( # input: 3*56*56
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False), # 16*28*28
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 32*14*14
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 64*7*7
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Dropout2d(p=1-dropout_keep_ratio),
            nn.Linear(64*7*7, hidden_size),
            # nn.ReLU(True)
        )
    return encode

def build_obs_decoder_house3d(hidden_size):
    decode=nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False), # (32, 14,14)
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False), # (16, 28,28)
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False), # (3, 56, 56)
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    return decode

def build_decoder_house3d(hidden_size):
    decode=nn.Sequential(
            nn.Linear(hidden_size, 7 * 7 * 64),
            nn.ReLU(True),
            nn.Unflatten(-1, (64, 7, 7)),  # -1 means the last dim, (64, 7, 7)
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False), # (32, 14,14)
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False), # (16, 28,28)
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False), # (3, 56, 56)
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    return decode

def build_particle_encoder_house3d(hidden_size, state_dim=4):
    particle_encode=nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, hidden_size),
            # nn.ReLU(True)
        )
    return particle_encode

# def build_likelihood_house3d(hidden_size=128, state_dim=4):
#     likelihood=nn.Sequential(
#             nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16*28*28
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),  # 32*14*14
#             nn.ReLU(True),
#             nn.BatchNorm2d(8),
#             nn.Flatten(),
#             nn.Linear(72, 128),
#             nn.ReLU(True),
#             nn.Linear(128,64),
#             nn.ReLU(True),
#             nn.Linear(64,1),
#             nn.Sigmoid()
#         )
#     return likelihood

def build_likelihood_house3d(hidden_size, state_dim):
    likelihood=nn.Sequential(
            nn.Linear(2*hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64,64),
            nn.ReLU(True),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    return likelihood


# class LocallyConnected2D(nn.Module):
#     def __init__(self, in_channels, out_channels, output_size, kernel_size):
#         super(LocallyConnected2D, self).__init__()
#
#         self.kernel_size = kernel_size
#         self.output_size = output_size
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.weight = nn.Parameter(torch.Tensor(
#             out_channels, in_channels, output_size[0], output_size[1], kernel_size[0], kernel_size[1]
#         ))
#         self.bias = nn.Parameter(torch.Tensor(out_channels, output_size[0], output_size[1]))
#
#         nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
#         nn.init.zeros_(self.bias)
#
#     def forward(self, x):
#         output = []
#         for i in range(self.output_size[0]):
#             for j in range(self.output_size[1]):
#                 patch = x[:, :, i:i + self.kernel_size[0], j:j + self.kernel_size[1]]
#                 local_weight = self.weight[:, :, i, j, :, :]
#                 local_bias = self.bias[:, i, j]
#                 out_ij = F.conv2d(patch, local_weight, local_bias)
#                 output.append(out_ij.squeeze(-1).squeeze(-1))
#
#         output = torch.stack(output, dim=2)
#         output = output.view(x.size(0), self.out_channels, self.output_size[0], self.output_size[1])
#         return output


class build_likelihood_house3d_spatial(nn.Module):
    def __init__(self, hidden_size, state_dim):
        super(build_likelihood_house3d_spatial, self).__init__()
        self.local1 = nn.Conv2d(24, 8, kernel_size=(3, 3), padding=0)
        self.local2 = nn.Conv2d(24, 8, kernel_size=(3, 3), padding=0)

        # Max pooling layer
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Dense (fully connected) layer
        self.fc1 = nn.Linear(16 * 5 * 5, 1)  # Assuming the output shape after max_pool1 is (16, 5, 5)

    def forward(self, x):
        # x_pad1 = F.pad(x, pad=(1, 1, 1, 1, 0, 0, 0, 0))
        out1 = self.local1(x)
        out2 = self.local2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.max_pool1(out)
        # Flatten the tensor before passing it through the dense layer
        out = out.view(out.size(0), -1)
        # Dense layer
        out = self.fc1(out)

        return out

# def build_likelihood_house3d_spatial(hidden_size=128, state_dim=4):
#     likelihood=nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16*28*28
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),  # 32*14*14
#             nn.ReLU(True),
#             nn.BatchNorm2d(8),
#             nn.Flatten(),
#             nn.Linear(168, 32),
#             nn.ReLU(True),
#             nn.Linear(32,16),
#             nn.ReLU(True),
#             nn.Linear(16,1),
#             nn.Sigmoid()
#         )
#     return likelihood

def build_action_noise_house3d(action_dim=3):
    mo_noise_generator = nn.Sequential(
        nn.Linear(2*action_dim, 32),
        nn.ReLU(True),
        nn.Linear(32, 32),
        nn.ReLU(True),
        nn.Linear(32, 3)
    )
    return  mo_noise_generator

class motion_update_house3d(nn.Module):
    def __init__(self, mo_noise_generator, stds_particle):
        super().__init__()
        self.mo_noise_generator=mo_noise_generator
        self.std_x, self.std_y, self.std_t = stds_particle
    def forward(self, particles, actions, environment_state):
        std_x, std_y, std_t = environment_state
        batch_size, num_particle = particles.shape[0], particles.shape[1]
        noisy_actions = actions[:, None, :]

        theta = particles[:, :, 2:3]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        noise_x = torch.normal(mean=0., std=self.std_x, size=(batch_size, num_particle, 1)).to(device)
        noise_y = torch.normal(mean=0., std=self.std_y, size=(batch_size, num_particle, 1)).to(device)
        noise_t = torch.normal(mean=0., std=self.std_t, size=(batch_size, num_particle, 1)).to(device)
        noise_state = torch.cat([noise_x, noise_y, noise_t], dim=-1)

        new_x, new_y, new_theta = particles[:,:,0:1] + noisy_actions[:,:,0:1] + noise_x, \
                                  particles[:,:,1:2] + noisy_actions[:,:,1:2] + noise_y, \
                                  particles[:,:,2:3] + noisy_actions[:,:,2:3] + noise_t

        # new_x = particles[:, :, 0:1] + (
        #         noisy_actions[:, :, 0:1] * cos_theta + noisy_actions[:, :, 1:2] * sin_theta) + noise_x
        # new_y = particles[:, :, 1:2] + (
        #         noisy_actions[:, :, 0:1] * sin_theta - noisy_actions[:, :, 1:2] * cos_theta) + noise_y
        # new_theta = particles[:, :, 2:3] + noisy_actions[:, :, 2:3] + noise_t


        moved_particles = torch.cat([new_x, new_y, new_theta], dim=-1)

        return moved_particles, noise_state

class LSTMNet_house3d(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNet_house3d, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])#
        return out