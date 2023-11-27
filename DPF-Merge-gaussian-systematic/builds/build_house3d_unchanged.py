import torch
import torch.nn as nn
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from nf.flows import  *
from utils import et_distance
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

def build_local_map_encoder_house3d(hidden_size, dropout_keep_ratio):
    encode=nn.Sequential( # input: 3*24*24
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False), # 16*12*12
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 32*6*6
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), # 64*3*3
            # nn.ReLU(True),
            nn.BatchNorm2d(32),
            # nn.Flatten(),
            # nn.Dropout2d(p=1-dropout_keep_ratio),
            # nn.Linear(64*3*3, hidden_size),
            # nn.ReLU(True)
        )
    return encode

def build_obs_feature_house3d(hidden_size, dropout_keep_ratio):
    encode=nn.Sequential( # input: 3*56*56
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False), # 16*28*28
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 32*14*14
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), # 32*14*14
            # nn.ReLU(True),
            nn.BatchNorm2d(32),
            # nn.Flatten(),
            # nn.Dropout2d(p=1-dropout_keep_ratio),
            # nn.Linear(64*7*7, hidden_size),
            # nn.ReLU(True)
        )
    return encode

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

def build_likelihood_house3d_spatial(hidden_size=128, state_dim=4):
    likelihood=nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16*28*28
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),  # 32*14*14
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Flatten(),
            nn.Linear(168, 32),
            nn.ReLU(True),
            nn.Linear(32,16),
            nn.ReLU(True),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    return likelihood

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