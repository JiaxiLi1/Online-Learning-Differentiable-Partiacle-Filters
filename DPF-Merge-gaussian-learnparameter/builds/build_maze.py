import torch
import torch.nn as nn
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from nf.flows import  *
from utils import et_distance,wrap_angle
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def build_encoder_maze(hidden_size, state_dim, dropout_keep_ratio):
    encode=nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Dropout2d(p=1 - dropout_keep_ratio),
            nn.Linear(64, hidden_size),
            # nn.ReLU(True)
        )
    # encode=nn.Sequential( # input: 3*24*24
    #         nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False), # 16*12*12
    #         nn.ReLU(True),
    #         nn.BatchNorm2d(16),
    #         nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 32*6*6
    #         nn.ReLU(True),
    #         nn.BatchNorm2d(32),
    #         nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 64*3*3
    #         nn.ReLU(True),
    #         nn.BatchNorm2d(64),
    #         nn.Flatten(),
    #         nn.Dropout2d(p=1-dropout_keep_ratio),
    #         nn.Linear(64*3*3, hidden_size),
    #         # nn.ReLU(True)
    #     )
    return encode

def build_encoder_maze_flow(input_dim, output_dim):
    encode=nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, output_dim),
            # nn.ReLU(True)
        )
    return encode

def build_decoder_maze(hidden_size,dropout_keep_ratio=0.8):
    decode=nn.Sequential(
            nn.Linear(hidden_size, 3 * 3 * 64),
            nn.ReLU(True),
            nn.Unflatten(-1, (64, 3, 3)),  # -1 means the last dim, (64, 3, 3)
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False), # (32, 6,6)
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False), # (16, 12,12)
            nn.Dropout2d(p=1 - dropout_keep_ratio),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False), # (3, 24, 24)
            nn.Dropout2d(p=1 - dropout_keep_ratio),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    return decode

def build_particle_encoder_maze(hidden_size, state_dim=4, dropout_keep_ratio=0.8):
    particle_encode=nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Dropout2d(p=1 - dropout_keep_ratio),
            nn.Linear(64, hidden_size),
            # nn.ReLU(True)
        )
    return particle_encode

def build_likelihood_maze(hidden_size=128, state_dim=4,dropout_keep_ratio=0.8):
    likelihood=nn.Sequential(
            nn.Linear(2*hidden_size, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128,1),
            nn.Dropout2d(p=1 - dropout_keep_ratio),
            nn.Sigmoid()
        )
    return likelihood

def build_action_noise_maze(action_dim=3):
    mo_noise_generator = nn.Sequential(
        nn.Linear(2*action_dim, 32),
        nn.ReLU(True),
        nn.Linear(32, 32),
        nn.ReLU(True),
        nn.Linear(32, 3)
    )
    return  mo_noise_generator

class motion_update_maze(nn.Module):
    def __init__(self, mo_noise_generator, stds_particle):
        super().__init__()
        self.mo_noise_generator=mo_noise_generator
        self.std_x, self.std_y, self.std_t = stds_particle
    def forward(self, particles, actions, environment_state):
        batch_size, num_particle, dim = particles.shape[0], particles.shape[1], particles.shape[2]
        if batch_size == 1:
            if actions.dim() == 1:
                actions=actions.unsqueeze(0)
        actions1 = actions[:, None, :]
        #actions1[:,:,:2] += torch.normal(size=(actions1.shape[0],actions1.shape[1],2), std=self.action_std_xy, mean=0.0).to(device)
        #actions1[:, :, 2:] += torch.normal(size=(actions1.shape[0], actions1.shape[1], 1), std=self.action_std_t, mean=0.0).to(device)
        # action_input = (actions1.to(device)).repeat([1, particles.shape[1], 1])  # (32,100,3)
        # random_input = torch.randn(size=action_input.shape).to(device)
        # input = torch.cat([action_input, random_input], dim=-1)

        # delta = self.mo_noise_generator(input.float())
        #
        # delta -= torch.mean(delta, dim=1, keepdim=True)
        # noisy_actions = actions1

        # noise_x = torch.normal(mean=0., std=self.std_x, size=(batch_size, num_particle, 1)).to(device)
        # noise_y = torch.normal(mean=0., std=self.std_y, size=(batch_size, num_particle, 1)).to(device)
        # noise_t = torch.normal(mean=0., std=self.std_t, size=(batch_size, num_particle, 1)).to(device)

        noise_state = torch.normal(mean=0., std=0.5, size=(batch_size, num_particle, dim)).to(device)

        # new_x = particles[:, :, 0:1] + noisy_actions[:, :, 0:1] + noise_x
        # new_y = particles[:, :, 1:2] + noisy_actions[:, :, 1:2] + noise_y
        # new_z = particles[:, :, 2:3] + noisy_actions[:, :, 2:3] + noise_t

        moved_particles = particles + actions1.repeat(1, particles.shape[1], 1) + noise_state
        # moved_particles = torch.cat([new_x, new_y, new_z], dim=-1)

        return moved_particles, noise_state

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])#
        return out