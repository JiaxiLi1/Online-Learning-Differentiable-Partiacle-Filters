import torch
from torch import nn
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from utils import et_distance
from nf.flows import *
from nf.cglow.CGlowModel import CondGlowModel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def build_encoder_disk(hidden_size):
    encode=nn.Sequential(  # input: 3*120*120, 3*128*128
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16*60*60, 64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32*30*30, 32
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 16
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 8
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 4
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            # nn.Dropout2d(p=1 - args.dropout_keep_ratio),
            nn.Linear(256 * 4 * 4, hidden_size),
            # nn.ReLU(True),
            # nn.Linear(256, self.hidden_size),
            # nn.ReLU(True) # output size: 32
        )
    return encode

def build_encoder_cglow_disk(hidden_size):
    encode=nn.Sequential(  # input: 3*120*120, 3*128*128
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16*60*60, 64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32*30*30, 32
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 16
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 8
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 64*15*15, 4
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            # nn.Dropout2d(p=1 - args.dropout_keep_ratio),
            nn.Linear(256 * 4 * 4, 192),
            # nn.ReLU(True),
            # nn.Linear(256, self.hidden_size),
            # nn.ReLU(True) # output size: 32
        )
    return encode

def build_decoder_disk(hidden_size):
    decode=nn.Sequential(
            nn.Linear(hidden_size, 256*4*4),
            # nn.ReLU(True),
            # nn.Linear(256, 64 * 15 * 15),
            # nn.ReLU(True),
            nn.Unflatten(-1,(256, 4, 4)), # -1 means the last dim, (64, 15, 15)

            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 8
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 16
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 32
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False),  # (16, 60,60), 64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False),  # (3, 120, 120), 128
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    return decode


def build_decoder_cglow_disk(hidden_size):
    decode=nn.Sequential(
            nn.Linear(192, 256*4*4),
            # nn.ReLU(True),
            # nn.Linear(256, 64 * 15 * 15),
            # nn.ReLU(True),
            nn.Unflatten(-1,(256, 4, 4)), # -1 means the last dim, (64, 15, 15)

            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 8
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 16
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False),  # (32, 30,30), 32
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False),  # (16, 60,60), 64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False),  # (3, 120, 120), 128
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    return decode

def build_likelihood_disk(hidden_size, state_dim):
    likelihood=nn.Sequential(
            nn.Linear(2*hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64,64),
            nn.ReLU(True),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    return likelihood

def build_particle_encoder_disk(hidden_size, state_dim):
    particle_encode=nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_size),
            # nn.ReLU()
        )
    return particle_encode

def build_particle_encoder_cglow_disk(hidden_size, state_dim):
    particle_encode=nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 192),
            # nn.ReLU()
        )
    return particle_encode

def build_transition_model_disk(state_dim):
    transition=nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),)
    return transition

def motion_update_disk(particles, vel, environment_state):
    pos_noise = environment_state
    B, N, d = particles.shape

    vel_p = vel[:, None, :].repeat((1, N, 1))

    particles_xy_update = particles[:, :, :] + vel_p

    # noise
    position_noise = torch.normal(mean=0., std=pos_noise, size=(B, N, 2)).to(
        device)

    particles_update = particles_xy_update + position_noise

    return particles_update, position_noise