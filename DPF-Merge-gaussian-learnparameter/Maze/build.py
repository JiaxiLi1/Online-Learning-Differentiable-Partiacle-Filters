import torch
import torch.nn as nn
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from nf.flows import  *
from utils import et_distance
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def build_encoder(hidden_size, dropout_keep_ratio):
    encode=nn.Sequential( # input: 3*24*24
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False), # 16*12*12
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

def build_decoder(hidden_size):
    decode=nn.Sequential(
            nn.Linear(hidden_size, 3 * 3 * 64),
            nn.ReLU(True),
            nn.Unflatten(-1, (64, 3, 3)),  # -1 means the last dim, (64, 3, 3)
            nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2, bias=False), # (32, 6,6)
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2, bias=False), # (16, 12,12)
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, padding=1, stride=2, bias=False), # (3, 24, 24)
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
    return decode

def build_particle_encoder(hidden_size, state_dim=4):
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

def build_likelihood(hidden_size=128, state_dim=4):
    likelihood=nn.Sequential(
            nn.Linear(hidden_size+state_dim, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    return likelihood

def build_action_noise(action_dim=3):
    mo_noise_generator = nn.Sequential(
        nn.Linear(2*action_dim, 32),
        nn.ReLU(True),
        nn.Linear(32, 32),
        nn.ReLU(True),
        nn.Linear(32, 3)
    )
    return  mo_noise_generator

def build_dyn_nf(n_sequence, state_dim, init_var=0.01):
    flows_dyn = [RealNVP(dim=state_dim) for _ in range(n_sequence)]

    for f in flows_dyn:
        f.zero_initialization(var=init_var)

    prior_dyn = MultivariateNormal(torch.zeros(state_dim).to(device), torch.eye(state_dim).to(device))

    nf_dyn = NormalizingFlowModel(prior_dyn, flows_dyn, device=device)

    return nf_dyn


def build_conditional_nf(n_sequence, hidden_size, state_dim, init_var=0.01, prior_mean=0.0, prior_std=1.0):
    flows = [RealNVP_cond(dim=state_dim, obser_dim=hidden_size) for _ in range(n_sequence)]

    for f in flows:
        f.zero_initialization(var=init_var)

    prior_init = MultivariateNormal(torch.zeros(state_dim).to(device) + prior_mean,
                                    torch.eye(state_dim).to(device) * prior_std**2)

    cond_model = NormalizingFlowModel_cond(prior_init, flows, device=device)

    return cond_model