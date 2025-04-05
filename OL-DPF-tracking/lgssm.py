import copy
import numpy as np
from torch.distributions import MultivariateNormal
import pykalman
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import aemath
import state
import math
import train
import inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def log_prob(distribution, value):
    """Log probability of value under distribution.

    Args:
        distribution: `torch.distributions.Distribution` of batch_shape either
            [batch_size, num_particles, ...] or
            [batch_size, ...] or
            [...] or `dict` thereof.
        value: `torch.Tensor` of size
            [batch_size, num_particles, ...] + distribution.event_shape
            or `dict` thereof

    Returns: `torch.Tensor` [batch_size, num_particles] or `dict` thereof
    """
    if isinstance(distribution, dict):
        return torch.sum(torch.cat([
            log_prob(v, value[k], non_reparam).unsqueeze(0)
            for k, v in distribution.items()
        ], dim=0), dim=0)
    elif isinstance(distribution, torch.distributions.Distribution):
        value_ndim = value.ndimension()
        batch_shape_ndim = len(distribution.batch_shape)
        event_shape_ndim = len(distribution.event_shape)
        value_batch_shape_ndim = value_ndim - event_shape_ndim
        if (
            (value_batch_shape_ndim == batch_shape_ndim) or
            ((value_batch_shape_ndim - 2) == batch_shape_ndim)
        ):
            distribution._validate_sample(value)
            logp = distribution.log_prob(value)
        elif (value_batch_shape_ndim - 1) == batch_shape_ndim:
            if len(value.shape) > 1:
                logp = distribution.log_prob(value.transpose(0, 1)).transpose(0, 1)
            else:
                logp = distribution.log_prob(value)
        else:
            raise RuntimeError(
                'Incompatible distribution.batch_shape ({}) and '
                'value.shape ({}).'.format(
                    distribution.batch_shape, value.shape))
        return torch.sum(logp.view(value.size(0), value.size(1), -1), dim=2)
    else:
        raise AttributeError(
            'distribution must be a dict or a torch.distributions.Distribution.\
            Got: {}'.format(distribution))

def build_encoder_maze(state_dim, dropout_keep_ratio, obs_dim):
    encode=nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Dropout(p=1 - dropout_keep_ratio),
            nn.Linear(64, state_dim),
            # nn.ReLU(True)
        )
    return encode

def build_particle_encoder_maze(hidden_size, state_dim=4, dropout_keep_ratio=0.8):
    particle_encode=nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Dropout(p=1 - dropout_keep_ratio),
            nn.Linear(64, hidden_size),
            # nn.ReLU(True)
        )
    return particle_encode

def obs_feature_maze(encodings, num_particles):
    if len(encodings.shape) == 1:
        encodings_obs = encodings.unsqueeze(-1)[:, None, :].repeat(1, num_particles, 1)
    else:
        encodings_obs = encodings[:, None, :].repeat(1, num_particles, 1)
    return encodings_obs

def state_feature_maze(particle_encoder, update_particles):
    particle_input = update_particles
    particle_encoder = particle_encoder.float()
    encodings_state = particle_encoder(particle_input.float())  # shape: (batch, num_particle, hidden_size)
    return encodings_state if len(update_particles.shape) == 3 else encodings_state.unsqueeze(-1)

def features_state_obs(encodings, num_particles,particle_encoder, update_particles):
    encodings_state = state_feature_maze(particle_encoder, update_particles)  # shape: (batch, particle_num, hidden_size)
    encodings_obs = obs_feature_maze(encodings, num_particles)  # shape: (batch_size, particle_num, hidden_size)
    return encodings_state, encodings_obs

class measurement_model_cnf(nn.Module):
    def __init__(self, particle_encoder, obs_encoder, CNF, type = None):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.obs_encoder = obs_encoder
        self.CNF = CNF
        self.type = type

    def forward(self, encodings, update_particles, environment_data=None, pretrain=False):
        self.hidden_dim = encodings.shape[-1] if len(encodings.shape) > 1 else 1
        n_batch, n_particles = update_particles.shape[:2]
        encodings_state, encodings_obs = features_state_obs(encodings, n_particles, self.particle_encoder,
                                                            update_particles)

        encodings_state = encodings_state.reshape([-1, self.hidden_dim])
        encodings_obs = encodings_obs.reshape([-1, self.hidden_dim])
        # if self.params.dataset =='maze':
        #     means, stds, encodings_maps = environment_data
        #     encodings_maps = encodings_maps.repeat(n_batch * n_particles, 1)
        #     encodings_state = torch.cat([encodings_state, encodings_maps], dim=-1)
        if self.type == 'nf':
            cnf_input = encodings_obs - encodings_state
            z, log_prob_z, log_det = self.CNF.forward(cnf_input)
        else:
            z, log_prob_z, log_det = self.CNF.forward(encodings_obs, encodings_state)
        likelihood = (log_prob_z + log_det).reshape([n_batch, n_particles])

        likelihood = likelihood - likelihood.max(dim=-1, keepdims=True)[0]
        if pretrain:
            return likelihood.exp() + 1e-12
        else:
            return likelihood

class Initial(nn.Module):
    def __init__(self, loc, scale):
        super(Initial, self).__init__()
        self.loc = loc
        self.scale = scale
        self.dim = self.loc.shape[0] if len(self.loc.shape)!= 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal
    def forward(self):
            return self.dist(self.loc, self.scale)


class Transition(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition, self).__init__()
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult##
        self.scale = scale#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        loc = self.mult * previous_latents[-1] if self.dim == 1 else torch.matmul(previous_latents[-1], self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)


class Emission(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission, self).__init__()
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult#
        self.scale = scale
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):
        mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        loc = mult * latents[-1] if self.dim ==1 else torch.matmul(latents[-1], mult)
        # print(loc)
        # print(1)
        # print(self.scale)
        # print(time)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_second_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_second_online, self).__init__()
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult##
        self.a = torch.tensor(-10.0).to(device)
        self.b = torch.tensor(3.0).to(device)
        self.scale = torch.tensor(1.0).to(device)#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = self.a * previous_latents[-1] / (1+self.b*(previous_latents[-1]**2)) #if self.dim == 1 else torch.matmul(previous_latents[-1], self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)

class Emission_second_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_second_online, self).__init__()
        self.c = torch.tensor(0.2).to(device)
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult#
        self.scale = torch.tensor(0.5).to(device)
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):

        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = latents[-1] #if self.dim ==1 else torch.matmul(latents[-1], mult)
        # print(loc)
        # print(1)
        # print(self.scale)
        # print(time)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_second_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_second_offline, self).__init__()
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult##
        self.a = torch.tensor(-10.0).to(device)
        self.b = torch.tensor(3.0).to(device)
        self.scale = torch.tensor(1.0).to(device)#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = previous_latents[-1] #if self.dim == 1 else torch.matmul(previous_latents[-1], self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)


class Emission_second_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_second_offline, self).__init__()
        self.c = torch.tensor(0.2).to(device)
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult#
        self.scale = torch.tensor(0.5).to(device)
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):

        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = (self.c*latents[-1]).exp() #if self.dim ==1 else torch.matmul(latents[-1], mult)
        # print(loc)
        # print(1)
        # print(self.scale)
        # print(time)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_third_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_third_online, self).__init__()
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult##
        self.a = torch.tensor(-10.0).to(device)
        self.b = torch.tensor(3.0).to(device)
        self.scale = ((0.1 + 9.9 * torch.tensor(0.4))**2).to(device)#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        scale = None
        loc = abs(previous_latents[-1])  #if self.dim == 1 else torch.matmul(previous_latents[-1], self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)

class Emission_third_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_third_online, self).__init__()
        self.c = torch.tensor(0.2).to(device)
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult#
        self.scale = ((0.1 + 9.9 * torch.tensor(0.1))**2).to(device)
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):

        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = (1.0/3 + 5.0)*(latents[-1]**2).log() #if self.dim ==1 else torch.matmul(latents[-1], mult)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_third_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_third_offline, self).__init__()
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult##
        self.a = torch.tensor(-10.0).to(device)
        self.b = torch.tensor(3.0).to(device)
        self.scale = ((0.1 + 9.9 * torch.tensor(0.3))**2).to(device)#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):

        loc = 0.5*abs(previous_latents[-1]) #if self.dim == 1 else torch.matmul(previous_latents[-1], self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)


class Emission_third_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_third_offline, self).__init__()
        self.c = torch.tensor(0.2).to(device)
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult#
        self.scale = ((0.1 + 9.9 * torch.tensor(0.2))**2).to(device)
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):

        loc = (1.0/3)*(latents[-1]**2).log() #if self.dim ==1 else torch.matmul(latents[-1], mult)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_fourth_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_fourth_offline, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult##
        self.scale = (((0.1 + 9.9 * torch.tensor(0.3))**2).to(device))*scale#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = 0.5 * abs(previous_latents[-1]) if self.dim == 1 else torch.matmul(0.5 * abs(previous_latents[-1]), self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)


class Emission_fourth_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_fourth_offline, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult#
        self.scale = (((0.1 + 9.9 * torch.tensor(0.2))**2).to(device))*scale
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):
        mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        loc = (1.0/3)*(latents[-1]**2).log() if self.dim ==1 else torch.matmul((1.0/3)*(latents[-1]**2).log(), mult)
        # print(loc)
        # print(1)
        # print(self.scale)
        # print(time)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_fourth_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_fourth_online, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult##
        self.scale = (((0.1 + 9.9 * torch.tensor(0.4))**2).to(device))*scale#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = abs(previous_latents[-1]) if self.dim == 1 else torch.matmul(abs(previous_latents[-1]), self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)

class Emission_fourth_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_fourth_online, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult#
        self.scale = (((0.1 + 9.9 * torch.tensor(0.1))**2).to(device))*scale
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):
        mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        loc = (1.0/3 + 5.0)*(latents[-1]**2).log() if self.dim ==1 else torch.matmul((1.0/3)*(latents[-1]**2).log(), mult)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_fifth_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_fifth_offline, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult##
        self.a = torch.tensor(-10.0).to(device)
        self.b = torch.tensor(3.0).to(device)
        self.scale = 1.0*scale#nn.Parameter(torch.Tensor([scale]).squeeze())#
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = ((self.a * previous_latents[-1]) /(1 + (self.b * (previous_latents[-1]**2)))) if self.dim == 1 else torch.matmul(((self.a * previous_latents[-1]) /(1 + (self.b * (previous_latents[-1]**2)))), self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)


class Emission_fifth_offline(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_fifth_offline, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult#
        self.scale = 0.5*scale
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):
        mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        loc = latents[-1] if self.dim ==1 else torch.matmul(latents[-1], mult)
        # print(loc)
        # print(1)
        # print(self.scale)
        # print(time)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)

class Transition_fifth_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Transition_fifth_online, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult##
        self.scale = 1.0*scale
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()

        loc = previous_latents[-1] if self.dim == 1 else torch.matmul(previous_latents[-1], self.mult)
        return state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            state.BatchShapeMode.FULLY_EXPANDED)

class Emission_fifth_online(nn.Module):
    def __init__(self, init_mult, scale):
        super(Emission_fifth_online, self).__init__()
        self.mult = init_mult.squeeze().clone()#init_mult#
        self.scale = 0.5*scale
        self.c = torch.tensor(0.2).to(device)
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):
        mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        loc = ((self.c*latents[-1]).exp()) if self.dim ==1 else torch.matmul(((self.c*latents[-1]).exp()), mult)
        return state.set_batch_shape_mode(
                self.dist(loc , self.scale),
                state.BatchShapeMode.FULLY_EXPANDED)


class Proposal(nn.Module):
    def __init__(self, scale_0, scale_t, device):
        super(Proposal, self).__init__()
        self.scale_0_vector = nn.Parameter(scale_0.squeeze().clone())#scale_0#
        self.scale_t_vector = nn.Parameter(scale_t.squeeze().clone())#scale_t#

        self.dim = self.scale_0_vector.shape[0] if len(self.scale_0_vector.shape) != 0 else 1

        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

        self.lin_0 = nn.Linear(self.dim, self.dim, bias=False).to(device)
        self.lin_t = nn.Linear(2*self.dim, self.dim, bias=False).to(device)

    def forward(self, previous_latents=None, time=None, observations=None):
        # self.scale_0 = torch.nn.functional.softplus(torch.diag(self.scale_0_vector if self.dim != 1 else self.scale_0_vector.unsqueeze(-1)).squeeze())
        # # self.scale_t = torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze()
        # self.scale_t = torch.nn.functional.softplus(torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze())
        min_value = 0.0001

        self.scale_0 = torch.diag(self.scale_0_vector if self.dim != 1 else self.scale_0_vector.unsqueeze(-1)).squeeze()
        self.scale_t = torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze()

        # Replace negative values with min_value
        self.scale_0[self.scale_0 < 0] = min_value
        self.scale_t[self.scale_t < 0] = min_value

        if time == 0:
            return state.set_batch_shape_mode(
                self.dist(self.lin_0(observations[0].unsqueeze(-1) if self.dim ==1 else observations[0]).squeeze(-1), self.scale_0),
                state.BatchShapeMode.BATCH_EXPANDED)
        else:
            if time == 0.1:
                time = 0
            num_particles = previous_latents[-1].shape[1]
            a=self.lin_t(torch.cat(
                            [previous_latents[-1].unsqueeze(-1) if self.dim ==1 else previous_latents[-1],
                             (observations[time].view(-1, 1, 1) if self.dim ==1 else observations[time].unsqueeze(1)).repeat(1, num_particles, 1)],
                            dim=2
                        ).view(-1, 2*self.dim)).squeeze(-1).view((-1, num_particles) if self.dim ==1 else (-1, num_particles, self.dim))
            # mask1 = (a < 0)
            # a[mask1] = 0.0
            # print(self.scale_t)
            # print(a)
            return state.set_batch_shape_mode(
                self.dist(
                        a,
                        self.scale_t),
                state.BatchShapeMode.FULLY_EXPANDED)

class Proposal_rnn(nn.Module):
    def __init__(self, scale_0, scale_t, device):
        super(Proposal_rnn, self).__init__()
        self.scale_0_vector = nn.Parameter(scale_0.squeeze().clone())#scale_0#
        self.scale_t_vector = nn.Parameter(scale_t.squeeze().clone())#scale_t#

        self.dim = self.scale_0_vector.shape[0] if len(self.scale_0_vector.shape) != 0 else 1

        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

        self.lin_0 = nn.Linear(self.dim, self.dim, bias=False).to(device)
        self.rnn_cell = nn.GRUCell(self.dim, self.dim) #nn.Linear(2*self.dim, self.dim, bias=False).to(device)
        self.lin_t = nn.Linear(self.dim, self.dim, bias=False).to(device)
    def forward(self, previous_latents=None, time=None, observations=None):
        self.scale_0 = torch.diag(self.scale_0_vector if self.dim != 1 else self.scale_0_vector.unsqueeze(-1)).squeeze()
        self.scale_t = torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze()

        if time == 0:
            return state.set_batch_shape_mode(
                self.dist(self.lin_0(observations[0].unsqueeze(-1) if self.dim ==1 else observations[0]).squeeze(-1), self.scale_0),
                state.BatchShapeMode.BATCH_EXPANDED)
        else:
            num_particles = previous_latents[-1].shape[1]

            previous_latent_input = previous_latents[-1].unsqueeze(-1).reshape(-1, self.dim) if self.dim == 1 else previous_latents[-1].reshape(-1, self.dim)
            observation_input = (observations[time].view(-1, 1, 1) if self.dim == 1 else observations[time].unsqueeze(1)).repeat(1,num_particles,1).reshape(-1, self.dim)
            rnn_output = self.rnn_cell(previous_latent_input,observation_input).squeeze(-1).view(-1, num_particles, self.dim)
            rnn_output = self.lin_t(rnn_output).reshape(-1, num_particles) if self.dim == 1 else self.lin_t(rnn_output)

            return state.set_batch_shape_mode(
                self.dist(rnn_output,self.scale_t),
                state.BatchShapeMode.FULLY_EXPANDED)

class Proposal_cnf(nn.Module):
    def __init__(self, transition, initial, scale_0, scale_t, device, type = 'planar', k=2, obs_dim = 2):
        super(Proposal_cnf, self).__init__()
        self.scale_0_vector = nn.Parameter(scale_0.squeeze().clone())#scale_0#
        self.scale_t_vector = nn.Parameter(scale_t.squeeze().clone())#scale_t#

        self.transition = transition
        self.initial = initial

        self.dim = self.scale_0_vector.shape[0] if len(self.scale_0_vector.shape) != 0 else 1
        self.obs_dim = obs_dim
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

        self.lin_0 = nn.Linear(self.obs_dim, self.dim, bias=False).to(device)
        self.lin_t = nn.Linear(obs_dim+self.dim, self.dim, bias=False).to(device)

        self.type_flow = type
        if self.type_flow == 'planar':
            self.k = k
            if self.dim == 1:
                self.planar = Planar_1d_compose(k=self.k, device=device)
            else:
                self.planar = Planar_compose(k=self.k, dim=self.dim, device=device)
        elif self.type_flow == 'radial':
            self.radial_flow = Radial(dim = 1)
        elif self.type_flow == 'nvp':
            self.k = k
            self.nvp = RealNVP_cond_compose(k=k, dim = self.dim, hidden_dim= 2 * self.dim, obser_dim = self.obs_dim, device=device)

    def sample(self, previous_latents=None, time=None, observations=None, batch_size = 10, num_particles = 100):
        self.scale_0 = torch.diag(self.scale_0_vector.exp() if self.dim != 1 else self.scale_0_vector.unsqueeze(-1).exp()).squeeze()
        self.scale_t = torch.diag(self.scale_t_vector.exp() if self.dim != 1 else self.scale_t_vector.unsqueeze(-1).exp()).squeeze()

        if self.type_flow != 'bootstrap':
            if time == 0:
                loc = self.lin_0(observations[0].unsqueeze(-1) if self.dim ==1 else observations[0]).squeeze(-1)

                dist_0 = state.set_batch_shape_mode(self.dist(loc,self.scale_0),
                                                          state.BatchShapeMode.BATCH_EXPANDED)
                samples = state.sample(dist_0, batch_size, num_particles)
                proposal_log_prob = state.log_prob(dist_0, samples)
                return samples, proposal_log_prob
            else:
                if time == 0.1:
                    time = 0
                loc = self.lin_t(torch.cat(
                        [previous_latents[-1].unsqueeze(-1) if self.dim ==1 else previous_latents[-1],
                         (observations[time].view(-1, 1, 1) if self.dim ==1 else observations[time].unsqueeze(1)).repeat(1, num_particles, 1)],
                        dim=2
                    ).view(-1, self.obs_dim+self.dim)).squeeze(-1).view((-1, num_particles) if self.dim ==1 else (-1, num_particles, self.dim))
                dist_t = state.set_batch_shape_mode(self.dist(loc,self.scale_t),
                                                          state.BatchShapeMode.FULLY_EXPANDED)
                proposal_samples = state.sample(dist_t, batch_size, num_particles)
                proposal_log_prob = state.log_prob(dist_t, proposal_samples)

                if self.type_flow == 'planar':
                    proposal_samples, log_det = self.planar(proposal_samples, observations,time, previous_latents=previous_latents)
                elif self.type_flow == 'nvp':
                    proposal_samples, log_det = self.nvp(proposal_samples,observations,time)
                elif self.type_flow == 'radial':
                    proposal_samples, log_det = self.radial_flow(proposal_samples)
                elif self.type_flow == 'normal':
                    log_det = 0.0
                else:
                    raise ValueError('Please select a type from {planar, radial, normal, bootstrap}.')
                proposal_log_prob = proposal_log_prob - log_det.squeeze(-1)
        elif self.type_flow == 'bootstrap':
            if time == 0:
                initial_samples = state.sample(self.initial(), batch_size, num_particles)
                initial_log_prob = state.log_prob(self.initial(), initial_samples)
                return initial_samples, initial_log_prob
            else:
                transition_dist = self.transition(previous_latents=previous_latents)
                proposal_samples = state.sample(transition_dist, batch_size, num_particles)
                proposal_log_prob = state.log_prob(
                    self.transition(previous_latents=previous_latents), proposal_samples)
        else:
            raise ValueError('Please select a type from {planar, radial, normal, bootstrap}.')
        return proposal_samples, proposal_log_prob

class Dynamic_cnf(nn.Module):
    def __init__(self, dyn_nf, prototype_transition, dim, type = 'planar', n_sequence=2, hidden_size = 5, init_var = 0.01):
        super(Dynamic_cnf, self).__init__()
        # self.scale_0_vector = nn.Parameter(scale_0.squeeze().clone())#scale_0#
        # self.scale_t_vector = nn.Parameter(scale_t.squeeze().clone())#scale_t#
        self.dyn_nf = dyn_nf
        self.n_sequence = n_sequence
        self.hidden_size = hidden_size
        self.init_var = init_var
        self.prototype_transition = prototype_transition

        self.dim = dim
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

        # self.lin_0 = nn.Linear(self.dim, self.dim, bias=False).to(device)
        # self.lin_t = nn.Linear(self.dim, self.dim, bias=False).to(device)

        # self.type_flow = type
        # if self.type_flow == 'planar':
        #     self.n_sequence = n_sequence
        #     if self.dim == 1:
        #         self.planar = Planar_1d_compose(k=self.n_sequence, device=device)
        #     else:
        #         self.planar = Planar_compose(k=self.n_sequence, dim=self.dim, device=device)
        # elif self.type_flow == 'radial':
        #     self.radial_flow = Radial(dim = 1)
        # elif self.type_flow == 'nvp':
        #     self.nvp = build_dyn_nf(self.n_sequence, self.hidden_size, self.dim, init_var=init_var)
    def forward(self, particles, previous_latents_bar):
        particles_pred_flatten = particles.reshape(-1, self.dim)
        particles_update_nf, _, log_det = self.dyn_nf.forward(particles_pred_flatten)

        log_det = log_det.reshape(particles.shape[:2])

        nf_dynamic_particles = particles_update_nf.reshape(particles.shape)

        transition_log_prob = log_prob(
            self.prototype_transition(previous_latents=previous_latents_bar), particles)

        dyn_log_prob = transition_log_prob + log_det

        return dyn_log_prob
class FCNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.Tanh(),, dropout_rate=0.5
            # nn.Dropout(dropout_rate),  # Adding dropout
            # nn.Linear(16, 16),
            nn.Tanh(),
            # nn.Dropout(dropout_rate),  # Adding another dropout layer
            nn.Linear(hidden_dim, out_dim),
        )


    def forward(self, x):
        return self.network(x.float())

functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor).to(x.device) + \
                            (x < 0).type(torch.FloatTensor).to(x.device) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor).to(x.device) + \
                     (x < 0).type(torch.FloatTensor).to(x.device) * torch.exp(x)
}

class Planar_1d_compose(nn.Module):
    def __init__(self, k=1, device='cuda'):
        super(Planar_1d_compose, self).__init__()
        self.flows = nn.Sequential(*[Planar_1d(device) for _ in range(k)])
    def forward(self, proposal_samples, observations, time, previous_latents = None):
        log_det = 0
        for flow in self.flows:
            proposal_samples, log_det_k = flow(proposal_samples, observations, time)
            log_det = log_det + log_det_k
        return  proposal_samples, log_det

class Planar_1d(nn.Module):
    def __init__(self, device='cuda'):
        super(Planar_1d, self).__init__()
        self.u = nn.Parameter(torch.tensor(0.1).to(device))
        self.b = nn.Parameter(torch.tensor(0.1).to(device))
        self.w = nn.Parameter(torch.tensor(0.1).to(device))
        self.reset_parameters()
    def reset_parameters(self):
        init.uniform_(self.w, 0.0, 0.25)
        init.uniform_(self.u, 0.0, 0.25)
        init.uniform_(self.b, 0.0, 0.25)
    def forward(self, proposal_samples, observations, time, previous_latents = None):
        num_particles = proposal_samples.shape[-1]
        proposal_samples = proposal_samples + self.u * torch.tanh(self.w * proposal_samples +
                                                                  self.b * observations[time].view(-1, 1).expand(-1,
                                                                                                                 num_particles))
        log_det = (1 + self.u * self.w * (1 - torch.tanh(self.w * proposal_samples +
                                                         self.b * observations[time].view(-1, 1).expand(-1,
                                                                                                        num_particles)) ** 2)).abs().log()
        return proposal_samples, log_det

class Planar_1d_marginal(nn.Module):
    def __init__(self, device='cuda'):
        super(Planar_1d_marginal, self).__init__()
        self.u = nn.Parameter(torch.tensor(0.1).to(device))
        self.b = nn.Parameter(torch.tensor(0.1).to(device))
        self.w = nn.Parameter(torch.tensor(0.1).to(device))
        self.reset_parameters()
    def reset_parameters(self):
        init.uniform_(self.w, 0.0, 0.01)
        init.uniform_(self.u, 0.0, 0.01)
        init.uniform_(self.b, 0.0, 0.01)
    def forward(self, proposal_samples, previous_latents = None):
        num_particles = proposal_samples.shape[-1]
        proposal_samples = proposal_samples + self.u * torch.tanh(self.w * proposal_samples +
                                                                  self.b)
        log_det = (1 + self.u * self.w * (1 - torch.tanh(self.w * proposal_samples +
                                                         self.b ) ** 2)).abs().log()
        return proposal_samples, log_det.squeeze(-1)

class Planar_compose(nn.Module):
    def __init__(self, k=1, dim = 2, device='cuda'):
        super(Planar_compose, self).__init__()
        self.flows = nn.Sequential(*[Planar(dim=dim, device=device).to(device) for _ in range(k)])
    def forward(self, proposal_samples, observations, time, previous_latents = None):
        log_det = 0
        for flow in self.flows:
            proposal_samples, log_det_k = flow(proposal_samples, observations, time, previous_latents = previous_latents)
            log_det = log_det + log_det_k
        return proposal_samples, log_det

class Planar(nn.Module):
    def __init__(self, dim, nonlinearity=torch.tanh, device='cuda'):
        super().__init__()
        self.h = nonlinearity
        self.w = nn.Parameter(torch.Tensor(dim))
        self.u = nn.Parameter(torch.Tensor(dim))
        self.b = nn.Parameter(torch.Tensor(dim))

        self.lin_b = nn.Linear(dim, dim)

        self.reset_parameters(dim)
        self.dim = dim


    def reset_parameters(self, dim):
        # init.constant_(self.w, 0.1)
        # init.constant_(self.u, 0.1)
        # init.constant_(self.b, 0.1)
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x, observation, time, previous_latents =None):

        observation = torch.cat([observation[time].unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
        # if self.h in (F.elu, F.leaky_relu):
        #     u = self.u
        # elif self.h == torch.tanh:
        #     scal = torch.log(1+torch.exp(self.w @ self.u)) - self.w @ self.u - 1
        #     u = self.u + scal * self.w / torch.norm(self.w) ** 2
        # else:
        #     raise NotImplementedError("Non-linearity is not supported.")
        u = self.u
        lin = torch.unsqueeze( x @ self.w, -1) + (self.b * self.lin_b(observation)).sum(-1, keepdims=True)
        z = x + u * self.h(lin)
        phi = functional_derivatives[self.h](lin) * self.w
        log_det = torch.log(torch.abs(1 + phi @ u) + 1e-4)
        return z.squeeze(-1), log_det


class Radial(nn.Module):
    """
    Radial flow.
        z = f(x) = = x + β h(α, r)(z − z0)
    [Rezende and Mohamed 2015]
    """
    def __init__(self, dim):
        super().__init__()
        self.x0 = nn.Parameter(torch.Tensor(dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def reset_parameters(self, dim):
        init.uniform_(self.x0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_alpha, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.beta, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Given x, returns z and the log-determinant log|df/dx|.
        """
        m, n = x.shape
        r = torch.norm(x - self.x0)
        h = 1 / (torch.exp(self.log_alpha) + r)
        beta = -torch.exp(self.log_alpha) + torch.log(1 + torch.exp(self.beta))
        z = x + beta * h * (x - self.x0)
        log_det = (n - 1) * torch.log(1 + beta * h) + \
                  torch.log(1 + beta * h - \
                            beta * r / (torch.exp(self.log_alpha) + r) ** 2)
        return z, log_det

class RealNVP_cond_compose(nn.Module):
    def __init__(self, k, dim, hidden_dim, obser_dim, device):
        super(RealNVP_cond_compose, self).__init__()
        self.k = k
        if k == 0:
            self.flows = nn.Identity()
        else:
            self.flows = nn.Sequential(*[RealNVP_cond_t(dim = dim,
                                                        hidden_dim = hidden_dim,
                                                        obser_dim = obser_dim).to(device) for _ in range(k)])
            for flow in self.flows:
                flow.zero_initialization(0.1)
    def forward(self, proposal_samples, observations, time):
        log_det = torch.zeros_like(proposal_samples[...,0])
        if self.k == 0:
            pass
        else:
            for flow in self.flows:
                proposal_samples, log_det_k = flow(proposal_samples, observations, time)
                log_det = log_det + log_det_k
        return proposal_samples, log_det

class RealNVP_cond_t(nn.Module):

    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, obser_dim=None):
        super().__init__()
        self.dim = dim
        self.dim_1 = self.dim - dim//2
        self.dim_2 = self.dim//2
        self.obser_dim=obser_dim
        self.t1 = base_network(self.dim_1+self.obser_dim, self.dim_2, hidden_dim)
        self.s1 = base_network(self.dim_1+self.obser_dim, self.dim_2, hidden_dim)
        self.t2 = base_network(self.dim_2+self.obser_dim, self.dim_1, hidden_dim)
        self.s2 = base_network(self.dim_2+self.obser_dim, self.dim_1, hidden_dim)
    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, obser, time):
        x_mean, x_std = x.mean(dim=1,keepdim=True), x.std(dim=1,keepdim=True)
        # x = (x-x_mean)/x_std
        obser = obser[time].unsqueeze(1).repeat(1, x.shape[1], 1)
        lower, upper = x[..., :self.dim_1], x[..., self.dim_1:]
        t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
        s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
        upper = t1_transformed + upper #* torch.exp(s1_transformed)
        t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
        s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
        lower = t2_transformed + lower #* torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=-1)
        # z = z*x_std+x_mean
        # log_det = torch.sum(s1_transformed, dim=-1) + \
        #           torch.sum(s2_transformed, dim=-1)
        log_det = torch.zeros_like(torch.sum(s2_transformed, dim=-1))
        return z, log_det


class NormalizingFlowModel(nn.Module):

    def __init__(self, prior, flows, device='cuda'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = nn.ModuleList(flows).to(self.device)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x.float())
        # z, prior_logprob = x, None
        return z, prior_logprob, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,)).to(self.device)
        x, _ = self.inverse(z)
        return x

def build_dyn_nf(n_sequence, hidden_size, state_dim, init_var=0.01, translate = False, type = 'nvp'):
    if type == 'nvp':
        flows_dyn = [RealNVP(dim=state_dim, hidden_dim= state_dim, translate = translate) for _ in range(n_sequence)]
        for f in flows_dyn:
            f.zero_initialization(var=init_var)
    else:
        flows_dyn = [Planar_1d_marginal() for _ in range(n_sequence)]

    prior_dyn = MultivariateNormal(torch.zeros(state_dim).to(device), torch.eye(state_dim).to(device))

    nf_dyn = NormalizingFlowModel(prior_dyn, flows_dyn, device=device)

    return nf_dyn


class RealNVP_cond(nn.Module):

    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, obser_dim=None):
        super().__init__()
        self.dim = dim
        self.dim_1 = self.dim - dim//2
        self.dim_2 = self.dim//2
        self.obser_dim=obser_dim
        self.t1 = base_network(self.dim_1+self.obser_dim, self.dim_2, hidden_dim)
        self.s1 = base_network(self.dim_1+self.obser_dim, self.dim_2, hidden_dim)
        self.t2 = base_network(self.dim_2+self.obser_dim, self.dim_1, hidden_dim)
        self.s2 = base_network(self.dim_2+self.obser_dim, self.dim_1, hidden_dim)
    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                pass
                # nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                # layer.bias.data.fill_(0.)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                pass
                # nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                # layer.bias.data.fill_(0.)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                pass
                # nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                # layer.bias.data.fill_(0.)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                # pass
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0.)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, obser):
        lower, upper = x[:, :self.dim_1], x[:, self.dim_1:]
        t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
        s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
        s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        # log_det = torch.zeros_like(torch.sum(s2_transformed, dim=-1))
        return z, log_det

    def inverse(self, z, obser):
        lower, upper = z[:,:self.dim_1], z[:,self.dim_1:]
        t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
        s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
        s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        # log_det = torch.zeros_like(torch.sum(s2_transformed, dim=-1))
        return x, log_det


class NormalizingFlowModel_cond(nn.Module):

    def __init__(self, prior, flows, device='cuda'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = nn.ModuleList(flows).to(self.device)

    def forward(self, x,obser):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows:
            x_, ld = flow.forward(x,obser)
            log_det += ld
        z, prior_logprob = x_, self.prior.log_prob(x_.float())
        return z, prior_logprob, log_det

    def inverse(self, z, obser):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z,obser)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples,obser):
        z = self.prior.sample((n_samples,)).to(self.device)
        x, _ = self.inverse(z,obser)
        return x


def build_conditional_nf(n_sequence, hidden_size, state_dim, init_var=0.01, prior_mean=0.0, prior_std=1.0,flow=RealNVP_cond):
    flows = [flow(dim=state_dim, obser_dim=state_dim) for _ in range(n_sequence)]

    for f in flows:
        f.zero_initialization(var=init_var)

    prior_init = MultivariateNormal(torch.zeros(state_dim).to(device) + prior_mean,
                                    torch.eye(state_dim).to(device) * prior_std**2)

    cond_model = NormalizingFlowModel_cond(prior_init, flows, device=device)

    return cond_model

class RealNVP(nn.Module):
    """
    Non-volume preserving flow.

    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, translate = False):
        super().__init__()
        self.dim = dim
        self.dim_1 = self.dim - dim//2
        self.dim_2 = self.dim//2
        self.t1 = base_network(self.dim_1, self.dim_2, hidden_dim)
        self.s1 = base_network(self.dim_1, self.dim_2, hidden_dim)
        self.t2 = base_network(self.dim_2, self.dim_1, hidden_dim)
        self.s2 = base_network(self.dim_2, self.dim_1, hidden_dim)
        self.translate = translate

    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                # layer.bias.data.fill_(0)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                # layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                # layer.bias.data.fill_(0)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                # layer.bias.data.fill_(0)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        lower, upper = x[:,:self.dim_1], x[:,self.dim_1:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper if self.translate else t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower if self.translate else t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        # log_det = torch.zeros_like(torch.sum(s2_transformed, dim=-1)) if self.translate else torch.sum(s1_transformed, dim=1) +  torch.sum(s2_transformed, dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:,:self.dim_1], z[:,self.dim_1:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed) if self.translate else lower - t2_transformed
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed) if self.translate else upper - t1_transformed
        x = torch.cat([lower, upper], dim=1)
        # log_det = torch.zeros_like(torch.sum(s2_transformed, dim=-1)) if self.translate else torch.sum(-s1_transformed, dim=1) +  torch.sum(-s2_transformed, dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det




def lgssm_true_posterior(observations, initial_loc, initial_scale,
                         transition_mult, transition_bias, transition_scale,
                         emission_mult, emission_bias, emission_scale):
    ssm_parameter_lists = [initial_loc, initial_scale,
    transition_mult, transition_scale,
    emission_mult, emission_scale,
    transition_bias, emission_bias]

    initial_loc, initial_scale, \
    transition_mult, transition_scale, \
    emission_mult, emission_scale,\
    transition_bias, emission_bias = [ssm_parameter.cpu().detach().numpy() for
                                                    ssm_parameter in ssm_parameter_lists if torch.is_tensor(ssm_parameter)]
    dim = initial_loc.shape[0] if len(initial_loc.shape) != 0 else 1

    if dim == 1:
        kf = pykalman.KalmanFilter(
            initial_state_mean=[initial_loc],
            initial_state_covariance=[[initial_scale**2]],
            transition_matrices=[[transition_mult]],
            transition_offsets=[transition_bias],
            transition_covariance=[[transition_scale**2]],
            observation_matrices=[[emission_mult]],
            observation_offsets=[emission_bias],
            observation_covariance=[[emission_scale**2]])
    else:
        kf = pykalman.KalmanFilter(
            initial_state_mean=initial_loc,
            initial_state_covariance=initial_scale ** 2,
            transition_matrices=transition_mult,
            transition_offsets=transition_bias,
            transition_covariance=transition_scale ** 2,
            observation_matrices=emission_mult,
            observation_offsets=emission_bias,
            observation_covariance=emission_scale ** 2)

    return kf.smooth(torch.stack([observation[0].cpu().squeeze() for observation in observations], dim = 0).numpy())


class TrainingStats(object):
    def __init__(self, true_transition_mult_online1, true_emission_mult_online1, true_transition_mult_online2, true_emission_mult_online2, initial_loc, initial_scale, true_transition_mult,
                 transition_scale, true_emission_mult, emission_scale,
                 num_timesteps, num_test_obs, test_inference_num_particles,
                 saving_interval=100, logging_interval=100, algorithm='is',args=None, num_iterations=500, dataloader = None):
        device = args.device
        self.dim = 1 if len(initial_loc.shape)==0 else initial_loc.shape[0]
        #offline parameter
        self.true_transition_mult = true_transition_mult
        self.true_transition_mult = self.true_transition_mult if self.dim >1  else self.true_transition_mult.squeeze()
        self.true_emission_mult = true_emission_mult
        self.true_emission_mult = torch.diag(self.true_emission_mult if self.dim != 1 else
                                               self.true_emission_mult.unsqueeze(-1)).squeeze()
        #online1 parameter
        self.true_transition_mult_online1 = true_transition_mult_online1
        self.true_transition_mult_online1 = self.true_transition_mult_online1 if self.dim > 1 else self.true_transition_mult_online1.squeeze()
        self.true_emission_mult_online1 = true_emission_mult_online1
        self.true_emission_mult_online1 = torch.diag(self.true_emission_mult_online1 if self.dim != 1 else
                                             self.true_emission_mult_online1.unsqueeze(-1)).squeeze()

        # online2 parameter
        self.true_transition_mult_online2 = true_transition_mult_online2
        self.true_transition_mult_online2 = self.true_transition_mult_online2 if self.dim > 1 else self.true_transition_mult_online2.squeeze()
        self.true_emission_mult_online2 = true_emission_mult_online2
        self.true_emission_mult_online2 = torch.diag(self.true_emission_mult_online2 if self.dim != 1 else
                                                     self.true_emission_mult_online2.unsqueeze(-1)).squeeze()

        self.test_inference_num_particles = test_inference_num_particles
        self.saving_interval = saving_interval
        self.logging_interval = logging_interval
        self.p_l2_history = []
        self.q_l2_history = []
        self.normalized_log_weights_history = []
        self.iteration_idx_history = []
        self.loss_history = []
        self.initial = Initial(initial_loc, initial_scale).to(device)

        #offline validation data
        self.true_transition = Transition(true_transition_mult,
                                          transition_scale).to(device)
        self.true_emission = Emission(true_emission_mult, emission_scale).to(device)

        if dataloader is None:
            dataloader = train.get_synthetic_dataloader(self.initial,
                                                              self.true_transition,
                                                              self.true_emission,
                                                              num_timesteps,
                                                              num_test_obs)
        else:
            pass

        self.test_obs = next(iter(dataloader))

        #online1 validation data
        #test stage
        self.true_transition_online1 = Transition(true_transition_mult_online1,
                                          transition_scale).to(device)
        self.true_emission_online1 = Emission(true_emission_mult_online1, emission_scale).to(device)
        dataloader_online1 = train.get_synthetic_dataloader(self.initial,
                                                          self.true_transition_online1,
                                                          self.true_emission_online1,
                                                          num_timesteps,
                                                          num_test_obs)
        self.test_obs_online1 = next(iter(dataloader_online1))

        # online2 validation data
        # test stage
        self.true_transition_online2 = Transition(true_transition_mult_online2,
                                                  transition_scale).to(device)
        self.true_emission_online2 = Emission(true_emission_mult_online2, emission_scale).to(device)
        dataloader_online2 = train.get_synthetic_dataloader(self.initial,
                                                                  self.true_transition_online2,
                                                                  self.true_emission_online2,
                                                                  num_timesteps,
                                                                  num_test_obs)
        self.test_obs_online2 = next(iter(dataloader_online2))

        if algorithm == 'iwae':
            self.algorithm = 'is'
        else:
            self.algorithm = 'smc'

        self.args = args
        self.device = args.device
        self.num_iterations = num_iterations
    def __call__(self, initial_state, epoch_idx, epoch_iteration_idx, loss, initial,
                 transition, emission, proposal, test=False, stage=0, args = None):
        # if test == True:
        #     latents_test=self.test_obs[0]
        #     latents_test = [latent_test.to(self.device).unsqueeze(-1) if len(latent_test.shape) == 1 else latent_test.to(self.device)
        #                for latent_test in latents_test]
        #     inference_result, _ = aesmc.inference.infer(
        #         [None,None],self.algorithm, self.test_obs[1], self.initial,
        #         transition, emission, proposal,
        #         self.test_inference_num_particles, args=self.args, true_latents=latents_test,
        #         return_log_marginal_likelihood=True, measurement=args.measurement)
        #     loss_rmse_test=inference_result['loss_report'].cpu().detach().numpy()
        #     print('loss_rmse_test:', loss_rmse_test)
        #     return loss_rmse_test


        if epoch_iteration_idx % self.saving_interval == 0 or epoch_iteration_idx + 1 == self.num_iterations:
            if args.measurement == 'CRNVP':
                if len(self.true_emission_mult.shape) == 0:
                    emission.mult = torch.zeros_like(self.true_emission_mult).to(device)
                else:
                    emission.mult = torch.zeros_like(self.true_emission_mult.diag()).to(device)
            if args.NF_dyn:
                if len(self.true_emission_mult.shape) == 0:
                    transition.mult = torch.zeros_like(self.true_transition_mult).to(device)
                else:
                    transition.mult = torch.zeros_like(self.true_transition_mult).to(device)
            print('theta1:', transition.mult.flatten().cpu().detach().numpy()[0], 'theta2:', emission.mult.cpu().detach().numpy())
            if stage == 0:
                self.p_l2_history.append(np.linalg.norm(
                    np.concatenate([transition.mult.flatten().cpu().detach().numpy(), emission.mult.cpu().detach().numpy()]) -
                    np.concatenate([self.true_transition_mult.flatten().cpu().detach().numpy(), self.true_emission_mult.diag().cpu().detach().numpy()])
                    if self.dim !=1 else np.array([transition.mult.flatten().cpu().detach().numpy(), emission.mult.cpu().detach().numpy()],dtype=object)-
                                                  np.array([self.true_transition_mult.cpu().detach().numpy(), self.true_emission_mult.cpu().detach().numpy()],dtype=object)
                ).squeeze())
                latents = self.test_obs[0]
                latents = [latent.to(self.device).unsqueeze(-1) if len(latent.shape) == 1 else latent.to(self.device)
                           for latent in latents]
                inference_result, _, _ = inference.infer(
                    [None,None,initial_state], self.algorithm, self.test_obs[1], self.initial,
                    transition, emission, proposal,
                    self.test_inference_num_particles, args=self.args, true_latents=latents,
                    return_log_marginal_likelihood=True, measurement=args.measurement)
                normalized_weights = aemath.normalize_log_probs(
                    torch.stack(inference_result['log_weights'], dim=0)) + 1e-8
                self.normalized_log_weights_history.append(normalized_weights.cpu().detach().numpy())
                # self.loss_history.append(inference_result['log_marginal_likelihood'].cpu().detach().numpy())
                self.loss_history.append(inference_result['loss_report'].cpu().detach().numpy())
                self.iteration_idx_history.append(epoch_iteration_idx)
            # if stage ==1:
            #     self.p_l2_history.append(np.linalg.norm(
            #         np.concatenate(
            #             [transition.mult.flatten().cpu().detach().numpy(), emission.mult.cpu().detach().numpy()]) -
            #         np.concatenate([self.true_transition_mult_online1.flatten().cpu().detach().numpy(),
            #                         self.true_emission_mult_online1.diag().cpu().detach().numpy()])
            #         if self.dim != 1 else np.array(
            #             [transition.mult.flatten().cpu().detach().numpy(), emission.mult.cpu().detach().numpy()],
            #             dtype=object) -
            #                               np.array([self.true_transition_mult_online1.cpu().detach().numpy(),
            #                                         self.true_emission_mult_online1.cpu().detach().numpy()],
            #                                        dtype=object)
            #     ).squeeze())
            #     latents = self.test_obs_online1[0]
            #     latents = [latent.to(self.device).unsqueeze(-1) if len(latent.shape) == 1 else latent.to(self.device)
            #                for latent in latents]
            #     inference_result, _ = aesmc.inference.infer(
            #         [None,None],self.algorithm, self.test_obs_online1[1], self.initial,
            #         transition, emission, proposal,
            #         self.test_inference_num_particles, args=self.args, true_latents=latents,
            #         return_log_marginal_likelihood=True, measurement=args.measurement)
            #     normalized_weights = aesmc.math.normalize_log_probs(
            #         torch.stack(inference_result['log_weights'], dim=0)) + 1e-8
            #     self.normalized_log_weights_history.append(normalized_weights.cpu().detach().numpy())
            #     # self.loss_history.append(inference_result['log_marginal_likelihood'].cpu().detach().numpy())
            #     self.loss_history.append(inference_result['loss_report'].cpu().detach().numpy())
            #     self.iteration_idx_history.append(epoch_iteration_idx)
            # if stage ==2:
            #     self.p_l2_history.append(np.linalg.norm(
            #         np.concatenate(
            #             [transition.mult.flatten().cpu().detach().numpy(), emission.mult.cpu().detach().numpy()]) -
            #         np.concatenate([self.true_transition_mult_online2.flatten().cpu().detach().numpy(),
            #                         self.true_emission_mult_online2.diag().cpu().detach().numpy()])
            #         if self.dim != 1 else np.array(
            #             [transition.mult.flatten().cpu().detach().numpy(), emission.mult.cpu().detach().numpy()],
            #             dtype=object) -
            #                               np.array([self.true_transition_mult_online2.cpu().detach().numpy(),
            #                                         self.true_emission_mult_online2.cpu().detach().numpy()],
            #                                        dtype=object)
            #     ).squeeze())
            #     latents = self.test_obs_online2[0]
            #     latents = [latent.to(self.device).unsqueeze(-1) if len(latent.shape) == 1 else latent.to(self.device)
            #                for latent in latents]
            #     inference_result = aesmc.inference.infer(
            #         self.algorithm, self.test_obs_online2[1], self.initial,
            #         transition, emission, proposal,
            #         self.test_inference_num_particles, args=self.args, true_latents=latents,
            #         return_log_marginal_likelihood=True)
            #     normalized_weights = aesmc.math.normalize_log_probs(
            #         torch.stack(inference_result['log_weights'], dim=0)) + 1e-8
            #     self.normalized_log_weights_history.append(normalized_weights.cpu().detach().numpy())
            #     # self.loss_history.append(inference_result['log_marginal_likelihood'].cpu().detach().numpy())
            #     self.loss_history.append(inference_result['loss_report'].cpu().detach().numpy())
            #     self.iteration_idx_history.append(epoch_iteration_idx)

        if epoch_iteration_idx % self.logging_interval == 0 or epoch_iteration_idx + 1 == self.num_iterations:
            print('Iteration {}:'
                  ' Loss = {:.3f},'
                  ' parameter error = {:.6f},'
                  .format(epoch_iteration_idx,inference_result['loss_report'],
                          self.p_l2_history[-1]))
