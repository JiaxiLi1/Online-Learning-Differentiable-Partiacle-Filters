import torch
import torch.nn as nn
from utils import wrap_angle, et_distance, transform_particles_as_input
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from nf.flows import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class dynamic_model(nn.Module):
    def __init__(self, mo_noise_generator, stds_particle):
        super().__init__()
        self.mo_noise_generator=mo_noise_generator
        self.std_x, self.std_y, self.std_t = stds_particle
        self.action_std_xy=10.0
        self.action_std_t=0.2
    def forward(self, actions, particles, means, stds, state_step_sizes):
        batch_size, num_particle = particles.shape[0], particles.shape[1]
        actions1 = actions[:, None, :]
        #actions1[:,:,:2] += torch.normal(size=(actions1.shape[0],actions1.shape[1],2), std=self.action_std_xy, mean=0.0).to(device)
        #actions1[:, :, 2:] += torch.normal(size=(actions1.shape[0], actions1.shape[1], 1), std=self.action_std_t, mean=0.0).to(device)
        action_input = (actions1 / torch.tensor(stds['a']).to(device)).repeat([1, particles.shape[1], 1])  # (32,100,3)
        random_input = torch.randn(size=action_input.shape).to(device)
        input = torch.cat([action_input, random_input], dim=-1)

        delta = self.mo_noise_generator(input.float())

        delta -= torch.mean(delta, dim=1, keepdim=True)
        noisy_actions = actions1 + delta

        theta = particles[:, :, 2:3]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        noise_x = torch.normal(mean=0., std=self.std_x, size=(batch_size, num_particle, 1)).to(device)
        noise_y = torch.normal(mean=0., std=self.std_y, size=(batch_size, num_particle, 1)).to(device)
        noise_t = torch.normal(mean=0., std=self.std_t, size=(batch_size, num_particle, 1)).to(device)
        noise_state = torch.cat([noise_x, noise_y, noise_t], dim=-1)

        # new_x, new_y, new_theta = particles[:,:,0:1] + noisy_actions[:,:,0:1] + noise_x, \
        #                           particles[:,:,1:2] + noisy_actions[:,:,1:2] + noise_y, \
        #                           particles[:,:,2:3] + noisy_actions[:,:,2:3] + noise_t

        new_x = particles[:, :, 0:1] + (
                noisy_actions[:, :, 0:1] * cos_theta + noisy_actions[:, :, 1:2] * sin_theta) + noise_x
        new_y = particles[:, :, 1:2] + (
                noisy_actions[:, :, 0:1] * sin_theta - noisy_actions[:, :, 1:2] * cos_theta) + noise_y
        new_theta = particles[:, :, 2:3] + noisy_actions[:, :, 2:3] + noise_t


        moved_particles = torch.cat([new_x, new_y, new_theta], dim=-1)

        return moved_particles, noise_state

class measurement_model_cosine_distance(nn.Module):
    def __init__(self, particle_encoder):
        super().__init__()
        self.particle_encoder=particle_encoder

    def forward(self, encodings, particles, means, stds, encodings_maps=None):

        particle_input = transform_particles_as_input(particles, means, stds)
        particle_encoder = self.particle_encoder.float()
        encodings_state = particle_encoder(particle_input.float()) # shape: (batch, num_particle, hidden_size)

        # encodings shape: (batch, hidden_size)
        encodings_obs = encodings[:,None,:].repeat(1, particles.shape[1], 1)

        obs_likelihood = 1/(1e-8 + et_distance(encodings_obs, encodings_state))

        return obs_likelihood.log()

class measurement_model_cnf(nn.Module):
    def __init__(self, particle_encoder, CNF):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.CNF = CNF

    def forward(self, encodings, particles, means, stds, encodings_maps):
        hidden_dim = encodings.shape[-1]
        n_batch, n_particles = particles.shape[:2]

        particle_input = transform_particles_as_input(particles, means, stds)
        particle_encoder = self.particle_encoder.float()
        encodings_state = particle_encoder(particle_input.float())  # shape: (batch, particle_num, hidden_size)
        encodings_state = encodings_state.reshape([-1, hidden_dim])

        encodings_obs = encodings[:, None, :].repeat(1, particles.shape[1], 1)
        encodings_obs = encodings_obs.reshape([-1, hidden_dim])

        encodings_maps = encodings_maps.repeat(n_batch * n_particles, 1)

        encodings = torch.cat([encodings_state, encodings_maps], dim=-1)

        z, log_prob_z, log_det = self.CNF.forward(encodings_obs, encodings)
        obs_likelihood = (log_prob_z + log_det).reshape([n_batch, n_particles])

        return obs_likelihood

def nf_dynamic_model(dynamical_nf, dynamic_particles, jac_shape, NF=False, forward=False, mean=None, std=None):

    if NF:
        n_batch, n_particles, dimension = dynamic_particles.shape
        if not forward:
            dyn_particles_mean, dyn_particles_std = dynamic_particles.mean(dim=1, keepdim=True).detach().clone().repeat([1, n_particles, 1]), \
                                                    dynamic_particles.std(dim=1, keepdim=True).detach().clone().repeat([1, n_particles, 1])
        else:
            dyn_particles_mean, dyn_particles_std = mean.detach().clone().repeat([1, n_particles, 1]),\
                                                    std.detach().clone().repeat([1, n_particles, 1])
        dyn_particles_mean_flatten, dyn_particles_std_flatten = dyn_particles_mean.reshape(-1, dimension), dyn_particles_std.reshape(-1,dimension)
        context = torch.cat([dyn_particles_mean_flatten, dyn_particles_std_flatten], dim=-1)
        dynamic_particles = (dynamic_particles - dyn_particles_mean) / dyn_particles_std

        particles_pred_flatten = dynamic_particles.reshape(-1, dimension)

        if forward:
            particles_update_nf, _, log_det = dynamical_nf.forward(particles_pred_flatten, context)
        else:
            particles_update_nf, log_det = dynamical_nf.inverse(particles_pred_flatten, context)
        jac_dynamic = -log_det
        jac_dynamic = jac_dynamic.reshape(dynamic_particles.shape[:2])

        nf_dynamic_particles = particles_update_nf.reshape(dynamic_particles.shape)
        nf_dynamic_particles = nf_dynamic_particles * dyn_particles_std + dyn_particles_mean
    else:
        nf_dynamic_particles=dynamic_particles
        jac_dynamic = torch.zeros(jac_shape).to(device)
    return nf_dynamic_particles, jac_dynamic

def normalising_flow_propose(cond_model, particles_pred, obs, flow=RealNVP_cond, n_sequence=2, hidden_dimension=8, obser_dim=None):

    B, N, dimension = particles_pred.shape

    pred_particles_mean, pred_particles_std = particles_pred.mean(dim=1, keepdim=True).detach().clone().repeat([1, N, 1]), \
                                            particles_pred.std(dim=1, keepdim=True).detach().clone().repeat([1, N, 1])
    dyn_particles_mean_flatten, dyn_particles_std_flatten = pred_particles_mean.reshape(-1, dimension), pred_particles_std.reshape(-1, dimension)
    context = torch.cat([dyn_particles_mean_flatten, dyn_particles_std_flatten], dim=-1)
    particles_pred = (particles_pred - pred_particles_mean) / pred_particles_std

    particles_pred_flatten=particles_pred.reshape(-1,dimension)
    obs_reshape = obs[:, None, :].repeat([1,N,1]).reshape(B*N,-1)
    obs_reshape = torch.cat([obs_reshape, context], dim=-1)

    particles_update_nf, log_det=cond_model.inverse(particles_pred_flatten, obs_reshape)

    jac=-log_det
    jac=jac.reshape(particles_pred.shape[:2])

    particles_update_nf = particles_update_nf.reshape(particles_pred.shape)
    particles_update_nf = particles_update_nf * pred_particles_std + pred_particles_mean

    return particles_update_nf, jac

def proposal_likelihood(cond_model, dynamical_nf, measurement_model, particles_dynamical, particles_physical,
                        encodings, noise, jac_dynamic, NF, NF_cond, prototype_density, means, stds, encodings_maps):
    encodings_clone = encodings.detach().clone()
    encodings_clone.requires_grad = False

    if NF_cond:
        propose_particle, jac_prop = normalising_flow_propose(cond_model, particles_dynamical, encodings_clone)
        if NF:
            particle_prop_dyn_inv, jac_prop_dyn_inv = nf_dynamic_model(dynamical_nf, propose_particle,jac_dynamic.shape, NF=NF, forward=True,
                                                                       mean=particles_physical.mean(dim=1, keepdim=True),
                                                                       std=particles_physical.std(dim=1, keepdim=True))
            prior_log = prototype_density(particle_prop_dyn_inv - (particles_physical - noise)) - jac_prop_dyn_inv  #####
        else:
            prior_log = prototype_density(propose_particle - (particles_physical - noise))
        propose_log = prototype_density(noise) + jac_dynamic + jac_prop
    else:
        propose_particle = particles_dynamical
        prior_log = prototype_density(noise) + jac_dynamic
        propose_log = prototype_density(noise) + jac_dynamic

    lki_log = measurement_model(encodings, propose_particle, means, stds, encodings_maps)
    return propose_particle, lki_log, prior_log, propose_log