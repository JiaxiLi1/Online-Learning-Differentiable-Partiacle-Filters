import torch
from torch import nn
import torch.nn.functional as F
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from utils import et_distance, transform_particles_as_input, transform_particles_as_input_house3d
from nf.flows import *
from nf.cglow.CGlowModel import CondGlowModel
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def     build_conditional_nf(n_sequence, hidden_size, state_dim, init_var=0.01, prior_mean=0.0, prior_std=1.0,flow=RealNVP_cond):
    flows = [flow(dim=state_dim, obser_dim=hidden_size) for _ in range(n_sequence)]

    for f in flows:
        f.zero_initialization(var=init_var)

    prior_init = MultivariateNormal(torch.zeros(state_dim).to(device) + prior_mean,
                                    torch.eye(state_dim).to(device) * prior_std**2)

    cond_model = NormalizingFlowModel_cond(prior_init, flows, device=device)

    return cond_model

def build_conditional_glow(args):
    conditional_glow = CondGlowModel(args)

    return conditional_glow

def build_dyn_nf(n_sequence, hidden_size, state_dim, init_var=0.01):
    flows_dyn = [RealNVP(dim=state_dim) for _ in range(n_sequence)]

    for f in flows_dyn:
        f.zero_initialization(var=init_var)

    prior_dyn = MultivariateNormal(torch.zeros(state_dim).to(device), torch.eye(state_dim).to(device))

    nf_dyn = NormalizingFlowModel(prior_dyn, flows_dyn, device=device)

    return nf_dyn


def obs_feature_disk(encodings, num_particles):
    encodings_obs = encodings[:, None, :].repeat(1, num_particles, 1)
    return encodings_obs


def state_feature_disk(particle_encoder, update_particles):
    particle_encoder = particle_encoder.float()
    encodings_state = particle_encoder(update_particles.float())  # shape: (batch, particle_num, hidden_size)
    return encodings_state

def obs_feature_maze(encodings, num_particles):
    encodings_obs = encodings[:, None, :].repeat(1, num_particles, 1)
    return encodings_obs

def state_feature_maze(particle_encoder, update_particles, environment_data):
    # means, stds, encodings_maps = environment_data
    # particle_input = transform_particles_as_input(update_particles, means, stds)
    particle_input = update_particles
    particle_encoder = particle_encoder.float()
    encodings_state = particle_encoder(particle_input.float())  # shape: (batch, num_particle, hidden_size)
    return encodings_state

def obs_feature_house3d(encodings, num_particles):
    encodings_obs = encodings[:, None, :].repeat(1, num_particles, 1, 1, 1)
    return encodings_obs

def state_feature_house3d(particle_encoder, update_particles, environment_data=None):
    particle_input = transform_particles_as_input_house3d(update_particles)
    particle_encoder = particle_encoder.float()
    encodings_state = particle_encoder(particle_input.float())  # shape: (batch, num_particle, hidden_size)
    return encodings_state

def state_feature_house3d_spatial(particle_encoder, update_particles, maps=None, map_size_original=None):

    # start_time = time.time()
    map_width_original, map_height_original = map_size_original
    map_width = maps.shape[-2]
    map_height = maps.shape[-1]
    map_n_channels = maps.shape[-3]

    update_particles_resized = torch.zeros_like(update_particles)
    update_particles_resized[..., 0] = update_particles[..., 0].clone() / map_width_original * map_width
    update_particles_resized[..., 1] = update_particles[..., 1].clone() / map_height_original * map_height

    num_batches, num_particles = update_particles.shape[0], update_particles.shape[1]
    num_particles_batches = num_batches * num_particles
    theta = torch.zeros([num_particles_batches, 2, 3]).to(device)
    update_particles_flatten = update_particles_resized.reshape(-1, 3)

    cos_state = torch.cos(update_particles_flatten[:, 2] + np.pi / 2).clone()
    sin_state = torch.sin(update_particles_flatten[:, 2] + np.pi / 2).clone()
    theta[:, 0] = torch.tensor(torch.stack(
        [cos_state / 6, -sin_state/ 6, (update_particles_flatten[:, 0]) / (map_width / 2) - 1], dim=1))
    theta[:, 1] = torch.tensor(torch.stack(
        [sin_state / 6, cos_state / 6, (update_particles_flatten[:, 1]) / (map_height / 2) - 1], dim=1))
    grid = F.affine_grid(theta, (num_particles_batches, map_n_channels, map_width // 14, map_height // 14), align_corners=True)
    transformed_input_tensor = F.grid_sample(torch.cat([maps] * num_particles, dim=1).reshape([num_particles_batches, 1, map_width, map_height]),
                                             grid,align_corners=True)
    # end_time = time.time()
    #
    # # Compute the elapsed time
    # elapsed_time = end_time - start_time
    #
    # print(f"spatial transform took {elapsed_time} seconds to run.")
    #
    # start_time = time.time()
    encodings_state = particle_encoder(transformed_input_tensor)

    # end_time = time.time()
    #
    # # Compute the elapsed time
    # elapsed_time = end_time - start_time
    #
    # print(f"particle encoder took {elapsed_time} seconds to run.")
    return encodings_state

def features_state_obs(encodings, num_particles,particle_encoder, update_particles, environment_data, params):
    if params.dataset == 'maze':
        encodings_state = state_feature_maze(particle_encoder, update_particles,
                                             environment_data)  # shape: (batch, particle_num, hidden_size)
        encodings_obs = obs_feature_maze(encodings, num_particles)  # shape: (batch_size, particle_num, hidden_size)
    elif params.dataset == 'disk':
        encodings_state = state_feature_disk(particle_encoder,
                                             update_particles)  # shape: (batch, particle_num, hidden_size)
        encodings_obs = obs_feature_disk(encodings, num_particles)  # shape: (batch_size, particle_num, hidden_size)
    elif params.dataset == 'house3d':
        if params.spatial:
            encodings_state = state_feature_house3d_spatial(particle_encoder,
                                                            update_particles,
                                                            maps=environment_data[0],
                                                            map_size_original=environment_data[1])  # shape: (batch, particle_num, hidden_size)
        else:
            encodings_state = state_feature_house3d(particle_encoder,
                                                 update_particles, environment_data=environment_data)
        encodings_obs = obs_feature_house3d(encodings, num_particles)  # shape: (batch_size, particle_num, hidden_size)
    else:
        raise ValueError('Please select a dataset from {disk, maze}')
    return encodings_state, encodings_obs

class measurement_model_cosine_distance(nn.Module):
    def __init__(self, particle_encoder, params):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.params = params

    def forward(self, encodings, update_particles, environment_data=None, pretrain=False):
        n_batch, n_particles = update_particles.shape[:2]
        encodings_state, encodings_obs = features_state_obs(encodings, n_particles, self.particle_encoder,
                                                            update_particles, environment_data, self.params)



        if pretrain:
            likelihood =  (2 - et_distance(encodings_obs, encodings_state))/2
            return likelihood + 1e-12
        else:
            likelihood = 1 / (1e-7 + et_distance(encodings_obs, encodings_state))
            return (likelihood+1e-12).log()


class measurement_model_NN(nn.Module):
    def __init__(self, particle_encoder, likelihood_estimator, params):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.likelihood_estimator = likelihood_estimator
        self.params = params

    def forward(self, encodings, update_particles, environment_data=None, pretrain = False):
        n_batch, n_particles = update_particles.shape[:2]
        # start_time = time.time()
        encodings_state, encodings_obs = features_state_obs(encodings, n_particles, self.particle_encoder,
                                                            update_particles, environment_data, self.params)
        # end_time = time.time()
        #
        # # Compute the elapsed time
        # elapsed_time = end_time - start_time
        #
        # print(f"features_state_obs took {elapsed_time} seconds to run.")

        # start_time = time.time()
        likelihood = self.likelihood_estimator(torch.cat([encodings_obs.reshape([encodings_state.shape[0]] + list(encodings_obs.shape[2:])),
                                                          encodings_state], dim=1))
        # end_time = time.time()
        #
        # # Compute the elapsed time
        # elapsed_time = end_time - start_time
        #
        # print(f"likelihood_estimator took {elapsed_time} seconds to run.")
        # likelihood = self.likelihood_estimator(torch.cat([encodings_obs.reshape(encodings_state.shape),
        #                                                   encodings_state], dim=-1))
        likelihood=likelihood.reshape([n_batch, n_particles,1])
        if pretrain:
            return likelihood[..., 0] + 1e-12
        else:
            return (likelihood[..., 0]+1e-12)  #.log() directly output log liklihood

class measurement_model_Gaussian(nn.Module):
    def __init__(self, particle_encoder, gaussian_distribution, params):
        super().__init__()
        self.particle_encoder = particle_encoder
        #gaussian_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(noise_feature.shape[-1]), torch.eye(noise_feature.shape[-1])).to(device)
        self.gaussian_distribution = gaussian_distribution
        self.params = params

    def forward(self, encodings, update_particles,  environment_data=None):
        n_batch, n_particles = update_particles.shape[:2]
        encodings_state, encodings_obs = features_state_obs(encodings, n_particles, self.particle_encoder,
                                                            update_particles, environment_data, self.params)

        noise_feature = encodings_obs - encodings_state

        likelihood = self.gaussian_distribution.log_prob(noise_feature)
        likelihood = likelihood - likelihood.max(dim=-1, keepdims=True)[0]

        return likelihood

class measurement_model_cnf(nn.Module):
    def __init__(self, particle_encoder, CNF, params):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.CNF = CNF
        self.params = params

    def forward(self, encodings, update_particles, environment_data=None, pretrain=False):
        self.hidden_dim = encodings.shape[-1]
        n_batch, n_particles = update_particles.shape[:2]
        encodings_state, encodings_obs = features_state_obs(encodings, n_particles, self.particle_encoder,
                                                            update_particles, environment_data, self.params)

        encodings_state = encodings_state.reshape([-1, self.hidden_dim])
        encodings_obs = encodings_obs.reshape([-1, self.hidden_dim])
        # if self.params.dataset =='maze':
        #     means, stds, encodings_maps = environment_data
        #     encodings_maps = encodings_maps.repeat(n_batch * n_particles, 1)
        #     encodings_state = torch.cat([encodings_state, encodings_maps], dim=-1)

        z, log_prob_z, log_det = self.CNF.forward(encodings_obs, encodings_state)
        likelihood = (log_prob_z + log_det).reshape([n_batch, n_particles])

        likelihood = likelihood - likelihood.max(dim=-1, keepdims=True)[0]
        if pretrain:
            return likelihood.exp() + 1e-12
        else:
            return likelihood
        # return likelihood

class measurement_model_cglow(nn.Module):
    def __init__(self, particle_encoder, CGLOW):
        super().__init__()
        self.particle_encoder = particle_encoder
        self.CGLOW = CGLOW

    def forward(self, encodings, update_particles):
        n_batch, n_particles, state_dim = update_particles.shape

        update_particles = update_particles.reshape([-1,state_dim])

        encodings_state = state_feature_disk(self.particle_encoder, update_particles)
        encodings_state = encodings_state.reshape([n_batch * n_particles,3,8,8])

        encodings_obs = obs_feature_disk(encodings,n_particles)
        encodings_obs = encodings_obs.reshape([-1]+list(encodings_state.shape[-3:]))

        z, nll = self.CGLOW(encodings_state, encodings_obs)
        #print(z[0].abs().mean(dim=0), z[20].abs().mean(dim=0), z[10].abs().mean(dim=0), z[30].abs().mean(dim=0), log_prob_z.mean())
        likelihood = -nll.reshape([n_batch, n_particles])
        likelihood = likelihood - likelihood.max(dim=-1, keepdims=True)[0]

        return likelihood

def nf_dynamic_model(dynamical_nf, particles_physic, jac_shape, NF=False, forward=False, mean=None, std=None):
    if NF:
        n_batch, n_particles, dimension = particles_physic.shape
        if mean is None:
            dyn_particles_mean, dyn_particles_std = particles_physic.mean(dim=1, keepdim=True).detach().clone().repeat([1, n_particles, 1]), \
                                                    particles_physic.std(dim=1, keepdim=True).detach().clone().repeat([1, n_particles, 1])
        else:
            dyn_particles_mean, dyn_particles_std = mean.detach().clone().repeat([1, n_particles, 1]),\
                                                    std.detach().clone().repeat([1, n_particles, 1])
        dyn_particles_mean_flatten, dyn_particles_std_flatten = dyn_particles_mean.reshape(-1, dimension), dyn_particles_std.reshape(-1,dimension)
        context = torch.cat([dyn_particles_mean_flatten, dyn_particles_std_flatten], dim=-1)
        particles_physic = (particles_physic - dyn_particles_mean) / dyn_particles_std #here

        particles_pred_flatten = particles_physic.reshape(-1, dimension)

        if forward:
            particles_update_nf, _, log_det = dynamical_nf.forward(particles_pred_flatten, context)
        else:
            particles_update_nf, log_det = dynamical_nf.inverse(particles_pred_flatten, context)
        jac_dynamic = -log_det
        jac_dynamic = jac_dynamic.reshape(particles_physic.shape[:2])

        nf_dynamic_particles = particles_update_nf.reshape(particles_physic.shape)
        nf_dynamic_particles = nf_dynamic_particles * dyn_particles_std + dyn_particles_mean #here
    else:
        nf_dynamic_particles=particles_physic
        jac_dynamic = torch.zeros(jac_shape).to(device)
    return nf_dynamic_particles, jac_dynamic

def normalising_flow_propose(cond_model, particles_pred, obs, flow=RealNVP_cond, n_sequence=2, hidden_dimension=8, obser_dim=None):

    B, N, dimension = particles_pred.shape

    pred_particles_mean, pred_particles_std = particles_pred.mean(dim=1, keepdim=True).detach().clone().repeat([1, N, 1]), \
                                            particles_pred.std(dim=1, keepdim=True).detach().clone().repeat([1, N, 1])
    dyn_particles_mean_flatten, dyn_particles_std_flatten = pred_particles_mean.reshape(-1, dimension), pred_particles_std.reshape(-1, dimension)
    context = torch.cat([dyn_particles_mean_flatten, dyn_particles_std_flatten], dim=-1)
    # particles_pred = (particles_pred - pred_particles_mean) / pred_particles_std

    particles_pred_flatten=particles_pred.reshape(-1,dimension)
    obs_reshape = obs[:, None, :].repeat([1,N,1]).reshape(B*N,-1)
    obs_reshape = torch.cat([obs_reshape, context], dim=-1)

    particles_update_nf, log_det=cond_model.inverse(particles_pred_flatten, obs_reshape)

    jac=-log_det
    jac=jac.reshape(particles_pred.shape[:2])

    particles_update_nf=particles_update_nf.reshape(particles_pred.shape)
    # particles_update_nf = particles_update_nf * pred_particles_std + pred_particles_mean

    return particles_update_nf, jac

def proposal_likelihood(cond_model, dynamical_nf, measurement_model, particles_dynamic, particles_physical,
                        encodings, noise, jac_dynamic, NF, NF_cond, prototype_density, environment_measurement, obs_feature,
                        encoder_flow=None):
    # encodings_clone = encodings.detach().clone() #here
    # encodings_clone.requires_grad = False #here ? 

    if NF_cond:
        propose_particle, jac_prop = normalising_flow_propose(cond_model, particles_dynamic, encodings)
        if NF:
            particle_prop_dyn_inv, jac_prop_dyn_inv = nf_dynamic_model(dynamical_nf, propose_particle,jac_dynamic.shape, NF=NF, forward=True,
                                                                       mean=particles_physical.mean(dim=1, keepdim=True),
                                                                       std=particles_physical.std(dim=1, keepdim=True))
            prior_log = prototype_density(particle_prop_dyn_inv - (particles_physical - noise)) - jac_prop_dyn_inv #####
        else:
            prior_log = prototype_density(propose_particle - (particles_physical - noise))
        propose_log = prototype_density(noise) + jac_dynamic + jac_prop
    else:
        propose_particle = particles_dynamic
        prior_log = prototype_density(noise) + jac_dynamic
        propose_log = prototype_density(noise) + jac_dynamic

    encodings = obs_feature
    # start_time = time.time()
    lki_log = measurement_model(encodings, propose_particle, environment_measurement)
    # end_time = time.time()
    #
    # # Compute the elapsed time
    # elapsed_time = end_time - start_time
    #
    # print(f"measurement model took {elapsed_time} seconds to run.")
    return propose_particle, lki_log, prior_log, propose_log