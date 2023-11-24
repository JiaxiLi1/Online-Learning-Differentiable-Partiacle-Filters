import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def et_distance(encoding_input,e_t):

    # tf.reduce_mean((encoding_input-e_t)**2,axis=-1)
    encoding_input=F.normalize(encoding_input,p=2,dim=-1,eps=1e-12)
    e_t=F.normalize(e_t,p=2,dim=-1, eps=1e-12)
    cosd = 1.0 - torch.sum(encoding_input*e_t,dim=-1)

    return cosd

class compute_normal_density_disk(nn.Module):
    def __init__(self, pos_noise=1.0, vel_noise=1.0):
        super().__init__()
        self.pos_noise=pos_noise
        self.vel_noise=vel_noise
    def forward(self, noise, std_pos = None, std_vel = None):
        log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi))
        if std_pos is None:
            std_pos = self.pos_noise
        if std_vel is None:
            std_vel = self.vel_noise

        noise_pos = noise[:, :, :2]
        noise_vel = noise[:, :, 2:]

        log_prior = noise.shape[-1] * log_c - 2 * torch.log(torch.tensor(std_pos)) - torch.sum(
            noise_pos ** 2 / (2 * torch.tensor(std_pos) ** 2), dim=-1) + \
                    - (noise.shape[-1]-2) * torch.log(torch.tensor(std_vel)) - torch.sum(
            noise_vel ** 2 / (2 * torch.tensor(std_vel) ** 2), dim=-1)

        return log_prior

def normalize_log_probs(probs):
    probs_max=probs.max(dim=-1, keepdims=True)[0]
    probs_minus_max = probs-probs_max
    probs_normalized = probs_minus_max.exp()
    probs_normalized = probs_normalized / torch.sum(probs_normalized, dim=-1, keepdim=True)
    return probs_normalized

def particle_initialization(start_state, param, environment_data=None, train=True):
    if param.dataset == 'disk':
        initial_particles, init_weights_log = particle_initialization_disk(start_state, param.width, param.num_particles,
                                                                           param.state_dim, param.init_with_true_state)
    elif param.dataset == 'maze':
        # if train:
        #     param.init_with_true_state = False
        # else:
        #     param.init_with_true_state = False
        initial_particles, init_weights_log = particle_initialization_maze(start_state, param.num_particles, param.std_x, param.std_t,
                                                                           environment_data, param.state_dim, param.init_with_true_state)
    elif param.dataset == 'house3d':
        if train:
            param.init_with_true_state = False
        else:
            param.init_with_true_state = False
        initial_particles, init_weights_log = particle_initialization_house3d(start_state, param.num_particles, param.std_x, param.std_t,
                                                                           environment_data, param.state_dim, param.init_with_true_state)

    return initial_particles, init_weights_log

def particle_initialization_disk(start_state, width, num_particles, state_dim=2,init_with_true_state=False):
    batch_size = start_state.shape[0]
    if init_with_true_state:
        initial_noise = torch.randn(batch_size, num_particles, state_dim).to(device)
        initial_particles = start_state[:, None, :].repeat(1, num_particles, 1) + initial_noise
    else:
        bound_max = width / 2.0
        bound_min = -width / 2.0
        pos = torch.tensor((bound_max - bound_min)).to(device) * torch.rand(batch_size, num_particles,
                                                                            2).to(device) + torch.tensor(
            bound_min).to(device)
        initial_particles = pos
        vel = torch.randn(batch_size, num_particles, 2).to(device)

    init_weights_log = torch.log(torch.ones([batch_size, num_particles]).to(device) / num_particles)

    return initial_particles, init_weights_log

def particle_initialization_maze(start_state, num_particles, std_x, std_t, environment_data, state_dim=2,init_with_true_state=False):
    maps, statistics = environment_data
    (means, stds, state_step_sizes, state_mins, state_maxs) = statistics
    batch_size = start_state.shape[0]
    if init_with_true_state:
        initial_noise_pos = torch.normal(mean=0., std=std_x,
                                         size=(batch_size, num_particles, 2)).to(device)
        initial_noise_t = torch.normal(mean=0., std=std_t, size=(batch_size, num_particles, 1)).to(
            device)
        initial_noise = torch.cat([initial_noise_pos, initial_noise_t], dim=-1)

        initial_particles = start_state[:, None, :].repeat(1, num_particles, 1) + initial_noise
    else:
        initial_particles = torch.cat(
            [(torch.tensor(state_maxs[d]).to(device) - torch.tensor(state_mins[d]).to(device)) * (
                torch.rand(batch_size, num_particles, 1).to(device)) + torch.tensor(state_mins[d]).to(device) for d
             in
             range(state_dim)], dim=2)
    log_mu_s = torch.log(torch.ones([batch_size, num_particles]) / initial_particles.shape[1]).to(device)
    init_weights_log = log_mu_s

    return initial_particles, init_weights_log

def particle_initialization_house3d(start_state, num_particles, std_x, std_t, environment_data, state_dim=2,init_with_true_state=False):
    maps, statistics = environment_data
    (state_mins, state_maxs) = statistics
    batch_size = start_state.shape[0]
    if init_with_true_state:
        initial_noise_pos = torch.normal(mean=0., std=std_x,
                                         size=(batch_size, num_particles, 2)).to(device)
        initial_noise_t = torch.normal(mean=0., std=std_t, size=(batch_size, num_particles, 1)).to(
            device)
        initial_noise = torch.cat([initial_noise_pos, initial_noise_t], dim=-1)

        initial_particles = start_state[:, None, :].repeat(1, num_particles, 1) + initial_noise
    else:
        initial_particles = torch.cat(
            [(state_maxs[d] - state_mins[d])[:,None, None] * (
                torch.rand(batch_size, num_particles, 1).to(device)) + state_mins[d][:,None, None] for d
             in
             range(state_dim)], dim=2)
    log_mu_s = torch.log(torch.ones([batch_size, num_particles]) / initial_particles.shape[1]).to(device)
    init_weights_log = log_mu_s

    return initial_particles, init_weights_log

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def checkpoint_state(model,epoch):
    state_dict={
        "model": model.state_dict(),
        'model_optim': model.optim.state_dict(),
        'model_optim_scheduler': model.optim_scheduler.state_dict(),
        "epoch": epoch
    }
    return state_dict

def load_model(model, ckpt_e2e):
    model.load_state_dict(ckpt_e2e['model'])
    model.optim.load_state_dict(ckpt_e2e['model_optim'])
    model.optim_scheduler.load_state_dict(ckpt_e2e['model_optim_scheduler'])

def noisyfy_data(measurements):
    new_o = np.zeros([measurements.shape[0], measurements.shape[1],24, 24, 4])
    for i in range(measurements.shape[0]):
        for j in range(measurements.shape[1]):
            offsets = np.random.random_integers(0, 8, 2)
            new_o[i, j] = measurements[i, j, offsets[0]:offsets[0] + 24, offsets[1]:offsets[1] + 24, :]
    new_o += np.random.normal(0.0, 20, new_o.shape)
    return new_o


def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi

def compute_statistics(data):
    means = dict()
    stds = dict()
    state_step_sizes = []
    state_mins = []
    state_maxs = []

    for key in 'osa':
        # compute means
        means[key] = np.mean(data[key], axis=(0, 1), keepdims=True)
        if key == 's':
            means[key][:, :, 2] = 0  # don't touch orientation because we'll feed this into cos/sin functions
        if key == 'a':
            means[key][:, :, :] = 0  # don't change means of velocities, 0.0, positive and negative values have semantics

        # compute stds
        axis = tuple(range(len(data[key].shape) - 1))  # compute std by averaging over all but the last dimension
        stds[key] = np.std(data[key] - means[key], axis=axis, keepdims=True)
        if key == 's':
            stds[key][:, :, :2] = np.mean(stds[key][:, :, :2])  # scale x and by by the same amount
        if key == 'a':
            stds[key][:, :, :2] = np.mean(stds[key][:, :, :2])  # scale x and by by the same amount

    # compute average step size in x, y, and theta for the distance metric
    for i in range(3):
        steps = np.reshape(data['s'][:, 1:, i] - data['s'][:, :-1, i], [-1])
        if i == 2:
            steps = wrap_angle(steps)
        state_step_sizes.append(np.mean(abs(steps)))
    state_step_sizes[0] = state_step_sizes[1] = (state_step_sizes[0] + state_step_sizes[1]) / 2
    state_step_sizes = np.array(state_step_sizes)

    # compute min and max in x, y and theta
    for i in range(3):
        state_mins.append(np.min(data['s'][:, :, i]))
        state_maxs.append(np.max(data['s'][:, :, i]))
    state_mins = np.array(state_mins)
    state_maxs = np.array(state_maxs)

    return (means, stds, state_step_sizes, state_mins, state_maxs)



def particles_to_prediction(particle_list, particle_probs_list):
    mean_position = torch.sum(particle_probs_list[:, :, :, None] * particle_list[:, :, :, :2], dim=2)
    mean_orientation = wrap_angle(torch.atan2(
        torch.sum(particle_probs_list[:, :, :, None] * torch.cos(particle_list[:, :, :, 2:]), dim=2),
        torch.sum(particle_probs_list[:, :, :, None] * torch.sin(particle_list[:, :, :, 2:]), dim=2)
    ))
    return torch.cat([mean_position, mean_orientation], dim=2)

def transform_particles_as_input(particles, means, stds):
    # particles = torch.cat([particles[:, :, :2].to(device), torch.cos(particles[:, :, 2:3]), torch.sin(particles[:, :, 2:3])], dim=-1)
    # particles_mean, particles_std = particles.mean(dim=1, keepdim=True).detach().clone(), particles.std(dim=1, keepdim=True).detach().clone()
    # particles[:, :, :2] = (particles[:, :, :2] - particles_mean[:, :, :2])/particles_std[:, :, :2]
    #
    # particles_mean, particles_std = particles_mean.repeat(1, particles.shape[1], 1), particles_std.repeat(1, particles.shape[1], 1)
    # particles_mean[:, :, :2] = torch.cat([(particles_mean[:, :, :2].to(device) - torch.tensor(means['s'][:, :, :2]).to(device))/
    #                             torch.tensor(stds['s'][:, :, :2]).to(device)], dim=-1)
    #
    # return torch.cat([particles, particles_mean], dim=-1)
    return torch.cat([
                   (particles[:, :, :2] - torch.tensor(means['s'][:, :, :2]).to(device)) / torch.tensor(stds['s'][:, :, :2]).to(device),  # normalized pos
                   torch.cos(particles[:, :, 2:3]),  # cos
                   torch.sin(particles[:, :, 2:3])], dim=-1)

def transform_particles_as_input_house3d(particles):
    return torch.cat([
        particles[:, :, :2],  # normalized pos
        torch.cos(particles[:, :, 2:3]),  # cos
        torch.sin(particles[:, :, 2:3])],  # sin
        dim=-1
    )


def preprocess_obs(observations, environment_obs, params):
    if params.dataset == 'maze':
        means_o, stds_o = environment_obs
        observations_transform = (observations - means_o) / stds_o
    elif params.dataset == 'disk':
        observations_transform = observations
    elif params.dataset == 'house3d':
        observations_transform = observations
    else:
        raise ValueError('Pease select a dataset from {disk, maze, house3d}')
    return observations_transform.float()

