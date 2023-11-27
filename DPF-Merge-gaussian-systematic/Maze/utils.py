import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from distributions import rand_cirlce2d

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def particles_to_prediction(particle_list, particle_probs_list):
    mean_position = torch.sum(particle_probs_list[:, :, :, None] * particle_list[:, :, :, :2], dim=2)
    mean_orientation = wrap_angle(torch.atan2(
        torch.sum(particle_probs_list[:, :, :, None] * torch.cos(particle_list[:, :, :, 2:]), dim=2),
        torch.sum(particle_probs_list[:, :, :, None] * torch.sin(particle_list[:, :, :, 2:]), dim=2)
    ))
    return torch.cat([mean_position, mean_orientation], dim=2)

def transform_particles_as_input(particles, means, stds):
    return torch.cat([
        (particles[:, :, :2] - torch.tensor(means['s'][:, :, :2]).to(device)) / torch.tensor(stds['s'][:, :, :2]).to(
            device),  # normalized pos
        torch.cos(particles[:, :, 2:3]),  # cos
        torch.sin(particles[:, :, 2:3])],  # sin
        dim=-1
    )

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

def et_distance_old(encoding_input,e_t):

    # tf.reduce_mean((encoding_input-e_t)**2,axis=-1)
    encoding_input=F.normalize(encoding_input,p=2,dim=-1,eps=1e-12)
    e_t=F.normalize(e_t,p=2,dim=-1, eps=1e-12)
    cosd = torch.ones(e_t.shape[0], e_t.shape[1]).to(device) - torch.sum(encoding_input*e_t,dim=-1)

    return cosd

def et_distance(encoding_input,e_t):

    # tf.reduce_mean((encoding_input-e_t)**2,axis=-1)
    encoding_input=F.normalize(encoding_input,p=2,dim=-1,eps=1e-12)
    e_t=F.normalize(e_t,p=2,dim=-1, eps=1e-12)
    cosd = 1.0 - torch.sum(encoding_input*e_t,dim=-1)

    return cosd

def generate_mask(args, state):
    N1 = int(state.shape[0]*state.shape[1]*args.labeledRatio)
    N0 = state.shape[0]*state.shape[1] - N1
    arr = np.array([0] * N0 + [1] * N1)
    np.random.shuffle(arr)
    mask = arr.reshape(state.shape[0], state.shape[1])
    return mask

def noisyfy_data(measurements):
    new_o = np.zeros([measurements.shape[0], measurements.shape[1],24, 24, 4])
    for i in range(measurements.shape[0]):
        for j in range(measurements.shape[1]):
            offsets = np.random.random_integers(0, 8, 2)
            new_o[i, j] = measurements[i, j, offsets[0]:offsets[0] + 24, offsets[1]:offsets[1] + 24, :]
    new_o += np.random.normal(0.0, 20, new_o.shape)
    return new_o

def compute_sq_distance(a, b, state_step_sizes):
    result = 0.0
    for i in range(a.shape[-1]):
        # compute difference
        diff = a[..., i] - b[..., i]
        # wrap angle for theta
        if i == 2:
            diff = wrap_angle(diff)
        # add up scaled squared distance
        result += (diff / state_step_sizes[i]) ** 2
    return result

def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(encoded_samples,
                                distribution_fn=rand_cirlce2d,
                                num_projections=50,
                                p=2,
                                device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    batch_size = encoded_samples.size(0)
    # draw random samples from latent space prior distribution
    z = distribution_fn(batch_size).to(device)
    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, z,
                                       num_projections, p, device)
    return swd

def normal_log_density(noise, std):

    log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi))
    dimension = noise.shape[-1]
    # log_prior = (2 * log_c - 2 * torch.log(torch.tensor(std_pos)) - torch.sum(
    #     noise_pos ** 2 / (2 * torch.tensor(std_pos) ** 2), dim=-1))

    log_density =  dimension * log_c - dimension * torch.log(torch.tensor(std)) - torch.sum(
        noise ** 2 / (2*torch.tensor(std) ** 2), dim=-1)

    return log_density

def normalize_log_probs(probs):
    probs_max=probs.max(dim=1, keepdims=True)[0]
    probs_minus_max = probs-probs_max
    probs_normalized = probs_minus_max.exp()
    probs_normalized = probs_normalized / torch.sum(probs_normalized, dim=1, keepdim=True)
    return probs_normalized

class compute_normal_density(nn.Module):
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
                    (noise.shape[-1]-2) * torch.log(torch.tensor(std_vel)) - torch.sum(
            noise_vel ** 2 / (2 * torch.tensor(std_vel) ** 2), dim=-1)

        return log_prior