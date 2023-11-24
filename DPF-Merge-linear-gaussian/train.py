
from aesmc import statistics

import itertools
import sys
import torch.nn as nn
import torch.utils.data

def get_chained_params(*objects):
    result = []
    for object in objects:
        if (object is not None) and isinstance(object, nn.Module):
            result = itertools.chain(result, object.parameters())

    if isinstance(result, list):
        return None
    else:
        return result


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, initial, transition, emission, num_timesteps,
                 batch_size):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

    def __getitem__(self, index):
        # TODO this is wrong, obs can be dict
        latents_and_observations = statistics.sample_from_prior(self.initial, self.transition,
                                     self.emission, self.num_timesteps,
                                     self.batch_size)
        return [list(map(lambda latent: latent.detach().squeeze(0), latents_and_observations[0])),
                list(map(lambda observation: observation.detach().squeeze(0), latents_and_observations[1]))]

    def __len__(self):
        return sys.maxsize  # effectively infinite

class SyntheticDataset_online(torch.utils.data.Dataset):
    def __init__(self, initial, transition, emission, num_timesteps,
                 batch_size, total_timesteps=10000):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.total_timesteps = total_timesteps

        # Generate one trajectory of length total_timesteps
        self.latents_and_observations = statistics.sample_from_prior(
            self.initial, self.transition, self.emission, self.total_timesteps, self.batch_size)
        self.latents = list(map(lambda latent: latent.detach().squeeze(0), self.latents_and_observations[0]))
        self.observations = list(map(lambda observation: observation.detach().squeeze(0), self.latents_and_observations[1]))

    def __getitem__(self, index):
        start_idx = index * self.num_timesteps
        end_idx = start_idx + self.num_timesteps

        latents_batch = self.latents[start_idx:end_idx]
        observations_batch = self.observations[start_idx:end_idx]

        return [latents_batch, observations_batch]

    def __len__(self):
        return self.total_timesteps // self.num_timesteps  # total_timesteps must be divisible by num_timesteps

def get_synthetic_dataloader(initial, transition, emission, num_timesteps,
                             batch_size):
    return torch.utils.data.DataLoader(
        SyntheticDataset(initial, transition, emission, num_timesteps,
                         batch_size),
        batch_size=1,
        collate_fn=lambda x: x[0])

def get_synthetic_dataloader_online(initial, transition, emission, num_timesteps,
                             batch_size, total_timesteps=10000):
    return torch.utils.data.DataLoader(
        SyntheticDataset_online(initial, transition, emission, num_timesteps,
                         batch_size, total_timesteps),
        batch_size=1,
        collate_fn=lambda x: x[0])
