from aesmc import losses
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


def train(dataloader_online1, dataloader_online2, dataloader, num_particles, algorithm, initial, transition, emission,
          proposal, num_epochs, num_iterations_per_epoch=None, num_iterations_per_epoch_online=None,
          optimizer_algorithm=torch.optim.Adam, optimizer_kwargs={},
          callback=None, args=None):
    device = args.device
    parameters_model = get_chained_params(initial, transition, emission)
    parameters_proposal = get_chained_params(proposal)
    optimizer_model = optimizer_algorithm(parameters_model, **optimizer_kwargs)
    optimizer_proposal = optimizer_algorithm(parameters_proposal, **optimizer_kwargs)

    for epoch_idx in range(num_epochs):
        training_stage = 'offline'
        print(training_stage)
        for epoch_iteration_idx, latents_and_observations in enumerate(dataloader):
            true_latents = latents_and_observations[0]
            true_latents = [true_latent.to(device).unsqueeze(-1) if len(true_latent.shape)==1 else true_latent.to(device)
                            for true_latent in true_latents]
            observations = latents_and_observations[1]
            observations = [observation.to(device) for observation in observations]
            if num_iterations_per_epoch is not None:
                if epoch_iteration_idx == num_iterations_per_epoch:
                    break
            optimizer_model.zero_grad()
            optimizer_proposal.zero_grad()
            loss, data_pre_step = losses.get_loss(training_stage, observations, num_particles, algorithm,
                                   initial, transition, emission, proposal, args = args, true_latents=true_latents)
            loss.backward()
            optimizer_model.step()
            optimizer_proposal.step()

            if callback is not None:
                callback(epoch_idx, epoch_iteration_idx, loss, initial,
                         transition, emission, proposal, stage=0)

        training_stage = 'online1'
        online_state = 'start'
        data_current = 0.0
        print(training_stage)
        for epoch_iteration_idx, latents_and_observations in enumerate(dataloader_online1):
            true_latents = latents_and_observations[0]
            true_latents = [x.unsqueeze(0) for x in true_latents]
            true_latents = [true_latent.to(device).unsqueeze(-1) if len(true_latent.shape)==1 else true_latent.to(device)
                            for true_latent in true_latents]
            observations = latents_and_observations[1]
            observations = [x.unsqueeze(0) for x in observations]
            observations = [observation.to(device) for observation in observations]
            if num_iterations_per_epoch_online is not None:
                if epoch_iteration_idx == num_iterations_per_epoch_online:
                    break
            optimizer_model.zero_grad()
            optimizer_proposal.zero_grad()
            loss, data_pre_step = losses.get_loss([data_current,online_state], training_stage, observations, num_particles, algorithm,
                                   initial, transition, emission, proposal, args = args, true_latents=true_latents)
            data_current = data_pre_step
            if args.trainType != 'DPF':
                loss.backward()
                optimizer_model.step()
                optimizer_proposal.step()

            if callback is not None:
                callback(epoch_idx, epoch_iteration_idx, loss, initial,
                         transition, emission, proposal, stage=1)

        # training_stage = 'online2'
        # print(training_stage)
        # for epoch_iteration_idx, latents_and_observations in enumerate(dataloader_online2):
        #     true_latents = latents_and_observations[0]
        #     true_latents = [x.unsqueeze(0) for x in true_latents]
        #     true_latents = [
        #         true_latent.to(device).unsqueeze(-1) if len(true_latent.shape) == 1 else true_latent.to(device)
        #         for true_latent in true_latents]
        #     observations = latents_and_observations[1]
        #     observations = [x.unsqueeze(0) for x in observations]
        #     observations = [observation.to(device) for observation in observations]
        #     if num_iterations_per_epoch_online is not None:
        #         if epoch_iteration_idx == num_iterations_per_epoch_online:
        #             break
        #     optimizer_model.zero_grad()
        #     optimizer_proposal.zero_grad()
        #     loss = losses.get_loss(training_stage, observations, num_particles, algorithm,
        #                            initial, transition, emission, proposal, args=args, true_latents=true_latents)
        #     if args.trainType != 'DPF':
        #         loss.backward()
        #         optimizer_model.step()
        #         optimizer_proposal.step()
        #
        #     if callback is not None:
        #         callback(epoch_idx, epoch_iteration_idx, loss, initial,
        #                  transition, emission, proposal, stage=2)


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
