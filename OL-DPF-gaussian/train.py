import losses
import statistics
import itertools
import torch.nn as nn
import torch.utils.data
import numpy as np
def find_min_max_in_states(state_list):
    concatenated_states = np.concatenate(state_list, axis=0)
    min_values = np.amin(concatenated_states, axis=0)
    max_values = np.amax(concatenated_states, axis=0)
    return min_values, max_values

def sample_initial_particles(min_values, max_values, batch_size_online, num_particles, dim):
    # Sample uniformly within the min-max range for each dimension
    initial_particles = np.random.uniform(min_values, max_values, (batch_size_online, num_particles, dim))
    return initial_particles

def to_gpu_tensor(numpy_array):
    # Convert numpy array to PyTorch tensor and transfer to GPU
    return torch.tensor(numpy_array).float().to('cuda')

def get_chained_params(*objects):
    result = []
    for object in objects:
        if (object is not None) and isinstance(object, nn.Module):
            result = itertools.chain(result, object.parameters())

    if isinstance(result, list):
        return None
    else:
        return result
def create_mask(time_steps, batch_size, labelled_ratio):
    total_data_points = batch_size * time_steps
    retain_data_points = int(total_data_points * labelled_ratio)

    # Create and shuffle the flattened mask
    mask_flat = torch.cat((torch.ones(retain_data_points), torch.zeros(total_data_points - retain_data_points)))
    mask_flat = mask_flat[torch.randperm(total_data_points)]

    # Reshape the mask
    mask = mask_flat.view(time_steps, batch_size)

    return mask

def train(initial_state, dataloader_val, dataloader_test, dataloader, num_particles, algorithm, initial, transition, emission,
          proposal, num_epochs, num_iterations_per_epoch=None, num_iterations_per_epoch_online=None,
          optimizer_algorithm=torch.optim.Adam, optimizer_kwargs={},
          callback=None, args=None):
    device = args.device
    parameters_model = get_chained_params(initial, transition, emission)
    parameters_proposal = get_chained_params(proposal)
    optimizer_model = optimizer_algorithm(parameters_model, **optimizer_kwargs)
    optimizer_proposal = optimizer_algorithm(parameters_proposal, **optimizer_kwargs)
    rmse_plot = []
    elbo_plot = []
    rmse_box_plot = []

    mask = create_mask(100, 10, args.labelled_ratio).to(device)

    for epoch_idx in range(num_epochs):
        rmse_temp = []
        elbo_temp = []
        for epoch_iteration_idx, latents_and_observations in enumerate(dataloader):
            true_latents = latents_and_observations[0]
            true_latents = [true_latent.to(device).detach().unsqueeze(-1) if len(true_latent.shape)==1 else true_latent.to(device).detach()
                            for true_latent in true_latents]
            observations = latents_and_observations[1]
            observations = [observation.to(device).detach() for observation in observations]
            optimizer_model.zero_grad()
            optimizer_proposal.zero_grad()
            loss, data_pre_step, rmse_list, elbo, loss_rmse = losses.get_loss(mask, initial_state.detach(), observations, num_particles, algorithm,
                                   initial, transition, emission, proposal, args = args, true_latents=true_latents, measurement = args.measurement)
            loss.backward()
            rmse_temp.append(loss_rmse.detach().cpu().numpy())
            elbo_temp.append(elbo.detach().cpu().numpy())
            optimizer_model.step()
            optimizer_proposal.step()

        print('train', 'epoch', epoch_idx + 1, 'rmse', np.mean(rmse_temp), 'elbo', np.mean(elbo_temp))
        rmse_temp = []
        elbo_temp = []
        with torch.no_grad():
            for epoch_iteration_idx_val, latents_and_observations in enumerate(dataloader_val):
                true_latents = latents_and_observations[0]
                true_latents = [
                    true_latent.to(device).detach().unsqueeze(-1) if len(
                        true_latent.shape) == 1 else true_latent.to(
                        device).detach()
                    for true_latent in true_latents]
                observations = latents_and_observations[1]
                observations = [observation.to(device).detach() for observation in observations]

                optimizer_model.zero_grad()
                optimizer_proposal.zero_grad()
                loss, data_pre_step, rmse_list, elbo, loss_rmse = losses.get_loss(mask, initial_state.detach(),
                                                                                  observations,
                                                                                  num_particles, algorithm,
                                                                                  initial, transition, emission,
                                                                                  proposal, args=args,
                                                                                  true_latents=true_latents,
                                                                                  measurement=args.measurement)
                elbo_save = elbo.detach().cpu().numpy()
                rmse_save = loss_rmse.detach().cpu().numpy()
                rmse_temp.append(rmse_save)
                elbo_temp.append(elbo_save)
            mean_rmse = np.mean(rmse_temp)
            mean_elbo = np.mean(elbo_temp)
            rmse_plot.append(mean_rmse)
            elbo_plot.append(mean_elbo)
            print('val', 'epoch', epoch_idx + 1, 'rmse', mean_rmse, 'elbo', mean_elbo)

    with torch.no_grad():
        rmse_temp = []
        elbo_temp = []
        for epoch_iteration_idx, latents_and_observations in enumerate(dataloader_test):
            true_latents = latents_and_observations[0]
            true_latents = [
                true_latent.to(device).detach().unsqueeze(-1) if len(true_latent.shape) == 1 else true_latent.to(
                    device).detach()
                for true_latent in true_latents]
            observations = latents_and_observations[1]
            observations = [observation.to(device).detach() for observation in observations]

            optimizer_model.zero_grad()
            optimizer_proposal.zero_grad()
            loss, data_pre_step, rmse_list, elbo, loss_rmse = losses.get_loss(mask, initial_state.detach(), observations, num_particles, algorithm,
                                   initial, transition, emission, proposal, args = args, true_latents=true_latents, measurement = args.measurement)
            elbo_save = elbo.detach().cpu().numpy()
            rmse_save = loss_rmse.detach().cpu().numpy()
            rmse_temp.append(rmse_save)
            elbo_temp.append(elbo_save)
            rmse_box_plot.append(rmse_list)
        mean_rmse = np.mean(rmse_temp)
        mean_elbo = np.mean(elbo_temp)
        rmse_plot.append(mean_rmse)
        elbo_plot.append(mean_elbo)
        print('test', 'rmse', mean_rmse, 'elbo', mean_elbo)

    return rmse_plot, elbo_plot, [tensor.detach().cpu().numpy() for tensor in rmse_box_plot]

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, latents, observations, initial, transition, emission, num_timesteps,
                 batch_size):
        self.latents = latents
        self.observations = observations
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

    def __getitem__(self, index):
        """
        For each timestep, clip a batch of data from the latents and observations lists.
        """
        # Calculate start and end indices for the batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Clip the batch for each timestep
        batch_latents = [latent[start_idx:end_idx] for latent in self.latents]
        batch_observations = [observation[start_idx:end_idx] for observation in self.observations]

        return batch_latents, batch_observations

    def __len__(self):
        return len(self.latents[0]) // self.batch_size

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
                             batch_size, num_iter, num_particles, dim):
    latents,observations = statistics.sample_from_prior(initial, transition,
                                                            emission, num_timesteps,
                                                            batch_size*num_iter)
    latents_cpu = [latent.detach().cpu().numpy() for latent in latents]
    min_values, max_values = find_min_max_in_states(latents_cpu)
    initial_particles = sample_initial_particles(min_values, max_values, batch_size, num_particles, dim)
    initial_particles = to_gpu_tensor(initial_particles)
    return initial_particles, torch.utils.data.DataLoader(
        SyntheticDataset(latents,observations,initial, transition, emission, num_timesteps,
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
