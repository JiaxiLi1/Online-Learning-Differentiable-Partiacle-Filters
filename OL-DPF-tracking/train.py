import losses
import statistics
import matplotlib.pyplot as plt
import itertools
import sys
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
# from tqdm import tqdm

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def checkpoint_state(initial, transition, emission, proposal, optimizer_model, optimizer_proposal, epoch):
    state_dict = {
        "initial_state": initial.state_dict(),
        "transition_state": transition.state_dict(),
        "emission_state": emission.state_dict(),
        "proposal_state": proposal.state_dict(),
        "optimizer_model_state": optimizer_model.state_dict(),
        "optimizer_proposal_state": optimizer_proposal.state_dict(),
        "epoch": epoch
    }
    return state_dict


def load_checkpoint(checkpoint_path, initial, transition, emission, proposal, optimizer_model, optimizer_proposal):
    checkpoint = torch.load(checkpoint_path)
    initial.load_state_dict(checkpoint['initial_state'])
    transition.load_state_dict(checkpoint['transition_state'])
    emission.load_state_dict(checkpoint['emission_state'])
    proposal.load_state_dict(checkpoint['proposal_state'])
    optimizer_model.load_state_dict(checkpoint['optimizer_model_state'])
    optimizer_proposal.load_state_dict(checkpoint['optimizer_proposal_state'])
    return checkpoint['epoch']


def find_min_max_in_states(state_list):
    concatenated_states = np.concatenate(state_list, axis=0)
    min_values = np.amin(concatenated_states, axis=0)
    max_values = np.amax(concatenated_states, axis=0)

    # Mean and Variance
    mean_values = np.mean(concatenated_states, axis=0)
    std_values = np.std(concatenated_states, axis=0)

    normalising_value = (max_values-min_values)/2
    normalising_value = np.array([(500, 500, 50, 50, 10)])
    return min_values, max_values, normalising_value

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

def plot_particles(particles, weights, true_state, title, save_dir, adjust=1.0, plot_lag=10):
    particles = particles.detach().cpu().numpy()
    true_state = true_state.squeeze().detach().cpu().numpy()
    weights = weights.detach().cpu().numpy()
    n_timesteps = particles.shape[0]  # Last 20 timesteps
    n_plot_lag = plot_lag
    n_cols = 4  # Max 4 subplots per row
    total_plot = (int(n_timesteps/n_plot_lag))
    n_rows = total_plot // n_cols + (total_plot % n_cols > 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    previous_particles_i = None  # Initialize variable for storing previous particles
    previous_weight_i = None
    ax_count = -1

    for i in range(n_timesteps):
        if i==0 or (i+1)%n_plot_lag ==0:
        # if (n_timesteps - i) % n_plot_lag == 0:
            ax_count += 1
            row, col = divmod(ax_count, n_cols)
            # row, col = divmod(int(((n_timesteps - i) / n_plot_lag)-1), n_cols)
            ax = axes[row, col]

            particles_i=particles[i]
            true_state_i=true_state[i]
            weights_i=weights[i]

            # Update previous particles for the next iteration
            if i != 0:
                previous_particles_i = particles[i - 1]
                previous_weights_i = weights[i - 1]
                previous_true_states = true_state[i - 1]

            weighted_avg_position = np.average(particles_i, axis=0, weights=weights_i)

            # Plot weighted average position
            # Plot true state
            x_vel = weighted_avg_position[2]  # x_velocity
            y_vel = weighted_avg_position[3]  # y_velocity
            # # Optional: Normalize the velocity vector
            norm = np.sqrt(x_vel ** 2 + y_vel ** 2)
            if norm != 0:
                x_vel /= norm
                y_vel /= norm
            ax.quiver(weighted_avg_position[0], weighted_avg_position[1], x_vel, y_vel,
                      color='black', scale=10, linewidth=1)

            # Plot previous particles (if available)
            if previous_particles_i is not None:
                for prev_particle, prev_weight in zip(previous_particles_i, previous_weights_i):
                    x_vel = prev_particle[2]  # x_velocity
                    y_vel = prev_particle[3]  # y_velocity
                    # # Optional: Normalize the velocity vector
                    norm = np.sqrt(x_vel ** 2 + y_vel ** 2)
                    if norm != 0:
                        x_vel /= norm
                        y_vel /= norm
                    if prev_weight <0.02:
                        prev_weight+=0.1
                    ax.quiver(prev_particle[0], prev_particle[1], x_vel, y_vel,
                              color='green', alpha=prev_weight/adjust, scale=15, linewidth=8)
                # Plot true state
                x_vel = previous_true_states[2]  # x_velocity
                y_vel = previous_true_states[3]  # y_velocity
                # # Optional: Normalize the velocity vector
                norm = np.sqrt(x_vel ** 2 + y_vel ** 2)
                if norm != 0:
                    x_vel /= norm
                    y_vel /= norm
                ax.quiver(previous_true_states[0], previous_true_states[1], x_vel, y_vel,
                          color='red', alpha=0.5, scale=10, linewidth=1)

            # Plot current particles
            for particle, weight in zip(particles_i, weights_i):
                x_vel = particle[2]  # x_velocity
                y_vel = particle[3]  # y_velocity
                # # Optional: Normalize the velocity vector
                norm = np.sqrt(x_vel ** 2 + y_vel ** 2)
                if norm != 0:
                    x_vel /= norm
                    y_vel /= norm
                if weight <0.02:
                    weight +=0.1
                ax.quiver(particle[0], particle[1], x_vel, y_vel,
                          color='blue', alpha=weight / adjust, scale=15, linewidth=8)

            # Plot true state
            x_vel = true_state_i[2]  # x_velocity
            y_vel = true_state_i[3]  # y_velocity
            # # Optional: Normalize the velocity vector
            norm = np.sqrt(x_vel ** 2 + y_vel ** 2)
            if norm != 0:
                x_vel /= norm
                y_vel /= norm
            ax.quiver(true_state_i[0], true_state_i[1], x_vel, y_vel,
                      color='red', scale=10, linewidth=1)

            # Set subplot size and aspect
            ax.set_title(f"Time Step {i+1}")
            # ax.set_title(f"Time Step {i + 1}")
            # ax.set_xlim([-301, 1101])
            # ax.set_ylim([-901, 501])
            ax.set_xlim([-1101, 301])
            ax.set_ylim([-101, 1301])
            ax.set_aspect('equal', adjustable='box')

    # Adjust layout and display
    plt.tight_layout()
    filename = title.replace(' ', '_') + '.png'
    saving_path = os.path.join(save_dir, filename)
    # Save the plot , weight_decay=1e-5
    plt.savefig(saving_path, dpi=150)
    # plt.show()
    # Close the plot to free up memory
    plt.close()

def plot_trajectories(predicted_state, ground_truth, title, save_dir):
    # Extracting the predicted and true coordinates
    predicted_state = predicted_state.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()
    predicted_x = predicted_state[:, 0]
    predicted_y = predicted_state[:, 1]
    true_x = ground_truth[:, 0]
    true_y = ground_truth[:, 1]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_x, predicted_y, label='Predicted Trajectory', color='blue')
    plt.plot(true_x, true_y, label='True Trajectory', color='red')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(title)
    plt.legend()
    # Generate saving path from title
    filename = title.replace(' ', '_') + '.png'
    saving_path = os.path.join(save_dir, filename)
    # Save the plot , weight_decay=1e-5
    plt.savefig(saving_path)
    # plt.show()
    # Close the plot to free up memory
    plt.close()

def train(normalising_value, saving_folder, initial_state, dataloader_val, dataloader_test, dataloader, num_particles, algorithm, initial, transition, emission,
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
    mask = create_mask(100, 10, args.labelled_ratio).to(device)
    best_eval_loss = 1e10

    for epoch_idx in range(num_epochs):
        rmse_temp = []
        elbo_temp = []
        for epoch_iteration_idx, latents_and_observations in enumerate(dataloader):
            true_latents = latents_and_observations[0]
            true_latents = [true_latent.to(device) if len(true_latent.shape)==1 else true_latent.to(device)
                            for true_latent in true_latents]#.unsqueeze(-1)
            observations = latents_and_observations[1]
            observations = [observation.to(device) for observation in observations]
            if num_iterations_per_epoch is not None:
                if epoch_iteration_idx == num_iterations_per_epoch:
                    break
            optimizer_model.zero_grad()
            optimizer_proposal.zero_grad()
            loss, data_pre_step, rmse_list, elbo, loss_rmse, predicted_state, ground_truth, particles, normalized_particle_weights = losses.get_loss(normalising_value, mask, initial_state, observations, num_particles, algorithm,
                                   initial, transition, emission, proposal, args = args, true_latents=true_latents, measurement = args.measurement)
            if loss != 0:
                loss.backward()
                optimizer_model.step()
                optimizer_proposal.step()
            rmse_temp.append(loss_rmse.detach().cpu().numpy())
            elbo_temp.append(elbo.detach().cpu().numpy())

        print('train', 'epoch', epoch_idx + 1, 'rmse', np.mean(rmse_temp), 'elbo', np.mean(elbo_temp))
        # if (epoch_idx + 1) % 5 == 0:
        #     plot_trajectories(predicted_state, ground_truth[:, 0, 0:2], f'T_Train_Ep {epoch_idx + 1}', saving_folder)
        #     plot_particles(particles, normalized_particle_weights, ground_truth, f'P_Train_Ep {epoch_idx + 1}', saving_folder)
        rmse_temp = []
        elbo_temp = []
        with torch.no_grad():
            for epoch_iteration_idx, latents_and_observations in enumerate(dataloader_val):
                true_latents = latents_and_observations[0]
                true_latents = [true_latent.to(device) if len(true_latent.shape) == 1 else true_latent.to(device)
                                for true_latent in true_latents]  # .unsqueeze(-1)
                observations = latents_and_observations[1]
                observations = [observation.to(device) for observation in observations]
                optimizer_model.zero_grad()
                optimizer_proposal.zero_grad()
                loss, data_pre_step, rmse_list, elbo, loss_rmse, predicted_state, ground_truth, particles, normalized_particle_weights = losses.get_loss(normalising_value, mask, initial_state, observations,
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
            # if (epoch_idx + 1) % 5 == 0:
            #     plot_trajectories(predicted_state, ground_truth[:, 0, 0:2], f'T_Val_Ep {epoch_idx + 1}', saving_folder)
            #     plot_particles(particles, normalized_particle_weights, ground_truth, f'P_Val_Ep {epoch_idx + 1}',
            #                    saving_folder)
        if mean_rmse < best_eval_loss:
            best_eval_loss = mean_rmse
            print('Save best validation model')
            checkpoint = checkpoint_state(initial, transition, emission, proposal, optimizer_model, optimizer_proposal,
                                          epoch_idx)
            torch.save(checkpoint, os.path.join(saving_folder, 'model_checkpoint.pth'))

    rmse_box_plot = []
    with torch.no_grad():
        rmse_temp = []
        elbo_temp = []
        # Load the model
        epoch = load_checkpoint(
            os.path.join(saving_folder, 'model_checkpoint.pth'),
            initial, transition, emission, proposal,
            optimizer_model, optimizer_proposal
        )
        for epoch_iteration_idx, latents_and_observations in enumerate(dataloader_test):
            true_latents = latents_and_observations[0]
            true_latents = [true_latent.to(device) if len(true_latent.shape) == 1 else true_latent.to(device)
                            for true_latent in true_latents]  # .unsqueeze(-1)
            observations = latents_and_observations[1]
            observations = [observation.to(device) for observation in observations]

            optimizer_model.zero_grad()
            optimizer_proposal.zero_grad()
            loss, data_pre_step, rmse_list, elbo, loss_rmse, predicted_state, ground_truth, particles, normalized_particle_weights = losses.get_loss(normalising_value, mask, initial_state.detach(),
                                                                              observations, num_particles, algorithm,
                                                                              initial, transition, emission, proposal,
                                                                              args=args, true_latents=true_latents,
                                                                              measurement=args.measurement)
            elbo_save = elbo.detach().cpu().numpy()
            rmse_save = loss_rmse.detach().cpu().numpy()
            rmse_temp.append(rmse_save)
            elbo_temp.append(elbo_save)
            rmse_box_plot.append(rmse_list)
        mean_rmse = np.mean(rmse_temp)
        mean_elbo = np.mean(elbo_temp)
        rmse_plot.append(mean_rmse)
        elbo_plot.append(mean_elbo)
        print('test', 'rmse', mean_rmse, 'elbo', mean_elbo, 'epoch in val', epoch)
        # plot_trajectories(predicted_state, ground_truth[:, 0, 0:2], f'T_Test', saving_folder)
        # plot_particles(particles, normalized_particle_weights, ground_truth, f'P_Test',
        #                saving_folder)

    return rmse_plot, elbo_plot, [tensor.detach().cpu().numpy() for tensor in rmse_box_plot]


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

class SyntheticDataset_online_position(torch.utils.data.Dataset):
    def __init__(self, initial, transition, emission, num_timesteps,
                 batch_size, states, observations, total_timesteps=10000):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.total_timesteps = total_timesteps

        self.latents = states
        self.observations = observations

    def __getitem__(self, index):
        start_idx = index * self.num_timesteps
        end_idx = start_idx + self.num_timesteps

        latents_batch = self.latents[start_idx:end_idx]
        observations_batch = self.observations[start_idx:end_idx]

        return [latents_batch, observations_batch]

    def __len__(self):
        return self.total_timesteps // self.num_timesteps  # total_timesteps must be divisible by num_timesteps

class SyntheticDataset_offline_position(torch.utils.data.Dataset):
    def __init__(self, initial, transition, emission, num_timesteps,
                 batch_size, states, observations):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

        self.latents = states
        self.observations = observations

    def __getitem__(self, index):
        start_batch_idx = index * 10
        end_batch_idx = start_batch_idx + 10

        end_batch_idx = min(end_batch_idx, len(self.latents[0]))

        latents_batch = [latents[start_batch_idx:end_batch_idx] for latents in self.latents]
        observations_batch = [observations[start_batch_idx:end_batch_idx] for observations in self.observations]

        return [latents_batch, observations_batch]

    def __len__(self):
        return len(self.latents[0]) // self.batch_size

def get_synthetic_dataloader_online_position(initial, transition, emission, num_timesteps,
                             batch_size, state, observations, total_timesteps=10000):
    return torch.utils.data.DataLoader(
        SyntheticDataset_online_position(initial, transition, emission, num_timesteps,
                         batch_size, state, observations, total_timesteps),
        batch_size=1,
        collate_fn=lambda x: x[0])

def get_synthetic_dataloader_offline_position(initial, transition, emission, num_timesteps,
                             batch_size, state, observations):
    return torch.utils.data.DataLoader(
        SyntheticDataset_offline_position(initial, transition, emission, num_timesteps,
                         batch_size, state, observations),
        batch_size=1,
        collate_fn=lambda x: x[0])

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


