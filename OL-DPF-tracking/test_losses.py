import os
import train
import losses
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
import unittest
from arguments import parse_args
# from varname import nameof

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main(args):
    setup_seed(5)
    Models = TestModels()
    # Models.test_gaussian(args)
    Models.test_lgssm(args)

def save_data(num_time_online, data_list, name_list, algorithm, lr, num_exps, num_iterations, dim, labelled_ratio, trainType):
    dir_name = f'./test_autoencoder_plots_{algorithm}_{lr}_{num_exps}_{num_iterations}_{dim}_{trainType}_{labelled_ratio}_{num_time_online}/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i, data in enumerate(data_list):
        # np.save('./test_autoencoder_plots_{}_{}_{}_{}_{}_{}/{}_{}_{}_{}_{}_{}.npy'.format(algorithm, name_list[i], lr, num_exps, num_iterations, dim, algorithm, name_list[i], lr, num_exps, num_iterations, dim) ,data)
        file_name = f'{algorithm}_{name_list[i]}_{lr}_{num_exps}_{num_iterations}_{dim}_{num_time_online}.npy'
        full_path = os.path.join(dir_name, file_name)
        np.save(full_path, data)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_reshaped_data(batch_size, sequence_length, ts=1.0, p0=1.0, alpha=2.0, r=np.array([0.0, 0.0]),
                           state_noise_mean=0.0, state_noise_scale=0.1, turn_rate_noise_mean=0.0,
                           turn_rate_noise_scale=0.1, observation_noise_mean=0.0, observation_noise_scale1=0.1, observation_noise_scale2=0.1, a_t=0):
    # Initialize the data containers
    states = np.zeros((batch_size, sequence_length, 5))  # Including omega_t
    observations = np.zeros((batch_size, sequence_length, 2))  # Two observation components

    for batch in range(batch_size):
        # Initial state
        x_tilde = np.array([0.0, 0.0, 55/np.sqrt(2), 55/np.sqrt(2)])  # Initial position and velocity
        omega_t = a_t / np.linalg.norm(x_tilde[2:4]) + np.random.normal(turn_rate_noise_mean, turn_rate_noise_scale)  # Initial turn rate
        # Store the initial state
        states[batch, 0, :4] = x_tilde
        states[batch, 0, 4] = omega_t

        # Initial observation
        p_t = x_tilde[:2]
        h1 = 10 * np.log10(p0 / np.linalg.norm(r - p_t) ** alpha)
        h2 = np.arctan2(p_t[1] - r[1], p_t[0] - r[0])
        observations[batch, 0, :] = np.array([h1, h2]) + 0.7 * np.random.normal(observation_noise_mean,
                                                                                observation_noise_scale1,
                                                                                2) + 0.3 * np.random.normal(
            observation_noise_mean, observation_noise_scale2, 2)

        for t in range(1,sequence_length):

            # Update omega_t based on the model
            if np.linalg.norm(x_tilde[2:4]) != 0:
                omega_t = a_t / np.linalg.norm(x_tilde[2:4]) + np.random.normal(turn_rate_noise_mean, turn_rate_noise_scale)

            # Define A and B matrices
            A = np.array([
                [1, 0, np.sin(omega_t * ts) / omega_t if omega_t != 0 else ts,
                 -(1 - np.cos(omega_t * ts)) / omega_t if omega_t != 0 else 0],
                [0, 1, -(1 - np.cos(omega_t * ts)) / omega_t if omega_t != 0 else 0,
                 np.sin(omega_t * ts) / omega_t if omega_t != 0 else ts],
                [0, 0, np.cos(omega_t * ts), -np.sin(omega_t * ts)],
                [0, 0, np.sin(omega_t * ts), np.cos(omega_t * ts)]
            ])
            B = np.array([[ts**2 / 2, 0], [0, ts**2 / 2], [ts, 0], [0, ts]])

            # Update state x_tilde
            x_tilde = A @ x_tilde + B @ np.random.normal(state_noise_mean, state_noise_scale, 2)

            # Measurement model
            p_t = x_tilde[:2]
            h1 = 10 * np.log10(p0 / np.linalg.norm(r - p_t)**alpha)
            h2 = np.arctan2(p_t[1] - r[1], p_t[0] - r[0])

            # Store the state and observation
            states[batch, t, :4] = x_tilde
            states[batch, t, 4] = omega_t
            observations[batch, t, :] = np.array([h1, h2]) + 0.7*np.random.normal(observation_noise_mean, observation_noise_scale1, 2) + 0.3*np.random.normal(observation_noise_mean, observation_noise_scale2, 2)

    # Reshape data into lists of numpy arrays
    state_list = [states[:, t, :] for t in range(sequence_length)]
    observation_list = [observations[:, t, :] for t in range(sequence_length)]
    # Return data as a tuple
    return (state_list, observation_list)

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

class TestModels(unittest.TestCase):
    def test_gaussian(self,args):
        device = args.device
        import gaussian

        prior_std = 1

        true_prior_mean = 0
        true_obs_std = 1

        prior_mean_init = 2
        obs_std_init = 0.5

        q_init_mult, q_init_bias, q_init_std = 2, 2, 2
        q_true_mult, q_true_bias, q_true_std = gaussian.get_proposal_params(
            true_prior_mean, prior_std, true_obs_std)

        true_prior = gaussian.Prior(true_prior_mean, prior_std).to(device)
        true_likelihood = gaussian.Likelihood(true_obs_std).to(device)

        num_particles = 2
        batch_size = 10
        num_iterations = 2000

        training_stats = gaussian.TrainingStats(logging_interval=500)

        print('\nTraining the \"gaussian\" autoencoder.')
        prior = gaussian.Prior(prior_mean_init, prior_std).to(device)
        likelihood = gaussian.Likelihood(obs_std_init).to(device)
        inference_network = gaussian.InferenceNetwork(
            q_init_mult, q_init_bias, q_init_std).to(device)
        train.train(dataloader=train.get_synthetic_dataloader(
                        true_prior, None, true_likelihood, 1, batch_size),
                    num_particles=num_particles,
                    algorithm='iwae',
                    initial=prior,
                    transition=None,
                    emission=likelihood,
                    proposal=inference_network,
                    num_epochs=1,
                    num_iterations_per_epoch=num_iterations,
                    optimizer_algorithm=torch.optim.SGD,
                    optimizer_kwargs={'lr': 0.01},
                    callback=training_stats, args = args)

        fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
        fig.set_size_inches(10, 8)

        mean = training_stats.prior_mean_history
        obs = training_stats.obs_std_history
        mult = training_stats.q_mult_history
        bias = training_stats.q_bias_history
        std = training_stats.q_std_history
        data = [mean] + [obs] + [mult] + [bias] + [std]
        true = [true_prior_mean, true_obs_std, q_true_mult, q_true_bias,
                q_true_std]

        for ax, data_, true_, ylabel in zip(
            axs, data, true, ['$\mu_0$', '$\sigma$', '$a$', '$b$', '$c$']
        ):
            ax.plot(training_stats.iteration_idx_history, data_)
            ax.axhline(true_, color='black')
            ax.set_ylabel(ylabel)
            #  self.assertAlmostEqual(data[-1], true, delta=1e-1)

        axs[-1].set_xlabel('Iteration')
        fig.tight_layout()

        filename = './test/test_autoencoder_plots/gaussian.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

    def test_lgssm(self,args):
        device = args.device
        print('trainType:', args.trainType, 'labelled_ratio:', args.labelled_ratio)
        import lgssm
        print('\nTraining the \"object tracking model\"'
              ' autoencoder.')
        dim = 5
        obs_dim = 2

        initial_loc = torch.zeros([dim]).to(device).squeeze()
        initial_scale = torch.eye(dim).to(device).squeeze()
        if dim > 1:
            true_transition_mult = torch.eye(dim).to(device).squeeze()#torch.ones([dim, dim]).to(device).squeeze()
            true_transition_mult_online1 = torch.eye(dim).to(device).squeeze()#torch.ones([dim, dim]).to(device).squeeze()
            true_transition_mult_online2 = torch.ones([dim, dim]).to(device).squeeze()
            init_transition_mult = torch.diag(1.0 * torch.ones([dim])).to(device).squeeze()
            # for i in range(dim):
            #     for j in range(dim):
            #         true_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
            #         true_transition_mult_online1[i, j] = 0.2 ** (abs(i - j) + 1)
            #         true_transition_mult_online2[i, j] = 0.40 ** (abs(i - j) + 1)

                    # init_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
        else:
            true_transition_mult = (torch.ones(dim)).to(device).squeeze()
            true_transition_mult_online1 = (torch.ones(dim)).to(device).squeeze()
            true_transition_mult_online2 = (0.7 * torch.ones(dim)).to(device).squeeze()
            init_transition_mult = (1.0 * torch.ones(dim)).to(device).squeeze()

        transition_scale = torch.eye(dim).to(device).squeeze()
        true_emission_mult = (torch.ones(dim)).to(device).squeeze()
        true_emission_mult_online1 = (torch.ones(dim)).to(device).squeeze()
        true_emission_mult_online2 = (0.7 * torch.ones(dim)).to(device).squeeze()
        init_emission_mult = (1.0 * torch.ones(dim)).to(device).squeeze()

        init_proposal_scale_0 = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()
        init_proposal_scale_t = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()

        # emission_scale = ((0.1**0.5)* torch.eye(dim)).to(device).squeeze()
        emission_scale = (torch.eye(dim)).to(device).squeeze()

        num_timesteps = 50
        num_timesteps_online = 10
        num_iterations_online = 500
        total_timestep_online = num_timesteps_online * num_iterations_online
        num_test_obs = 10
        test_inference_num_particles = 100
        saving_interval = 10
        logging_interval = 10
        batch_size = 10
        batch_size_online = 1
        num_iterations = 50

        num_particles = 100
        num_experiments = 50
        num_of_flows = [1]
        labelled_ratio = 0.01
        flow_types = ['nvp']
        lr = 0.005
        if args.NF_cond:
            algorithms =  ['cnf-dpf-'+flow_type+'-'+str(num_of_flows[i]) for i in range(len(num_of_flows)) for flow_type in flow_types] #+['aesmc']#['bootstrap']# ['pfrnn']#
        else:
            algorithms = ['aesmc']
        colors = {'aesmc': 'red',
                  'bootstrap':'green',
                  'cnf-dpf-nvp-0': 'pink',
                  'cnf-dpf-nvp-1': 'blue',
                  'cnf-dpf-nvp-2': 'purple',
                  'cnf-dpf-nvp-4': 'orange',
                  'cnf-dpf-nvp-8': 'cyan',
                  'cnf-dpf-nvp-25': 'black',
                  'cnf-dpf-planar-1': 'blue',
                  'cnf-dpf-planar-2': 'purple',
                  'cnf-dpf-planar-4': 'orange',
                  'cnf-dpf-planar-8': 'cyan',
                  'cnf-dpf-planar-25': 'black',
                  'pfrnn': 'fuchsia',
                  }


        dataloader_online2 = None
        # dataloader_online2 = train.get_synthetic_dataloader_online(
        #     lgssm.Initial(initial_loc, initial_scale).to(device),
        #     lgssm.Transition(true_transition_mult_online2, transition_scale).to(device),
        #     lgssm.Emission(true_emission_mult_online2, emission_scale).to(device),
        #     num_timesteps_online, batch_size_online, total_timesteps=total_timestep_online)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(15, 10)

        # data_offline = np.load('offline_data.npz', allow_pickle=True)
        # state_offline = data_offline['states']
        # observation_offline = data_offline['observations']

        # data_online = np.load('online_data.npz', allow_pickle=True)
        # state_online = data_online['states']
        # observation_online = data_online['observations']


        for algorithm in algorithms:
            parameter_error_recorder, posterior_error_recorder, ESS_recorder = [], [], []
            rmse_recorder, elbo_recorder, all_loss_recorder = [],[],[]
            for i in range(num_experiments):
                setup_seed(i)

                state_offline, observation_offline = generate_reshaped_data(batch_size=500, sequence_length=50, ts=5.0,
                                                                            p0=1.0, alpha=2.0, r=np.array([2.0, 2.0]),
                                                                            state_noise_mean=0.0,
                                                                            state_noise_scale=0.01,
                                                                            turn_rate_noise_mean=0.0,
                                                                            turn_rate_noise_scale=0.0001,
                                                                            observation_noise_mean=0.0,
                                                                            observation_noise_scale1=0.4,
                                                                            observation_noise_scale2=0.25, a_t=5)

                min_values, max_values = find_min_max_in_states(state_offline)
                initial_particles = sample_initial_particles(min_values, max_values, batch_size, num_particles,
                                                             dim)
                initial_particles = to_gpu_tensor(initial_particles)

                # state_offline, observation_offline = np.array(state_offline), np.array(observation_offline)
                state_offline_tensors = [torch.tensor(s).float().to('cuda') for s in state_offline]
                observation_offline_tensors = [torch.tensor(o).float().to('cuda') for o in observation_offline]

                state_online, observation_online = generate_reshaped_data(batch_size=1, sequence_length=5000, ts=5.0,
                                                                          p0=1.0,
                                                                          alpha=2.0, r=np.array([2.0, 2.0]),
                                                                          state_noise_mean=0.0, state_noise_scale=0.01,
                                                                          turn_rate_noise_mean=0.0,
                                                                          turn_rate_noise_scale=0.0001,
                                                                          observation_noise_mean=0.0,
                                                                          observation_noise_scale1=0.4,
                                                                          observation_noise_scale2=0.25, a_t=-5)
                state_online_tensors = [torch.tensor(s).float().to('cuda') for s in state_online]
                observation_online_tensors = [torch.tensor(o).float().to('cuda') for o in observation_online]

                dataloader = train.get_synthetic_dataloader_offline_position(
                    lgssm.Initial(initial_loc, initial_scale).to(device),
                    lgssm.Transition(true_transition_mult, transition_scale).to(device),
                    lgssm.Emission(true_emission_mult, emission_scale).to(device),
                    num_timesteps, batch_size, state_offline_tensors, observation_offline_tensors)

                dataloader_online1 = train.get_synthetic_dataloader_online_position(
                    lgssm.Initial(initial_loc, initial_scale).to(device),
                    lgssm.Transition(true_transition_mult_online1, transition_scale).to(device),
                    lgssm.Emission(true_emission_mult_online1, emission_scale).to(device),
                    num_timesteps_online, batch_size_online, state_online_tensors, observation_online_tensors, total_timesteps=total_timestep_online)

                training_stats = lgssm.TrainingStats(
                    true_transition_mult_online1, true_emission_mult_online1, true_transition_mult_online2, true_emission_mult_online2, initial_loc, initial_scale, true_transition_mult,
                    transition_scale, true_emission_mult, emission_scale,
                    num_timesteps, num_test_obs, test_inference_num_particles,
                    saving_interval, logging_interval,algorithm=algorithm, args = args, num_iterations=num_iterations, dataloader=dataloader)
                Initial_dist = lgssm.Initial(initial_loc, initial_scale).to(device)
                if args.NF_dyn:
                    n_sequence, hidden_size, init_var = 1, dim, 0.01
                    dyn_nf = lgssm.build_dyn_nf(n_sequence, hidden_size, dim, init_var = init_var, translate=True, type=flow_types[0])
                    prototype_transition = lgssm.Transition(torch.ones_like(init_transition_mult).detach().clone(),
                                                            transition_scale.detach().clone()).to(device)
                    Transition_dist = lgssm.Dynamic_cnf(dyn_nf=dyn_nf,prototype_transition=prototype_transition,
                                                        dim=dim,type='nvp', n_sequence=n_sequence,
                                                        hidden_size=hidden_size,init_var=init_var)
                else:
                    Transition_dist = lgssm.Transition(init_transition_mult, transition_scale).to(device)
                if args.measurement == 'CRNVP':
                    n_sequence, hidden_size = 1, 10
                    particle_encoder = lgssm.build_particle_encoder_maze(hidden_size, dim).to(device)  #nn.Identity() #
                    obs_encoder = lgssm.build_encoder_maze(hidden_size, 0.3, obs_dim).to(device) #nn.Identity()#
                    type = 'nf'
                    if type == 'nf':
                        cnf_measurement = lgssm.build_dyn_nf(n_sequence, hidden_size, hidden_size, init_var=0.2, type=flow_types[0]).to(device)#
                    else:
                        cnf_measurement = lgssm.build_conditional_nf(n_sequence, hidden_size, hidden_size, init_var=0.01)  #
                    Emission_dist = lgssm.measurement_model_cnf(particle_encoder, obs_encoder, cnf_measurement, type=type)
                else:
                    Emission_dist = lgssm.Emission(init_emission_mult,
                                                        emission_scale).to(device)
                markers = {'aesmc': 'X',
                           'bootstrap': 'p',
                           'cnf-dpf-nvp-0': 'd',
                           'cnf-dpf-1': '*',
                           'cnf-dpf-nvp-1': '^',
                           'cnf-dpf-nvp-4': '+',
                           'cnf-dpf-nvp-8': '2',
                           'cnf-dpf-nvp-25': '>',
                           'cnf-dpf-planar-1': '*',
                           'cnf-dpf-planar-2': '^',
                           'cnf-dpf-planar-4': '+',
                           'cnf-dpf-planar-8': '2',
                           'cnf-dpf-planar-25': '>',
                           'pfrnn': 'p'
                           }
                marker_size = 200
                if algorithm == 'aesmc':
                    proposal = lgssm.Proposal(init_proposal_scale_0, init_proposal_scale_t, device).to(device)
                elif algorithm == 'pfrnn':
                    proposal = lgssm.Proposal_rnn(init_proposal_scale_0, init_proposal_scale_t, device).to(device)
                elif algorithm.startswith('cnf-dpf'):
                    if 'planar' in algorithm:
                        proposal = lgssm.Proposal_cnf(initial=Initial_dist,
                                                      transition=Transition_dist,
                                                      scale_0=init_proposal_scale_0,
                                                      scale_t=init_proposal_scale_t,
                                                      device=device,
                                                      k=int(algorithm.split('-')[-1]),
                                                      type='planar').to(device)
                    elif 'nvp' in algorithm:
                        proposal = lgssm.Proposal_cnf(initial=Initial_dist,
                                                      transition=Transition_dist,
                                                      scale_0=init_proposal_scale_0,
                                                      scale_t=init_proposal_scale_t,
                                                      device=device,
                                                      k=int(algorithm.split('-')[-1]),
                                                      type='nvp').to(device)
                elif algorithm == 'bootstrap':
                    proposal = lgssm.Proposal_cnf(initial=Initial_dist,
                                                  transition=Transition_dist,
                                                  scale_0=init_proposal_scale_0,
                                                  scale_t=init_proposal_scale_t,
                                                  device=device,
                                                  type='bootstrap').to(device)
                else:
                    raise ValueError('Please select an algorithm from {aesmc, cnf-dpf, bootstrap}.')
                rmse_plot, elbo_plot, rmse_box_plot = train.train(initial_state=initial_particles,
                            dataloader_online1=dataloader_online1,
                            dataloader_online2=dataloader_online2,
                            dataloader=dataloader,
                            num_particles=num_particles,
                            algorithm=algorithm,
                            initial=Initial_dist,
                            transition=Transition_dist,
                            emission=Emission_dist,
                            # proposal=lgssm.Proposal(optimal_proposal_scale_0,optimal_proposal_scale_t, device).to(device),
                            proposal = proposal,
                            num_epochs=1,
                            num_iterations_per_epoch=num_iterations,
                            num_iterations_per_epoch_online=num_iterations_online,
                            optimizer_algorithm=torch.optim.AdamW,
                            optimizer_kwargs={'lr': lr},
                            callback=training_stats,
                            args=args,
                            )
                print('Learning rate:', lr)
                rmse_temp=np.array(rmse_plot)
                elbo_temp=np.array(elbo_plot)
                rmse_box_temp=np.array(rmse_box_plot)
                rmse_recorder.append(rmse_temp)
                elbo_recorder.append(elbo_temp)
                all_loss_recorder.append(rmse_box_temp)
                print('No. Experiment:{}/{}, onlineType: {}, rmse:{:.2f}+-{:.2f}, elbo:{:.2f}+-{:.2f}, rmse_last_slice:{:.2f}, elbo_last_slice:{:.2f}, '
                      .format(i+1, num_experiments,
                              args.trainType,
                              rmse_temp.mean(),
                              rmse_temp.std(),
                              elbo_temp.mean(),
                              elbo_temp.std(),
                              rmse_temp[-1],
                              elbo_temp[-1]))

            rmse_recorder = np.array(rmse_recorder)
            elbo_recorder = np.array(elbo_recorder)
            all_loss_recorder = np.array(all_loss_recorder)

            folder_name = f"dim_{dim}_lr_{lr}_type_{args.trainType}"
            folder_path = os.path.join("logs", folder_name)
            os.makedirs(folder_path, exist_ok=True)
            np.savez(os.path.join(folder_path, "results.npz"),
                     rmse_recorder=rmse_recorder,
                     elbo_recorder=elbo_recorder,
                     all_loss_recorder=all_loss_recorder)

            # training_stats.iteration_idx_history = np.array(training_stats.iteration_idx_history)+1
            # axs[0,0].plot(training_stats.iteration_idx_history,
            #               parameter_error_recorder.mean(0),
            #               label=algorithm,
            #               color=colors[algorithm])
            # axs[0,0].scatter(training_stats.iteration_idx_history[::5],
            #                  parameter_error_recorder.mean(0)[::5],
            #                  color=colors[algorithm],
            #                  marker=markers[algorithm],
            #                  s=marker_size)
            # axs[0,0].fill_between(training_stats.iteration_idx_history,
            #                     parameter_error_recorder.mean(0) - parameter_error_recorder.std(0)*0.3,
            #                     parameter_error_recorder.mean(0) + parameter_error_recorder.std(0)*0.3,
            #                     color=colors[algorithm],
            #                     alpha=0.3)
            # axs[1,0].plot(training_stats.iteration_idx_history,
            #               elbo_recorder.mean(0),
            #               label=algorithm,
            #               color=colors[algorithm])
            # axs[1,0].scatter(training_stats.iteration_idx_history[::5],
            #                  elbo_recorder.mean(0)[::5],
            #                  color=colors[algorithm],
            #                  marker=markers[algorithm],
            #                  s=marker_size)
            # axs[1,0].fill_between(training_stats.iteration_idx_history,
            #                     elbo_recorder.mean(0) - elbo_recorder.std(0)*0.3,
            #                     elbo_recorder.mean(0) + elbo_recorder.std(0)*0.3,
            #                     color=colors[algorithm],
            #                     alpha=0.3)
            # axs[1,1].plot(np.arange(num_timesteps),
            #               ESS_recorder[:, -1].mean(0),
            #               label=algorithm,
            #               color=colors[algorithm])
            # axs[1,1].scatter(np.arange(num_timesteps)[::5],
            #                  ESS_recorder[:, -1].mean(0)[::5],
            #                  color=colors[algorithm],
            #                  marker=markers[algorithm],
            #                  s=marker_size)
            # axs[1,1].fill_between(np.arange(num_timesteps),
            #                     ESS_recorder[:,-1].mean(0) - ESS_recorder[:,-1].std(0)*0.3,
            #                     ESS_recorder[:,-1].mean(0) + ESS_recorder[:,-1].std(0)*0.3,
            #                     color=colors[algorithm],
            #                     alpha=0.3)
        #     data_list = [parameter_error_recorder,
        #                  posterior_error_recorder,
        #                  elbo_recorder,
        #                  ESS_recorder]
        #     data_name_list = ['parameter_error_recorder',
        #                       'posterior_error_recorder',
        #                       'elbo_recorder',
        #                       'ESS_recorder']
        #     save_data(num_timesteps_online, data_list, data_name_list, algorithm, lr, num_experiments, num_iterations, dim, args.labelled_ratio, args.trainType)
        #
        #     loss_rmse_test = training_stats(-1, -1, -1, initial=Initial_dist,
        #                                                           transition=Transition_dist, emission=Emission_dist,
        #                                                           proposal=proposal, test=True, args=args)
        #     data_list_test = [
        #                       loss_rmse_test]
        #     data_name_list_test = [
        #                            'loss_rmse_test']
        #     save_data(num_timesteps_online, data_list_test, data_name_list_test, algorithm, lr, num_experiments, num_iterations, dim,
        #               args.labelled_ratio,
        #               args.trainType)
        #
        # axs[0,0].set_ylabel('$||\\theta - \\theta_{true}||$')
        # axs[0, 0].set_xticks([1]+[i for i in np.arange((num_iterations+1)//5, num_iterations+1, (num_iterations+1)//5)])
        # axs[0,1].set_ylabel('Avg. L2 of\nmarginal posterior means')
        # axs[0, 1].set_xticks([1] + [i for i in np.arange((num_iterations+1)//5, num_iterations + 1, (num_iterations+1)//5)])
        # axs[1,0].set_ylabel('ELBO')
        # axs[1, 0].set_xticks([1] + [i for i in np.arange((num_iterations+1)//5, num_iterations + 1, (num_iterations+1)//5)])
        # axs[1,1].set_ylabel('ESS')
        # axs[0,0].set_xlabel('Iteration')
        # axs[0,1].set_xlabel('Iteration')
        # axs[1,0].set_xlabel('Iteration')
        # axs[1,1].set_xlabel('Step')
        # axs[0,0].legend()
        #
        # for ax in axs:
        #     for x in ax:
        #         x.grid(alpha=0.5)
        # fig.tight_layout()
        # filename = './test_autoencoder_plots/lgssm_elbo.pdf'
        # fig.savefig(filename, bbox_inches='tight')
        # print('\nPlot saved to {}'.format(filename))


        self.assertTrue(True)


if __name__ == '__main__':
    args = parse_args()
    # print(os.getcwd())
    main(args)
