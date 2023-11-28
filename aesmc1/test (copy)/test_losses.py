import os
import aesmc.train as train
import aesmc.losses as losses
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
import unittest
from aesmc.arguments import parse_args
from varname import nameof

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main(args):
    setup_seed(5)
    Models = TestModels()
    # Models.test_gaussian(args)
    Models.test_lgssm(args)

def save_data(data_list, name_list, algorithm, lr, num_exps, num_iterations, dim, labelled_ratio, trainType):
    dir_name = f'./test_autoencoder_plots_{algorithm}_{lr}_{num_exps}_{num_iterations}_{dim}_{trainType}_{labelled_ratio}/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i, data in enumerate(data_list):
        # np.save('./test_autoencoder_plots_{}_{}_{}_{}_{}_{}/{}_{}_{}_{}_{}_{}.npy'.format(algorithm, name_list[i], lr, num_exps, num_iterations, dim, algorithm, name_list[i], lr, num_exps, num_iterations, dim) ,data)
        file_name = f'{algorithm}_{name_list[i]}_{lr}_{num_exps}_{num_iterations}_{dim}.npy'
        full_path = os.path.join(dir_name, file_name)
        np.save(full_path, data)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TestModels(unittest.TestCase):
    def test_gaussian(self,args):
        device = args.device
        from models import gaussian

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
        from models import lgssm
        print('\nTraining the \"linear Gaussian state space model\"'
              ' autoencoder.')
        dim = 1

        initial_loc = torch.zeros([dim]).to(device).squeeze()
        initial_scale = torch.eye(dim).to(device).squeeze()

        # if dim > 1:
        #     true_transition_mult = torch.ones([dim, dim]).to(device).squeeze()
        #     init_transition_mult = (0.1 * torch.ones([dim, dim])).to(device).squeeze()
        #     for i in range(dim):
        #         for j in range(dim):
        #             true_transition_mult [i,j] = 0.42**(abs(i-j)+1)
        #             init_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
        # else:
        #     true_transition_mult = (0.9 * torch.ones(dim)).to(device).squeeze()
        #     init_transition_mult = (0.1 * torch.ones(dim)).to(device).squeeze()
        #
        # transition_scale =  torch.eye(dim).to(device).squeeze()
        # true_emission_mult = (0.5*torch.ones(dim)).to(device).squeeze()
        # init_emission_mult = (1.0*torch.ones(dim)).to(device).squeeze()
        #
        # init_proposal_scale_0 = 0.1*torch.ones(dim).to(device)#(0.1*torch.eye(dim)).to(device).squeeze()
        # init_proposal_scale_t = 0.1*torch.ones(dim).to(device)#(0.1*torch.eye(dim)).to(device).squeeze()
        if dim > 1:
            true_transition_mult = torch.ones([dim, dim]).to(device).squeeze()
            init_transition_mult = (torch.ones([dim, dim])).to(device).squeeze()
            for i in range(dim):
                for j in range(dim):
                    true_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
                    # init_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
        else:
            true_transition_mult = (0.9 * torch.ones(dim)).to(device).squeeze()
            true_transition_mult_online = (0.3 * torch.ones(dim)).to(device).squeeze()
            init_transition_mult = (1.0 * torch.ones(dim)).to(device).squeeze()

        transition_scale = torch.eye(dim).to(device).squeeze()
        true_emission_mult = (0.5 * torch.ones(dim)).to(device).squeeze()
        true_emission_mult_online = (0.8 * torch.ones(dim)).to(device).squeeze()
        init_emission_mult = (1.0 * torch.ones(dim)).to(device).squeeze()

        init_proposal_scale_0 = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()
        init_proposal_scale_t = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()

        emission_scale = ((0.1**0.5)* torch.eye(dim)).to(device).squeeze()

        num_timesteps = 51
        num_timesteps_online = 50
        num_test_obs = 10
        test_inference_num_particles = 100
        saving_interval = 10
        logging_interval = 10
        batch_size = 10
        batch_size_online = 1
        num_iterations = 500
        num_particles = 100
        num_experiments = 1
        num_of_flows = []
        labelled_ratio = 0.01
        flow_types = ['planar']
        lr = 0.02
        # http://tuananhle.co.uk/notes/optimal-proposal-lgssm.html
        Gamma_0 = true_emission_mult * initial_scale ** 2 / (emission_scale ** 2 + initial_scale ** 2 * true_emission_mult ** 2)
        optimal_proposal_scale_0 = torch.sqrt(initial_scale**2 - initial_scale**2 * true_emission_mult * Gamma_0)

        Gamma_t = true_emission_mult * transition_scale**2 / (emission_scale**2 + transition_scale**2 * true_emission_mult**2)
        optimal_proposal_scale_t = torch.sqrt(transition_scale**2 - transition_scale**2 * true_emission_mult * Gamma_t)

        algorithms =  ['cnf-dpf-'+flow_type+'-'+str(num_of_flows[i]) for i in range(len(num_of_flows))
                       for flow_type in flow_types] +['aesmc']#['bootstrap']# ['pfrnn']#
        colors = {'aesmc': 'red',
                  'bootstrap':'green',
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
        dataloader = train.get_synthetic_dataloader(
            lgssm.Initial(initial_loc, initial_scale).to(device),
            lgssm.Transition(true_transition_mult, transition_scale).to(device),
            lgssm.Emission(true_emission_mult, emission_scale).to(device),
            num_timesteps, batch_size)

        dataloader_online = train.get_synthetic_dataloader_online(
            lgssm.Initial(initial_loc, initial_scale).to(device),
            lgssm.Transition(true_transition_mult_online, transition_scale).to(device),
            lgssm.Emission(true_emission_mult_online, emission_scale).to(device),
            num_timesteps_online, batch_size_online)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(15, 10)

        for algorithm in algorithms:
            parameter_error_recorder, posterior_error_recorder, elbo_recorder, ESS_recorder = [], [], [], []
            for i in range(num_experiments):
                training_stats = lgssm.TrainingStats(
                    initial_loc, initial_scale, true_transition_mult,
                    transition_scale, true_emission_mult, emission_scale,
                    num_timesteps, num_test_obs, test_inference_num_particles,
                    saving_interval, logging_interval,algorithm=algorithm, args = args, num_iterations=num_iterations)
                Initial_dist = lgssm.Initial(initial_loc, initial_scale).to(device)
                Transition_dist = lgssm.Transition(init_transition_mult,
                                                        transition_scale).to(device)
                Emission_dist = lgssm.Emission(init_emission_mult,
                                                    emission_scale).to(device)
                markers = {'aesmc': 'X',
                           'bootstrap': 'p',
                           'cnf-dpf-1': '*',
                           'cnf-dpf-nvp-2': '^',
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
                train.train(dataloader_online=dataloader_online,
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
                            optimizer_algorithm=torch.optim.AdamW,
                            optimizer_kwargs={'lr': lr},
                            callback=training_stats,
                            args=args,
                            )
                print('Learning rate:', lr)
                parameter_error_recorder.append(training_stats.p_l2_history)
                posterior_error_recorder.append(training_stats.q_l2_history)
                elbo_recorder.append(np.array(training_stats.loss_history).mean(-1))
                ESS_recorder.append(1/((np.array(training_stats.normalized_log_weights_history)**2).sum(-1)).mean(-1))
                print('Exp. {}/{}, {}, parameter error:{:.6f}+-{:.6f}, posterrior_error:{:.3f}+-{:.3f}, ELBO:{:.6f}+-{:.6f}, ESS:{:.6f}+-{:.6f}'
                      .format(i+1, num_experiments, algorithm,
                              np.array(parameter_error_recorder).mean(0)[-1],
                              np.array(parameter_error_recorder).std(0)[-1],
                              np.array(posterior_error_recorder).mean(0)[-1],
                              np.array(posterior_error_recorder).std(0)[-1],
                              np.array(elbo_recorder).mean(0)[-1],
                              np.array(elbo_recorder).std(0)[-1],
                              np.array(ESS_recorder).mean(-1)[:,-1].mean(),
                              np.array(ESS_recorder).mean(-1)[:,-1].std()) )

            parameter_error_recorder = np.array(parameter_error_recorder)
            posterior_error_recorder = np.array(posterior_error_recorder)
            elbo_recorder = np.array(elbo_recorder)
            ESS_recorder = np.array(ESS_recorder)

            training_stats.iteration_idx_history = np.array(training_stats.iteration_idx_history)+1
            axs[0,0].plot(training_stats.iteration_idx_history,
                          parameter_error_recorder.mean(0),
                          label=algorithm,
                          color=colors[algorithm])
            axs[0,0].scatter(training_stats.iteration_idx_history[::5],
                             parameter_error_recorder.mean(0)[::5],
                             color=colors[algorithm],
                             marker=markers[algorithm],
                             s=marker_size)
            axs[0,0].fill_between(training_stats.iteration_idx_history,
                                parameter_error_recorder.mean(0) - parameter_error_recorder.std(0)*0.3,
                                parameter_error_recorder.mean(0) + parameter_error_recorder.std(0)*0.3,
                                color=colors[algorithm],
                                alpha=0.3)
            axs[0,1].plot(training_stats.iteration_idx_history,
                          posterior_error_recorder.mean(0),
                          label=algorithm,
                          color=colors[algorithm])
            axs[0,1].scatter(training_stats.iteration_idx_history[::5],
                             posterior_error_recorder.mean(0)[::5],
                             color=colors[algorithm],
                             marker=markers[algorithm],
                             s=marker_size)
            axs[0,1].fill_between(training_stats.iteration_idx_history,
                                posterior_error_recorder.mean(0) - posterior_error_recorder.std(0)*0.3,
                                posterior_error_recorder.mean(0) + posterior_error_recorder.std(0)*0.3,
                                color=colors[algorithm],
                                alpha=0.3)
            axs[1,0].plot(training_stats.iteration_idx_history,
                          elbo_recorder.mean(0),
                          label=algorithm,
                          color=colors[algorithm])
            axs[1,0].scatter(training_stats.iteration_idx_history[::5],
                             elbo_recorder.mean(0)[::5],
                             color=colors[algorithm],
                             marker=markers[algorithm],
                             s=marker_size)
            axs[1,0].fill_between(training_stats.iteration_idx_history,
                                elbo_recorder.mean(0) - elbo_recorder.std(0)*0.3,
                                elbo_recorder.mean(0) + elbo_recorder.std(0)*0.3,
                                color=colors[algorithm],
                                alpha=0.3)
            axs[1,1].plot(np.arange(num_timesteps),
                          ESS_recorder[:, -1].mean(0),
                          label=algorithm,
                          color=colors[algorithm])
            axs[1,1].scatter(np.arange(num_timesteps)[::5],
                             ESS_recorder[:, -1].mean(0)[::5],
                             color=colors[algorithm],
                             marker=markers[algorithm],
                             s=marker_size)
            axs[1,1].fill_between(np.arange(num_timesteps),
                                ESS_recorder[:,-1].mean(0) - ESS_recorder[:,-1].std(0)*0.3,
                                ESS_recorder[:,-1].mean(0) + ESS_recorder[:,-1].std(0)*0.3,
                                color=colors[algorithm],
                                alpha=0.3)
            data_list = [parameter_error_recorder,
                         posterior_error_recorder,
                         elbo_recorder,
                         ESS_recorder]
            data_name_list = ['parameter_error_recorder',
                              'posterior_error_recorder',
                              'elbo_recorder',
                              'ESS_recorder']
            save_data(data_list, data_name_list, algorithm, lr, num_experiments, num_iterations, dim, args.labelled_ratio, args.trainType)

            loss_rmse_test = training_stats(-1, -1, -1, initial=Initial_dist,
                                                                  transition=Transition_dist, emission=Emission_dist,
                                                                  proposal=proposal, test=True)
            data_list_test = [
                              loss_rmse_test]
            data_name_list_test = [
                                   'loss_rmse_test']
            save_data(data_list_test, data_name_list_test, algorithm, lr, num_experiments, num_iterations, dim,
                      args.labelled_ratio,
                      args.trainType)

        axs[0,0].set_ylabel('$||\\theta - \\theta_{true}||$')
        axs[0, 0].set_xticks([1]+[i for i in np.arange((num_iterations+1)//5, num_iterations+1, (num_iterations+1)//5)])
        axs[0,1].set_ylabel('Avg. L2 of\nmarginal posterior means')
        axs[0, 1].set_xticks([1] + [i for i in np.arange((num_iterations+1)//5, num_iterations + 1, (num_iterations+1)//5)])
        axs[1,0].set_ylabel('ELBO')
        axs[1, 0].set_xticks([1] + [i for i in np.arange((num_iterations+1)//5, num_iterations + 1, (num_iterations+1)//5)])
        axs[1,1].set_ylabel('ESS')
        axs[0,0].set_xlabel('Iteration')
        axs[0,1].set_xlabel('Iteration')
        axs[1,0].set_xlabel('Iteration')
        axs[1,1].set_xlabel('Step')
        axs[0,0].legend()

        for ax in axs:
            for x in ax:
                x.grid(alpha=0.5)
        fig.tight_layout()
        filename = './test_autoencoder_plots/lgssm_elbo.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))


        self.assertTrue(True)


if __name__ == '__main__':
    args = parse_args()
    # print(os.getcwd())
    main(args)
