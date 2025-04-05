import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# print(sys.path)
import train
import losses
import statistics
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
import unittest
from arguments import parse_args

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

class TestModels():

    def test_lgssm(self,args):
        device = args.device
        print('trainType:', args.trainType, 'labelled_ratio:', args.labelled_ratio)
        import lgssm
        print('\nTraining the \"linear Gaussian state space model\"'
              ' autoencoder.')
        dim = args.num_dim


        initial_loc = torch.zeros([dim]).to(device).squeeze()
        initial_scale = torch.eye(dim).to(device).squeeze()
        initial_loc_online = 3+torch.zeros([dim]).to(device).squeeze()
        initial_scale_online = 2+torch.eye(dim).to(device).squeeze()
        if dim > 1:
            true_transition_mult = torch.ones([dim, dim]).to(device).squeeze()
            true_transition_mult_online1 = torch.ones([dim, dim]).to(device).squeeze()
            true_transition_mult_online2 = torch.ones([dim, dim]).to(device).squeeze()
            init_transition_mult = torch.diag(1.0 * torch.ones([dim])).to(device).squeeze()
            for i in range(dim):
                for j in range(dim):
                    true_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
                    true_transition_mult_online1[i, j] = 0.2 ** (abs(i - j) + 1)
                    true_transition_mult_online2[i, j] = 0.40 ** (abs(i - j) + 1)

                    # init_transition_mult[i, j] = 0.42 ** (abs(i - j) + 1)
        else:
            true_transition_mult = (0.9 * torch.ones(dim)).to(device).squeeze()
            true_transition_mult_online1 = (0.7 * torch.ones(dim)).to(device).squeeze()
            true_transition_mult_online2 = (0.7 * torch.ones(dim)).to(device).squeeze()
            init_transition_mult = (1.0 * torch.ones(dim)).to(device).squeeze()

        transition_scale = torch.eye(dim).to(device).squeeze()
        true_emission_mult = (10.0 * torch.ones(dim)).to(device).squeeze()
        true_emission_mult_online1 = (10.0 * torch.ones(dim)).to(device).squeeze()
        true_emission_mult_online2 = (0.7 * torch.ones(dim)).to(device).squeeze()
        init_emission_mult = (1.0 * torch.ones(dim)).to(device).squeeze()

        init_proposal_scale_0 = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()
        init_proposal_scale_t = 0.1 * torch.ones(dim).to(device)  # (0.1*torch.eye(dim)).to(device).squeeze()

        emission_scale = ((0.1**0.5)* torch.eye(dim)).to(device).squeeze()

        num_timesteps = 100
        num_timesteps_online = 10
        num_iterations_online = 500
        total_timestep_online = num_timesteps_online * num_iterations_online
        num_test_obs = 10
        test_inference_num_particles = 100
        saving_interval = 10
        logging_interval = 10
        batch_size = 10
        batch_size_online = 1
        num_iterations = 3
        num_iterations_val = 1
        num_iterations_test = 1

        num_particles = 100
        num_experiments = args.num_exp
        num_of_flows = [1]
        labelled_ratio = 0.01
        flow_types = ['nvp']
        lr = 0.002
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

        for algorithm in algorithms:
            parameter_error_recorder, posterior_error_recorder, ESS_recorder = [], [], []
            rmse_recorder, elbo_recorder, all_loss_recorder = [],[],[]
            for i in range(num_experiments):
                setup_seed(i)
                initial_particles, dataloader = train.get_synthetic_dataloader(
                    lgssm.Initial(initial_loc, initial_scale).to(device),
                    lgssm.Transition(true_transition_mult, transition_scale).to(device),
                    lgssm.Emission(true_emission_mult, emission_scale).to(device),
                    num_timesteps, batch_size, num_iterations, num_particles, dim)

                _, dataloader_val = train.get_synthetic_dataloader(
                    lgssm.Initial(initial_loc, initial_scale).to(device),
                    lgssm.Transition(true_transition_mult, transition_scale).to(device),
                    lgssm.Emission(true_emission_mult, emission_scale).to(device),
                    num_timesteps, batch_size, num_iterations_val, num_particles, dim)

                _, dataloader_test = train.get_synthetic_dataloader(
                    lgssm.Initial(initial_loc, initial_scale).to(device),
                    lgssm.Transition(true_transition_mult, transition_scale).to(device),
                    lgssm.Emission(true_emission_mult, emission_scale).to(device),
                    num_timesteps, batch_size, num_iterations_test, num_particles, dim)

                # dataloader_online1 = train.get_synthetic_dataloader_online(
                #     lgssm.Initial(initial_loc_online, initial_scale_online).to(device),
                #     lgssm.Transition(true_transition_mult_online1, transition_scale).to(device),
                #     lgssm.Emission(true_emission_mult_online1, emission_scale).to(device),
                #     num_timesteps_online, batch_size_online, total_timesteps=total_timestep_online)
                training_stats = None
                    # lgssm.TrainingStats(
                    # true_transition_mult_online1, true_emission_mult_online1, true_transition_mult_online2, true_emission_mult_online2, initial_loc, initial_scale, true_transition_mult,
                    # transition_scale, true_emission_mult, emission_scale,
                    # num_timesteps, num_test_obs, test_inference_num_particles,
                    # saving_interval, logging_interval,algorithm=algorithm, args = args, num_iterations=num_iterations)
                Initial_dist = lgssm.Initial(initial_loc, initial_scale).to(device)
                if args.NF_dyn:
                    n_sequence, hidden_size, init_var = 1, dim, 0.01
                    dyn_nf = lgssm.build_dyn_nf(n_sequence, hidden_size, dim, init_var = init_var, translate=True)
                    prototype_transition = lgssm.Transition(torch.ones_like(init_transition_mult).detach().clone(),
                                                            transition_scale.detach().clone()).to(device)
                    Transition_dist = lgssm.Dynamic_cnf(dyn_nf=dyn_nf,prototype_transition=prototype_transition,
                                                        dim=dim,type='nvp', n_sequence=n_sequence,
                                                        hidden_size=hidden_size,init_var=init_var)
                else:
                    Transition_dist = lgssm.Transition(init_transition_mult, transition_scale).to(device)
                if args.measurement == 'CRNVP':
                    n_sequence, hidden_size = 1, dim
                    particle_encoder = nn.Identity() #lgssm.build_particle_encoder_maze(hidden_size, dim).to(device)  #
                    obs_encoder = nn.Identity()#lgssm.build_encoder_maze(dim, 0.3).to(device) #
                    type = 'nf'
                    if type == 'nf':
                        cnf_measurement = lgssm.build_dyn_nf(n_sequence, hidden_size, hidden_size, init_var=0.2).to(device)#
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
                rmse_plot, elbo_plot, rmse_box_plot = train.train(initial_state=initial_particles.detach(),
                            dataloader_val=dataloader_val,
                            dataloader_test=dataloader_test,
                            dataloader=dataloader,
                            num_particles=num_particles,
                            algorithm=algorithm,
                            initial=Initial_dist,
                            transition=Transition_dist,
                            emission=Emission_dist,
                            # proposal=lgssm.Proposal(optimal_proposal_scale_0,optimal_proposal_scale_t, device).to(device),
                            proposal = proposal,
                            num_epochs=args.num_epochs,
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

            folder_name = f"linear_{dim}_lr_{lr}_type_{args.trainType}_label_ratio_{args.labelled_ratio}"
            folder_path = os.path.join("logs", folder_name)
            os.makedirs(folder_path, exist_ok=True)
            np.savez(os.path.join(folder_path, "results.npz"),
                     rmse_recorder=rmse_recorder,
                     elbo_recorder=elbo_recorder,
                     all_loss_recorder=all_loss_recorder)




if __name__ == '__main__':
    args = parse_args()
    # print(os.getcwd())
    main(args)
