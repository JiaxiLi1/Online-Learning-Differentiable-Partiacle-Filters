import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from distributions import randn
from utils import wrap_angle, et_distance, compute_sq_distance, transform_particles_as_input, particles_to_prediction, normal_log_density,normalize_log_probs, compute_normal_density
from loss_functions import *
from resamplers.resamplers import *
from build import *
from models import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DPF(nn.Module):

    def __init__(self, args):
        super(DPF, self).__init__()
        self.param = args
        self.hidden_size = args.hiddensize
        self.state_dim =3
        self.expanded_state_dim=4
        self.action_dim=3
        self.batch_size = args.batchsize
        self.num_particle = args.num_particles
        self.labeledRatio = args.labeledRatio
        self.n_sequence = args.n_sequence
        self.init_var = 0.001
        self.NF_lr = args.NF_lr
        self.NF = args.NF_dyn
        self.NFcond = args.NF_cond
        self.CNF_measurement = args.CNF_measurement
        self.lr = args.lr

        self.resampler = resampler(self.param)

        self.encoder = build_encoder(self.hidden_size, self.param.dropout_keep_ratio)
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

        self.decoder = build_decoder(self.hidden_size)
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.map_encoder = build_encoder(self.hidden_size, self.param.dropout_keep_ratio)
        self.map_encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

        self.map_decoder = build_decoder(self.hidden_size)
        self.map_decoder_optim = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.particle_encoder = build_particle_encoder(self.hidden_size, self.expanded_state_dim)
        self.particle_encoder_optim = torch.optim.Adam(self.particle_encoder.parameters(), lr=self.lr)

        self.obs_like_estimator = build_likelihood(self.hidden_size, self.expanded_state_dim)

        self.mo_noise_generator = build_action_noise(self.action_dim)
        self.mo_noise_generator_optim = torch.optim.Adam(self.mo_noise_generator.parameters(), lr=self.lr)

        self.motion_update = dynamic_model(self.mo_noise_generator,[self.param.std_x, self.param.std_y, self.param.std_t])

        self.nf_dyn = build_conditional_nf(self.n_sequence, 2 * self.state_dim, self.state_dim, init_var=self.init_var)
        self.nf_dyn_optim = torch.optim.Adam(self.nf_dyn.parameters(), lr=self.NF_lr*self.lr)

        self.cond_model = build_conditional_nf(self.n_sequence, 2 * self.state_dim + self.hidden_size, self.state_dim, init_var=self.init_var)
        self.cond_model_optim = torch.optim.Adam(self.cond_model.parameters(), lr=self.NF_lr*self.lr)

        self.cnf_measurement = build_conditional_nf(self.n_sequence, 2 * self.hidden_size, self.hidden_size, init_var=0.1, prior_std=10.0)
        self.cnf_measurement_optim = torch.optim.Adam(self.cnf_measurement.parameters(), lr=self.lr)

        if self.CNF_measurement:
            self.measurement_model = measurement_model_cnf(self.particle_encoder, self.cnf_measurement)
        else:
            self.measurement_model = measurement_model_cosine_distance(self.particle_encoder)

        self.prototype_density = compute_normal_density(pos_noise=self.param.std_x, vel_noise=self.param.std_t)

    def filtering(self, obs, state, action, means, stds, state_step_sizes, state_mins, state_maxs, maps):

        batch_size = self.batch_size
        seq_len = self.seq_len
        state_dim = self.state_dim
        num_particles = self.param.num_particles

        obs_normalized = (obs - torch.tensor(means['o']).permute(0, 1, 4, 2, 3).to(device)) / torch.tensor(stds['o']).permute(0, 1, 4, 2, 3).to(device)  # shape: (batch, seq, 3, 24, 24)

        ## initialisa particles
        if self.param.init_with_true_state:
            initial_noise_pos = torch.normal(mean=0., std=self.param.std_x, size=(self.batch_size, self.num_particle, 2)).to(device)
            initial_noise_t = torch.normal(mean=0., std=self.param.std_t, size=(self.batch_size, self.num_particle, 1)).to(device)
            initial_noise = torch.cat([initial_noise_pos,initial_noise_t],dim=-1)

            initial_particles = state[:, 0, None, :].repeat(1, num_particles, 1) + initial_noise
            # compute initial stationary distribution log density
            log_mu_s = normal_log_density(initial_noise_pos, self.param.std_x) + normal_log_density(initial_noise_t, self.param.std_t)
        else:
            initial_particles = torch.cat(
                [(torch.tensor(state_maxs[d]).to(device) - torch.tensor(state_mins[d]).to(device)) * (
                    torch.rand(batch_size, num_particles, 1).to(device)) + torch.tensor(state_mins[d]).to(device) for d
                 in
                 range(state_dim)], dim=2)
            initial_noise = initial_particles - state[:, 0, None, :].repeat(1, self.num_particle, 1)
            mu_s = 1.0
            for d in range(self.state_dim):
                mu_s *= 1.0 / (state_maxs[d] - state_mins[d])
            log_mu_s = torch.log(mu_s*torch.ones([batch_size, num_particles])).to(device)

        initial_particle_probs = log_mu_s #torch.ones([batch_size, num_particles]).to(device) / num_particles

        for step in range(0, seq_len):

            encodings_obs = self.encoder(obs_normalized[:, step].float())
            encodings_maps = self.map_encoder(maps)
            if step == 0:
                particles = initial_particles
                lki = self.measurement_model(encodings_obs, particles, means, stds, encodings_maps)
                particle_probs_log = initial_particle_probs + lki
                particle_probs = normalize_log_probs(particle_probs_log)
                index_p = (torch.arange(self.num_particle) +
                           self.num_particle * torch.arange(self.batch_size)[:,None].repeat((1, self.num_particle))).type(torch.int64).to(device)
                obs_likelihood = particle_probs_log.mean()

                particle_probs_list = particle_probs[:, None, :]
                particle_probs_before_res_list = particle_probs[:, None, :]
                likelihood_list = lki[:, None, :]
                noise_list = initial_noise[:, None, :, :]
                particle_list = initial_particles[:, None, :, :]
                index_list = index_p[:, :, None]
                continue

            # resampling
            ESS= torch.mean(1/torch.sum(particle_probs**2, dim=-1))
            if ESS<0.5*self.num_particle:
                particles_resampled, particle_probs_resampled, index_p = self.resampler(particles, particle_probs)
                particle_probs_resampled = particle_probs_resampled.log()
            else:
                particles_resampled, particle_probs_resampled, index_p = particles, particle_probs.log(), (torch.arange(self.num_particle) + self.num_particle * torch.arange(self.batch_size)[:,
                                                                         None].repeat((1, self.num_particle))).type(torch.int64).to(device)

            # motion update
            new_action = action[:, step-1]
            particles_physical, noise = self.motion_update(new_action, particles_resampled, means,
                                                    stds, state_step_sizes)
            particles_dynamical, jac = nf_dynamic_model(self.nf_dyn, particles_physical, particle_probs.shape,NF=self.NF)

            propose_particle, lki_log, prior_log, propose_log = proposal_likelihood(self.cond_model,
                                                                                    self.nf_dyn,
                                                                                    self.measurement_model,
                                                                                    particles_dynamical,
                                                                                    particles_physical,
                                                                                    encodings_obs, noise, jac,
                                                                                    self.NF, self.NFcond,
                                                                                    prototype_density= self.prototype_density,
                                                                                    means=means, stds=stds, encodings_maps=encodings_maps)
            particle_probs_resampled = particle_probs_resampled + lki_log + prior_log - propose_log

            obs_likelihood += particle_probs_resampled.mean()
            # standard_particle_probs *= lki

            # NORMALIZE AND COMBINE PARTICLES
            particles = propose_particle
            particle_probs = particle_probs_resampled

            # NORMALIZE PROBABILITIES
            particle_probs = normalize_log_probs(particle_probs)

            particle_list = torch.cat([particle_list, particles[:, None]], dim=1)
            particle_probs_list = torch.cat([particle_probs_list, particle_probs[:, None]], dim=1)
            noise_list = torch.cat([noise_list, noise[:, None]], dim=1)
            likelihood_list = torch.cat([likelihood_list, lki[:, None]], dim=1)
            index_list = torch.cat([index_list, index_p[:, :, None]], dim=-1)

        return particle_list, particle_probs_list, noise_list, likelihood_list, log_mu_s, index_list, obs_likelihood

    def supervised_loss(self, particle_list, particle_weight_list, states, mask, state_step_sizes, train):

        sq_distance = compute_sq_distance(particle_list[... , :2], states[:, :, None, :2], state_step_sizes)
        std = self.param.particle_std
        activations = particle_weight_list[:, :] / np.sqrt(2 * np.pi * std ** 2) * torch.exp(-sq_distance / (
                2.0 * self.param.particle_std ** 2))  # activations shape (batch_size, time_steps, particle_num)
        # loss = torch.sqrt(torch.mean((prediction - true_state) ** 2))  # Rooted mean square error

        if train:
            if self.param.labeledRatio > 0:

                loss = torch.mean(-mask*torch.log(1e-16 + torch.sum(activations, dim=2)))/self.param.labeledRatio

                # second loss (which we will monitor during execution)
                pred = particles_to_prediction(particle_list, particle_weight_list)
                sq_distance = compute_sq_distance(pred[:, -1, :2], states[:, -1, :2], state_step_sizes)
                loss_last = torch.mean(sq_distance)
                return loss, loss_last, pred,
            elif self.param.labeledRatio == 0:
                return None, None, None
        else:
            loss = torch.mean(-torch.log(1e-16 + torch.sum(activations, dim=2)))

            # second loss (which we will monitor during execution)
            pred = particles_to_prediction(particle_list, particle_weight_list)
            sq_distance = compute_sq_distance(pred[:, -1, :2], states[:, -1, :2], state_step_sizes)
            loss_last = torch.mean(sq_distance)
            return loss, loss_last, pred,
    #
    def supervised_loss_rmse(self, particle_list, particle_weight_list, states, mask, state_step_sizes, train):
        pred = torch.sum(particle_list * particle_weight_list[:, :, :, None],
                              dim=2)
        if train:
            if self.param.labeledRatio > 0:

                loss = torch.mean(mask * torch.sum((pred[:,:,:2] - states[:,:,:2]) ** 2,
                                                               dim=-1)) / self.param.labeledRatio  # Rooted mean square error
                sq_distance = (pred[:, -1, :2] - states[:, -1, :2])**2
                loss_last = torch.mean(sq_distance.sum(dim=-1))**0.5
                return loss**0.5, loss_last, pred,
            elif self.param.labeledRatio == 0:
                return None, None, None
        else:
            loss = torch.mean(torch.sum((pred[:,:,:2] - states[:,:,:2]) ** 2,
                                                           dim=-1))  # Rooted mean square error
            sq_distance = (pred[:, -1, :2] - states[:, -1, :2])**2
            loss_last = torch.mean(sq_distance.sum(dim=-1))**0.5
            return loss**0.5, loss_last, pred

    def get_mask(self):

        # number of 0 and 1
        N1 = int(self.batch_size*self.seq_len*self.labeledRatio)
        N0 = self.batch_size*self.seq_len - N1
        arr = np.array([0] * N0 + [1] * N1)
        np.random.shuffle(arr)
        mask = arr.reshape(self.batch_size, self.seq_len)

        mask = torch.tensor(mask).to(device)

        return mask

    def compute_block_density(self, particle_weight_list, noise_list, likelihood_list, index_list):
        batch_size = self.batch_size
        num_resampled = self.num_particle

        block_len = self.param.block_length
        std_x = torch.tensor(self.param.std_x).to(device)
        std_y = torch.tensor(self.param.std_y).to(device)
        std_t = torch.tensor(self.param.std_t).to(device)
        # log_mu_s shape: (batch_size, num_particle)
        # block index
        b =0
        # pseudo_likelihood
        Q =0
        logyita = 0
        log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi))
        for k in range(self.seq_len):
            if (k+1)% block_len==0:
                for j in range(k, k-block_len, -1):
                    if j == k:
                        lik = likelihood_list[:,j,:]
                        noise = noise_list[:, j, :, :]
                        index_a = index_list[:,:,j]
                    else:
                        index_pre = index_list[:,:,j]
                        index_a = index_pre.view((batch_size * num_resampled,))[index_a]
                        lik = likelihood_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]
                        noise = noise_list[:, j, :, :].reshape((batch_size * num_resampled,-1))[index_a,:]

                    # compute the prior density
                    logprior_x = log_c - torch.log(std_x) - (noise[:, :, 0]) ** 2 / (
                                2 * (std_x) ** 2)
                    logprior_y = log_c - torch.log(std_y) - (noise[:, :, 1]) ** 2 / (
                                2 * (std_y) ** 2)
                    logprior_t = log_c - torch.log(std_t) - (noise[:, :, 2]) ** 2 / (
                                2 * (std_t) ** 2)
                    log_prior = logprior_x + logprior_y + logprior_t

                    logyita = logyita + log_prior + torch.log(lik)
                Q = Q + torch.sum(particle_weight_list[:, k, :] * logyita, dim=-1)
                b = b+1
        # Q shape: (batch_size,)
        return Q/b

    def pseudolikelihood_loss(self, particle_weight_list, noise_list, likelihood_list, index_list):

        return -1. * torch.mean(self.compute_block_density(particle_weight_list, noise_list, likelihood_list, index_list))

    def set_zero_grad(self):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.particle_encoder_optim.zero_grad()
        self.mo_noise_generator_optim.zero_grad()

    def set_optim_step(self):
        self.encoder_optim.step()
        self.decoder_optim.step()
        self.particle_encoder_optim.step()
        self.mo_noise_generator_optim.step()


    def forward(self, input, statistics, train=True, maps=None):

        (states, actions, measurements) = input

        states = states.to(device)
        actions = actions.to(device)
        measurements = measurements.permute(0, 1, 4, 2, 3).to(device)  # for pytorch, the channels are in front of width*height

        self.seq_len = states.shape[1]

        (means, stds, state_step_sizes, state_mins, state_maxs) = statistics
        state_step_sizes=torch.tensor(state_step_sizes).to(device)
        # could not convert dict to tensor gpu

        particle_list, particle_weight_list, noise_list, likelihood_list, log_mu_s, index_list, obs_likelihood = self.filtering(measurements, states, actions, means, stds, state_step_sizes, state_mins, state_maxs, maps)

        # mask
        if train:
            mask = self.get_mask()  # shape: (batch_size, seq_len)
        else:
            mask = 1.0

        loss_sup, loss_last, pred = self.supervised_loss_rmse(particle_list, particle_weight_list, states, mask, state_step_sizes, train)

        if self.param.trainType == 'sup':
            # total_loss = loss_sup
            # loss_pseud_lik = None
            # loss_ae = None
            lamda1 = 10.0
            lamda2 = 0.01
            lamda3 = 200.0
            loss_pseud_lik = None
            loss_ae = autoencoder_loss(self.encoder, self.decoder, measurements, means, stds)
            # total_loss = lamda2 *loss_pseud_lik + lamda3* loss_ae
            total_loss = loss_sup + loss_ae #- obs_likelihood/self.seq_len

        elif self.param.trainType == 'semi':
            lamda1 = 10.0
            lamda2 = 0.01
            lamda3 = 200.0
            # loss_pseud_lik = None
            loss_pseud_lik = self.pseudolikelihood_loss(particle_weight_list, noise_list, likelihood_list, index_list)
            # loss_ae = None
            loss_ae = autoencoder_loss(self.encoder, self.decoder,measurements, means, stds)
            # total_loss = lamda2 *loss_pseud_lik + lamda3* loss_ae
            total_loss = lamda1 * loss_sup + lamda2 * loss_pseud_lik + lamda3 * loss_ae
        elif self.param.trainType == 'unsup':
            lamda2 = 1.0
            lamda3 = 10.0
            loss_pseud_lik = self.pseudolikelihood_loss(particle_weight_list, noise_list, likelihood_list, log_mu_s,
                                                        index_list)
            loss_ae = autoencoder_loss(self.encoder, self.decoder,measurements, means, stds)
            total_loss = lamda2 * loss_pseud_lik + lamda3 * loss_ae

        return total_loss, loss_sup, loss_last, loss_pseud_lik, loss_ae, pred, particle_list, particle_weight_list, states, obs_likelihood