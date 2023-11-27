import torch
import torch.nn as nn
import numpy as np
from utils import *
from tqdm import tqdm
from nf.flows import *
from nf.models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
import os
from torch.utils.tensorboard import SummaryWriter
from plot import *
from model.models import *
from builds.build_maze import *
from builds.build_disk import *
from builds.build_house3d import *
import cv2
from resamplers.resamplers import resampler
from losses import *
from nf.cglow.CGlowModel import CondGlowModel
from copy import deepcopy
from skimage.transform import resize
import time
import lgssm
import aesmc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#6,1994715,10,311,1006,54,23,6,24,98

class DPF_base(nn.Module):

    def __init__(self, args):
        super().__init__()

    def forward(self, t, slice_data_next, inputs, train=True, params=None, environment_data = None):
        if params.dataset == 'disk':
            (start_image, start_state, image, state, q, visible) = inputs
            state = state.to(device)
            start_state = start_state.to(device)
            image = image.permute(0, 1, 4, 2, 3).to(device)
            actions = state[:, :, 2:] + torch.normal(0.0, 4.0, (state[:, :, 2:]).shape).to(device)
            environment_obs = None
        elif params.dataset == 'maze':
            if self.param.learnType == 'offline':
                (state, image) = inputs

                reshaped_state = state.reshape(-1, state.size(-1))
                state_max = torch.max(reshaped_state, dim=0).values  # Shape: [dimension]
                state_min = torch.min(reshaped_state, dim=0).values  # Shape: [dimension]  # Shape: [batch_size, d]
                environment_data = [state_max, state_min]

            if self.param.learnType == 'online':
                (state, image) = inputs
                if t == 0:
                    start_state = None
                else:
                    start_state = slice_data_next[1]
                reshaped_state = state.reshape(-1, state.size(-1))
                state_max = torch.max(reshaped_state, dim=0).values  # Shape: [dimension]
                state_min = torch.min(reshaped_state, dim=0).values  # Shape: [dimension]  # Shape: [batch_size, d]
                environment_data = [state_max, state_min]


        else:
            raise ValueError('Please select a dataset from {disk, maze, house3d}')
        self.seq_len = state.shape[1]
        self.param.state_dim = state.shape[2]

        # modify the dimension of hidden state
        # start_time = time.time()
        slice_data_current, elbo, particle_list, particle_weight_list, likelihood_list, \
        prior_list, obs_likelihood = \
            self.filtering_pos(t, slice_data_next, image, train=train, environment_data =environment_data)
        # end_time = time.time()
        # time_seconds = end_time - start_time
        # print(f"each forward took {time_seconds:.2f} seconds to run.")
        # mask
        if train:
            mask = self.get_mask()  # shape: (batch_size, seq_len)
        else:
            mask = 1.0
        if self.param.dataset == 'maze':
            loss_alltime, loss_sup, loss_sup_last, predictions = supervised_loss(self.param.learnType, particle_list, particle_weight_list,
                                                                                 state, mask,
                                                                                 train,
                                                                                 labeledRatio=self.param.labeledRatio)
        elif self.param.dataset == 'house3d':
            loss_alltime, loss_sup, loss_sup_last, predictions = supervised_loss_house(particle_list, particle_weight_list,
                                                                                 state, mask,
                                                                                 train,
                                                                                 labeledRatio=self.param.labeledRatio)
        else:
            loss_alltime, loss_sup, loss_sup_last, predictions = supervised_loss(particle_list, particle_weight_list,
                                                                                 state, mask,
                                                                                 train,
                                                                                 labeledRatio=self.param.labeledRatio)

        loss_ae = None
        if self.param.trainType == 'DPF':
            lamda1 = 1.0
            lamda2 = 0.01
            lamda3 = 2.0
            loss_pseud_lik = None

            # total_loss = loss_sup + loss_ae - obs_likelihood / (self.seq_len * 10)  #
            # total_loss = loss_sup - (elbo.mean() * self.param.elbo_ratio)
            elbo_value = elbo.mean().detach().cpu().numpy()
            if self.param.learnType == 'offline':
                total_loss = -1e-2*torch.mean(elbo)#- (1e-2*elbo.mean() )
            elif self.param.learnType == 'online':
                if self.param.onlineType == 'elbo':
                    print(loss_sup.detach().cpu().numpy(), elbo_value)
                    total_loss = - (elbo.mean())
                elif self.param.onlineType == 'fix':
                    total_loss = 0
                elif self.param.onlineType == 'rmse':
                    total_loss = loss_sup
        else:
            raise ValueError('Please select the training type in DPF (supervised learning) and SDPF (semi-supervised learning)')

        return slice_data_current, elbo_value, loss_alltime, total_loss, loss_sup, loss_sup_last, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, image, likelihood_list, obs_likelihood

    def sample_ancestral_index(self, log_weight):
        """Sample ancestral index using systematic resampling.

        Args:
            log_weight: log of unnormalized weights, tensor
                [batch_size, num_particles]
        Returns:
            zero-indexed ancestral index: LongTensor [batch_size, num_particles]
        """

        if torch.sum(log_weight != log_weight).item() != 0:
            raise FloatingPointError('log_weight contains nan element(s)')

        batch_size, num_particles = log_weight.size()
        indices = np.zeros([batch_size, num_particles])

        uniforms = np.random.uniform(size=[batch_size, 1])
        pos = (uniforms + np.arange(0, num_particles)) / num_particles

        normalized_weights = aesmc.math.exponentiate_and_normalize(
            log_weight.detach().cpu().numpy(), dim=1)

        # np.ndarray [batch_size, num_particles]
        cumulative_weights = np.cumsum(normalized_weights, axis=1)

        # hack to prevent numerical issues
        cumulative_weights = cumulative_weights / np.max(
            cumulative_weights, axis=1, keepdims=True)

        for batch in range(batch_size):
            indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

        if log_weight.is_cuda:
            return torch.from_numpy(indices).long().cuda()
        else:
            return torch.from_numpy(indices).long()

    def filtering_pos(self, t, slice_data_next, obs, train = True, environment_data = None):
        if self.param.dataset=='maze':
            batch_size = obs.shape[0]
            environment_measurement = None
        else:
            raise ValueError('Please select a dataset from {disk, maze, house3d}')

        obs_likelihood = 0.0
        ancestral_indices = []
        log_weights = []
        # if self.param.learnType == 'online':
        if t == 0 or self.param.learnType == 'offline':
            particles, particle_probs = self.proposal.sample(observations=obs[:, 0, :], time=t, batch_size=batch_size,
                                                        num_particles=self.num_particle)
            transition_log_prob = aesmc.state.log_prob(self.initial(), particles)
            emission_log_prob = aesmc.state.log_prob(
                self.emission(latents=[particles], time=0),
                aesmc.state.expand_observation(obs[:,0,:], self.num_particle))

            log_weights.append(transition_log_prob + emission_log_prob -
                               particle_probs)
            particle_probs_resampled = transition_log_prob + emission_log_prob - particle_probs
            unnormalize_weight_next = particle_probs_resampled
            unnormalize_weight_de = torch.logsumexp(unnormalize_weight_next, dim=1)
            elbo = torch.zeros(unnormalize_weight_next.shape[0]).to(device)
            elbo += unnormalize_weight_de
            particle_probs = particle_probs_resampled
            particle_probs = normalize_log_probs(particle_probs) + 1e-12
            list_value='start_point'
            particle_list = particles[:, None, :, :]
            particle_probs_list = particle_probs[:, None, :]
            likelihood_list = emission_log_prob[:, None, :]
            if self.NF:
                prior_list = transition_log_prob[:, None, :]

        else:
            init_weights_log = None
            _, particles, particle_probs, unnormalize_weight_next = slice_data_next
            elbo = torch.zeros(unnormalize_weight_next.shape[0]).to(device)
            list_value='continue_point_from_last_slice_window'
        # filtering_start_time = time.time()
        for step in range(self.seq_len):
            if t==0:
                t=1
                continue
            # index_p shape: (batch, num_p)
            ancestral_indices.append(self.sample_ancestral_index(log_weights[-1]))
            particles_resampled = aesmc.state.resample(particles, ancestral_indices[-1])

            # index_p = (torch.arange(self.num_particle)+self.num_particle* torch.arange(batch_size)[:, None].repeat((1, self.num_particle))).type(torch.int64).to(device)
            # ESS = torch.mean(1/torch.sum(particle_probs**2, dim=-1))
            #
            # if ESS<0.5*self.num_particle:
            #     if train:
            #         particles_resampled, particle_probs_resampled, index_p = self.resampler(particles, particle_probs)
            #     else:
            #         particles_resampled, particle_probs_resampled, index_p = self.resampler(particles,
            #                                                                                      particle_probs)
            #     particle_probs_resampled = particle_probs_resampled.log()
            #     particle_probs_elbo = particle_probs_resampled
            # else:
            #     particles_resampled = particles
            #     particle_probs_resampled = particle_probs.log()
            #     particle_probs_elbo = unnormalize_weight_next
            #
            # unnormalize_weight_nu = torch.logsumexp(particle_probs_elbo, dim=1)

            propose_particle, proposal_log_prob = self.proposal.sample(previous_latents=particles_resampled,
                                                        observations=obs[:,step,:],
                                                        time=time,
                                                        batch_size=batch_size,
                                                        num_particles=self.num_particle)
            transition_log_prob = aesmc.state.log_prob(
                self.transition(previous_latents=[particles_resampled], time=time),
                propose_particle)
            emission_log_prob = aesmc.state.log_prob(
                self.emission(latents=[propose_particle], time=time),
                aesmc.state.expand_observation(obs[:,step,:], self.num_particle))

            log_weights.append(transition_log_prob + emission_log_prob -
                               proposal_log_prob)

            # particle_probs_resampled = particle_probs_resampled + emission_log_prob + transition_log_prob - proposal_log_prob
            #
            # unnormalize_weight_next = particle_probs_resampled
            # unnormalize_weight_de = torch.logsumexp(unnormalize_weight_next, dim=1)
            # elbo += unnormalize_weight_de - unnormalize_weight_nu
            particles = propose_particle
            particle_probs = log_weights[-1]
            obs_likelihood += particle_probs.mean()
            particle_probs = normalize_log_probs(particle_probs)+1e-12

            if list_value == 'continue_point_from_last_slice_window':
                particle_list = particles[:, None, :, :]
                particle_probs_list = particle_probs[:, None, :]
                likelihood_list = emission_log_prob[:, None, :]
                if self.NF:
                    prior_list = transition_log_prob[:, None, :]
                list_value = 'continue'
            elif list_value == 'start_point' or list_value == 'continue':
                particle_list = torch.cat([particle_list, particles[:, None]], dim=1)
                particle_probs_list = torch.cat([particle_probs_list, particle_probs[:, None]], dim=1)
                likelihood_list = torch.cat([likelihood_list, emission_log_prob[:, None]], dim=1)
                if self.NF:
                    prior_list = torch.cat([prior_list, transition_log_prob[:, None]], dim=1)

        temp = torch.logsumexp(torch.stack(log_weights, dim=0), dim=2) - \
                np.log(self.num_particle)
        log_marginal_likelihood = torch.sum(temp, dim=0)
        return [obs_likelihood, particles, particle_probs, unnormalize_weight_next], log_marginal_likelihood, particle_list, particle_probs_list, likelihood_list, prior_list, obs_likelihood

    def get_mask(self):

        # number of 0 and 1
        N1 = int(self.batch_size*self.seq_len*self.labeledRatio)
        N0 = self.batch_size*self.seq_len - N1
        arr = np.array([0] * N0 + [1] * N1)
        np.random.shuffle(arr)
        mask = arr.reshape(self.batch_size, self.seq_len)

        mask = torch.tensor(mask).to(device)

        return mask

    # def get_mask(self, batch_number):
    #
    #     # number of 0 and 1
    #     N1 = int(batch_number*self.batch_size*self.seq_len*self.labeledRatio)
    #     N0 = batch_number*self.batch_size*self.seq_len - N1
    #     arr = np.array([0] * N0 + [1] * N1)
    #     np.random.shuffle(arr)
    #     mask = arr.reshape(batch_number, self.batch_size, self.seq_len)
    #
    #     mask = torch.tensor(mask).to(device)
    #
    #     return mask

    def pretrain_ae(self, train_loader, valid_loader, start_epoch=-1, epoch_num = 100, logger = None, params=None, environment_data=None):

        best_eval_loss = 1e10
        best_epoch = -1

        for epoch in range(start_epoch + 1, epoch_num):
            # train
            self.train()
            total_loss = []
            for batch_idx, inputs in enumerate(train_loader):
                if params.dataset == 'disk':
                    (start_image, start_state, image, state, q, visible) = inputs
                    state = state.to(device)
                    start_state = start_state.to(device)
                    image = image.permute(0, 1, 4, 2, 3).to(device)
                    actions = state[:, :, 2:] + torch.normal(0.0, 4.0, (state[:, :, 2:]).shape).to(device)
                    environment_obs = None
                elif params.dataset == 'maze':
                    (state, actions, image) = inputs
                    state = state.to(device)
                    start_state = state.clone()[:, 0]
                    state = state.clone()[:, 1:]
                    actions = actions.to(device)
                    image = image.permute(0, 1, 4, 2, 3).to(device)
                    image = image.clone()[:, 1:]
                    maps, statistics = environment_data
                    (means, stds, state_step_sizes, state_mins, state_maxs) = statistics
                    environment_obs = (torch.tensor(means['o']).permute(0, 1, 4, 2, 3).to(device),
                                       torch.tensor(stds['o']).permute(0, 1, 4, 2, 3).to(device))
                    environment_measurement = (means, stds, maps)
                elif params.dataset == 'house3d':
                    true_states, global_map, init_particles, observation, odometry, _ = inputs
                    state = torch.tensor(true_states).to(device)
                    state[..., -1] = wrap_angle(state[..., -1].clone())
                    state_mins = [state[..., i].min(dim=-1)[0] for i in range(state.shape[-1])]
                    state_maxs = [state[..., i].max(dim=-1)[0] for i in range(state.shape[-1])]
                    statistics = state_mins, state_maxs
                    start_state = state.clone()[:, 0]
                    state = state.clone()[:, 1:]
                    actions = torch.tensor(odometry).to(device)
                    image = torch.tensor(observation).to(device).permute(0, 1, 4, 2, 3)
                    image = image.clone()[:, 1:]

                    environment_data = global_map, statistics
                    environment_obs = None
                else:
                    raise ValueError('Please select a dataset from {disk, maze, house3d}')
                loss_measurement = 0
                loss_ae = 0
                image = preprocess_obs(image, environment_obs, self.param)
                for step in range(state.shape[1]):

                    pseudo_particles = torch.tile(state[None, :, step], [state.shape[0], 1, 1])
                    encodings = self.encoder(image[:, step].float())
                    measurement_model_out = self.measurement_model(encodings, pseudo_particles, environment_measurement,
                                                                   pretrain = True)
                    correct_samples = torch.diag(measurement_model_out)
                    incorrect_samples = measurement_model_out-torch.eye(correct_samples.shape[0]).to(device)*correct_samples
                    loss_measurement += torch.sum(-torch.log(correct_samples)) / self.batch_size \
                                        + torch.sum(-torch.log(1-incorrect_samples)) / self.batch_size * (self.batch_size - 1)                # image = image.permute(0, 1, 4, 2, 3)
                # image = image.reshape(-1, 3, 24, 24)
                loss_ae = autoencoder_loss(image, True, self.encoder, self.decoder, stats=environment_obs,
                                           params=params)

                # loss = loss_measurement + loss_ae
                loss = loss_measurement / 10000 + loss_ae
                self.zero_grad()

                loss.backward()

                self.optim.step()

                print(f"Train individually: Iter: {batch_idx}, measurement loss: {loss_measurement.detach().cpu().numpy()}, ae loss: {loss_ae.detach().cpu().numpy()}")
                total_loss.append(loss.detach().cpu().numpy())

            print(f"Train AE: Epoch: {epoch}, loss: {np.mean(total_loss)}")

            # validation
            self.eval()
            total_val_loss = []
            with torch.no_grad():
                for batch_idx, inputs in enumerate(valid_loader):
                    if params.dataset == 'disk':
                        (start_image, start_state, image, state, q, visible) = inputs
                        state = state.to(device)
                        start_state = start_state.to(device)
                        image = image.permute(0, 1, 4, 2, 3).to(device)
                        actions = state[:, :, 2:] + torch.normal(0.0, 4.0, (state[:, :, 2:]).shape).to(device)
                        environment_obs = None
                    elif params.dataset == 'maze':
                        (state, actions, image) = inputs
                        state = state.to(device)
                        start_state = state.clone()[:, 0]
                        state = state.clone()[:, 1:]
                        actions = actions.to(device)
                        image = image.permute(0, 1, 4, 2, 3).to(device)
                        image = image.clone()[:, 1:]
                        maps, statistics = environment_data
                        (means, stds, state_step_sizes, state_mins, state_maxs) = statistics
                        environment_obs = (torch.tensor(means['o']).permute(0, 1, 4, 2, 3).to(device),
                                           torch.tensor(stds['o']).permute(0, 1, 4, 2, 3).to(device))
                        environment_measurement = (means, stds, maps)
                    elif params.dataset == 'house3d':
                        true_states, global_map, init_particles, observation, odometry, _ = inputs
                        state = torch.tensor(true_states).to(device)
                        state[..., -1] = wrap_angle(state[..., -1].clone())
                        state_mins = [state[..., i].min(dim=-1)[0] for i in range(state.shape[-1])]
                        state_maxs = [state[..., i].max(dim=-1)[0] for i in range(state.shape[-1])]
                        statistics = state_mins, state_maxs
                        start_state = state.clone()[:, 0]
                        state = state.clone()[:, 1:]
                        actions = torch.tensor(odometry).to(device)
                        image = torch.tensor(observation).to(device).permute(0, 1, 4, 2, 3)
                        image = image.clone()[:, 1:]

                        environment_data = global_map, statistics
                        environment_obs = None
                    else:
                        raise ValueError('Please select a dataset from {disk, maze, house3d}')

                    loss_measurement = 0
                    loss_ae = 0
                    image = preprocess_obs(image, environment_obs, self.param)
                    for step in range(state.shape[1]):
                        pseudo_particles = torch.tile(state[None, :, step], [state.shape[0], 1, 1])
                        encodings = self.encoder(image[:, step].float())
                        measurement_model_out = self.measurement_model(encodings, pseudo_particles,environment_measurement,
                                                                       pretrain = True)
                        correct_samples = torch.diag(measurement_model_out)
                        incorrect_samples = measurement_model_out - torch.eye(correct_samples.shape[0]).to(
                            device) * correct_samples
                        loss_measurement += torch.sum(-torch.log(correct_samples)) / self.batch_size \
                                            + torch.sum(-torch.log(1 - incorrect_samples)) / self.batch_size * (
                                                        self.batch_size - 1)  # image = image.permute(0, 1, 4, 2, 3)
                        # image = image.permute(0, 1, 4, 2, 3)
                        # image = image.reshape(-1, 3, 24, 24)
                    loss_ae = autoencoder_loss(image, True, self.encoder, self.decoder, stats=environment_obs,
                                               params=params)

                    # loss = loss_measurement + loss_ae
                    loss = loss_measurement / 10000 + loss_ae
                    self.zero_grad()

                    print(f"Evaluation AE: Iter: {batch_idx},  measurement loss: {loss_measurement.detach().cpu().numpy()}, ae loss: {loss_ae.detach().cpu().numpy()}")
                    total_val_loss.append(loss.detach().cpu().numpy())

                    # plot_obs(img.reshape(batchsize, seq_len, 3, 128, 128),
                    #          recontr_img.reshape(batchsize, seq_len, 3, 128, 128))

                eval_loss_sup_mean = np.mean(total_val_loss)
                logger.add_scalar('PretrainAE_loss_eval/loss', eval_loss_sup_mean, epoch)
                print(f"Evaluation AE: Epoch: {epoch}, loss: {eval_loss_sup_mean}")

            # save pretain ae
            if eval_loss_sup_mean < best_eval_loss:
                best_eval_loss = eval_loss_sup_mean
                best_epoch = epoch
                print('Save best validation AE-model!')
                ckpt_ae = {
                    "model": self.state_dict(),
                    'optim': self.optim.state_dict(),
                }
                torch.save(ckpt_ae, './model/ae_pretrain.pth')
        # load the pretrained dynamic model
        self.load_state_dict(ckpt_ae['model'])
        self.optim.load_state_dict(ckpt_ae['optim'])

    def e2e_train(self, train_loader, valid_loader, start_epoch=-1, epoch_num = 100, logger = None,
                  run_id=None, environment_data = None, num_train_batch=1, num_val_batch=1, learnType='offline', slice_size = 100):

        params = self.param

        best_eval_loss = 1e10
        best_epoch = -1

        if self.param.load_pretrainModel:
            print('Load pretrained model')
            # load pretrained ae model
            ckpt_ae = torch.load('./model/ae_pretrain.pth')
            self.model.load_state_dict(ckpt_ae['model'])

        eval_loss_epoch = []
        eval_loss_std_epoch = []

        if learnType == 'offline':
            for epoch in range(start_epoch + 1, epoch_num):
                print('theta1:', self.transition.mult.flatten().cpu().detach().numpy()[0], 'theta2:',
                      self.emission.mult.cpu().detach().numpy())
                # train tqdm(
                self.train()
                total_sup_loss = []
                total_sup_last_loss = []
                total_elbo = []
                total_ae_loss = []
                for iteration, inputs in enumerate(tqdm(train_loader)):
                    if iteration >= num_train_batch:
                        break
                    self.zero_grad()
                    (state, image) = inputs
                    if state[0].dim() == 1:
                        for i in range(len(state)):
                            state[i] = state[i].unsqueeze(0)
                            image[i] = image[i].unsqueeze(0)
                    state = torch.stack(state, dim=1)
                    state = state.to(device)
                    image = (torch.stack(image, dim=1)).to(device)
                    image = image.clone()
                    # start_time = time.time()
                    slice_data_current, elbo_value, loss_alltime, loss_all, loss_sup, loss_sup_last, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, image, likelihood_list, obs_likelihood = self.forward(
                        0, None, [state, image], train=True, params=params, environment_data=environment_data)
                    if loss_all != 0:
                        loss_all.backward()
                        self.optim.step()  # self.set_optim_step()
                    total_sup_loss.append(loss_sup.detach().cpu().numpy())
                    total_sup_last_loss.append(loss_sup_last.detach().cpu().numpy())
                    total_elbo.append(elbo_value)
                    # total_ae_loss.append(loss_ae.detach().cpu().numpy())
                # self.optim_scheduler.step()
                train_loss_sup_mean = np.mean(total_sup_loss)
                train_loss_last_sup_mean = np.mean(total_sup_last_loss)
                # total_ae_loss_mean = np.mean(total_ae_loss)
                logger.add_scalar('Sup_loss/loss', train_loss_sup_mean, epoch)
                # print(f"End-to-end loss: epoch: {epoch}, loss: {train_loss_sup_mean}, loss_last: {train_loss_last_sup_mean}, loss_ae: {total_ae_loss_mean}, obs_likelihood: {obs_likelihood}")

                # evaluate tqdm(
                self.eval()
                total_sup_eval_loss = []
                total_sup_last_eval_loss = []
                with torch.no_grad():
                    for iteration, inputs in enumerate(tqdm(valid_loader)):

                        if iteration >= num_val_batch:
                            break
                        self.zero_grad()
                        (state, image) = inputs
                        if state[0].dim() == 1:
                            for i in range(len(state)):
                                state[i] = state[i].unsqueeze(0)
                                image[i] = image[i].unsqueeze(0)
                        state = torch.stack(state, dim=1)
                        state = state.to(device)
                        image = (torch.stack(image, dim=1)).to(device)
                        image = image.clone()
                        slice_data_current, elbo_value, loss_alltime, loss_all, loss_sup, loss_sup_last, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, image, likelihood_list, obs_likelihood = self.forward(
                            0, None, [state, image], train=False, params=params, environment_data=environment_data)
                        total_sup_eval_loss.append(loss_sup.detach().cpu().numpy())
                        total_sup_last_eval_loss.append(loss_sup_last.detach().cpu().numpy())

                    eval_loss_sup_mean = np.mean(total_sup_eval_loss)
                    eval_loss_sup_std = np.std(total_sup_eval_loss)
                    eval_loss_last_sup_mean = np.mean(total_sup_last_eval_loss)
                    mean_rmse = np.mean(np.sqrt(total_sup_last_eval_loss))
                    total_rmse = np.sqrt(np.mean(total_sup_last_eval_loss))
                    logger.add_scalar('Sup_loss_eval/loss', eval_loss_sup_mean, epoch)
                    print(
                        f"End-to-end loss evaluation: epoch: {epoch}, loss: {eval_loss_sup_mean}, loss_last: {eval_loss_last_sup_mean}, Mean RMSE: {mean_rmse}, Overall RMSE: {total_rmse}, obs_likelihood: {obs_likelihood}",
                        self.NF)

                eval_loss_epoch.append(eval_loss_sup_mean)  ##############
                eval_loss_std_epoch.append(eval_loss_sup_std)

                if eval_loss_sup_mean < best_eval_loss:
                    best_eval_loss = eval_loss_sup_mean
                    best_epoch = epoch
                    # print('Save best validation model')
                    np.savez(os.path.join('logs', run_id, "data", 'eval_result_best.npz'),
                             particle_list=particle_list.detach().cpu().numpy(),
                             particle_weight_list=particle_weight_list.detach().cpu().numpy(),
                             likelihood_list=likelihood_list.detach().cpu().numpy(),
                             pred=predictions.detach().cpu().numpy(),
                             state=state.detach().cpu().numpy(),
                             loss=total_sup_eval_loss)
                    checkpoint_e2e = checkpoint_state(self, epoch)
                    torch.save(checkpoint_e2e, os.path.join('logs', run_id, "models", 'e2e_model_bestval_e2e.pth'))
            np.save(os.path.join('logs', run_id, "data", 'eval_loss_mean_epoch.npy'), eval_loss_epoch)
            np.save(os.path.join('logs', run_id, "data", 'eval_loss_std_epoch.npy'), eval_loss_std_epoch)

        if learnType == 'online':
            slice_data_next = None
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            # train tqdm(
            self.train()
            total_sup_loss = []
            total_sup_last_loss = []
            total_elbo = []
            total_ae_loss = []
            loss_alltime_list = []
            for iteration, inputs in enumerate(tqdm(train_loader)):
                if iteration >= num_train_batch:
                    break
                (state, image) = inputs
                if state[0].dim() == 1:
                    for i in range(len(state)):
                        state[i] = state[i].unsqueeze(0)
                        image[i] = image[i].unsqueeze(0)
                state = torch.stack(state, dim=1)
                state = state.to(device)
                image = (torch.stack(image, dim=1)).to(device)
                image = image
                # Process in chunks of slice_size timesteps
                for t in range(0, 100, slice_size):
                    self.zero_grad()
                    # Slicing the state positions and observations
                    state_positions = state[:, t:min(t + slice_size, 100), :]
                    observations = image[:, t:min(t + slice_size, 100), ...]
                    input = [state_positions.clone(), observations.clone()]

                    # Forward pass for the sliced segment
                    slice_data_current, elbo_value, loss_alltime, loss_all, loss_sup, loss_sup_last, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state_, image_, likelihood_list, obs_likelihood = self.forward(
                        t, slice_data_next, input, train=True, params=params,
                        environment_data=environment_data)
                    detached_list = [tensor.detach() for tensor in slice_data_current]
                    slice_data_next = detached_list
                    # Backpropagate and update model parameters
                    if loss_all != 0:
                        loss_all.backward()
                        self.optim.step()

                    # Append losses for logging or monitoring
                    total_sup_loss.append(loss_sup.detach().cpu().numpy())
                    total_sup_last_loss.append(loss_sup_last.detach().cpu().numpy())
                    total_elbo.append(elbo_value)
                    loss_alltime_list.append(loss_alltime[0])
                    # total_ae_loss.append(loss_ae.detach().cpu().numpy())

                self.optim_scheduler.step()
            numpy_loss_alltime_list = [tensor.detach().cpu().numpy() for tensor in loss_alltime_list]
            np.savez(os.path.join('logs', run_id, "data", 'loss_alltime_list.npz'),
                     loss_alltime=numpy_loss_alltime_list,)
            np.save(os.path.join('logs', run_id, "data", 'online_rmse.npy'), total_sup_loss)
            np.save(os.path.join('logs', run_id, "data", 'online_elbo.npy'), total_elbo)

            # # train_loss_sup_mean = np.mean(total_sup_loss)
            # # logger.add_scalar('Sup_loss/loss', train_loss_sup_mean, epoch)
            # # evaluate tqdm(
            # self.eval()
            # total_sup_eval_loss = []
            # total_sup_last_eval_loss = []
            # total_elbo_val = []
            # with torch.no_grad():
            #     for iteration, inputs in enumerate(tqdm(valid_loader)):
            #         if iteration >= num_val_batch:
            #             break
            #         self.zero_grad()
            #
            #         for t in range(0, 100, slice_size):
            #             self.zero_grad()
            #
            #             # Slicing the state positions and observations
            #             state_positions = inputs[0][:, t:min(t + slice_size, 100), :]
            #             observations = inputs[2][:, t:min(t + slice_size, 100), ...]
            #             actions = inputs[1][:, t:min(t + slice_size, 99), :]
            #
            #             # Forward pass for the sliced segment
            #             elbo_value, loss_alltime, loss_all, loss_sup, loss_sup_last, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, start_state, image, likelihood_list, noise_list, obs_likelihood = self.forward(
            #                 [state_positions, actions, observations], train=True, params=params,
            #                 environment_data=environment_data)
            #             total_sup_eval_loss.append(loss_sup.detach().cpu().numpy())
            #             total_sup_last_eval_loss.append(loss_sup_last.detach().cpu().numpy())
            #             total_elbo_val.append(elbo_value)
            #
            #     eval_loss_sup_mean = np.mean(total_sup_eval_loss)
            #     eval_loss_sup_std = np.std(total_sup_eval_loss)
            #     eval_loss_last_sup_mean = np.mean(total_sup_last_eval_loss)
            #     mean_rmse = np.mean(np.sqrt(total_sup_last_eval_loss))
            #     total_rmse = np.sqrt(np.mean(total_sup_last_eval_loss))
            #     # logger.add_scalar('Sup_loss_eval/loss', eval_loss_sup_mean, epoch)
            #     print(f"online loss evaluation: loss: {eval_loss_sup_mean}, loss_last: {eval_loss_last_sup_mean}, Mean RMSE: {mean_rmse}, Overall RMSE: {total_rmse}, obs_likelihood: {obs_likelihood}", self.NF)
            #
            # np.save(os.path.join('logs', run_id, "data", 'online_eval_loss.npy'), total_sup_eval_loss)
            # np.save(os.path.join('logs', run_id, "data", 'online_eval_elbo.npy'), total_elbo_val)

    def load_model(self, file_name):
        ckpt_e2e = torch.load(file_name)
        load_model(self, ckpt_e2e)
        epoch = ckpt_e2e['epoch']

        print(f'Load epcoh: {epoch}')

    def train_val(self, train_loader, valid_loader, run_id, environment_data, num_train_batch, num_val_batch):
        params = self.param
        epoch_num = params.num_epochs

        dirs = ['result', 'model', 'checkpoint', 'logger']
        flags = [os.path.isdir(dir) for dir in dirs]
        for i, flag in enumerate(flags):
            if not flag:
                os.mkdir(dirs[i])

        logger = SummaryWriter('./logger')

        start_epoch = -1

        if params.resume:
            print('Resume training!')
            self.load_model('./model/e2e_model_bestval_e2e.pth')

        if params.pretrain_ae:#False
            print("Pretrain autoencoder model!")
            self.pretrain_ae(train_loader, valid_loader, start_epoch=start_epoch, epoch_num=150, logger=logger, params=params, environment_data = environment_data)

        if params.e2e_train:#True
            # end-to-end training
            # print('End-to-end training!')
            self.e2e_train(train_loader, valid_loader, start_epoch=start_epoch, epoch_num=epoch_num,
                           logger=logger, run_id = run_id, environment_data = environment_data,
                           num_train_batch=num_train_batch, num_val_batch = num_val_batch, learnType=params.learnType, slice_size=params.slice_size)

    def testing(self, test_loader, run_id, model_path='./model/e2e_model_bestval_e2e.pth', environment_data=None, num_test_batch=1):

        params = self.param
        if self.param.testing:
            print('Testing!')
            print('Load trained model')
            self.load_model(os.path.join(model_path, 'e2e_model_bestval_e2e.pth'))
        loss_buffer = torch.empty(0).float().to('cuda')
        for epoch in range(1):
            # test tqdm(
            self.eval()
            total_sup_eval_loss = []
            total_sup_last_eval_loss = []

            with torch.no_grad():
                for iteration, inputs in enumerate(tqdm(test_loader)):
                    if iteration >= num_test_batch:
                        break
                    self.zero_grad()
                    loss_alltime, loss_all, loss_sup, loss_sup_last, loss_pseud_lik, loss_ae, predictions, particle_list, particle_weight_list, state, start_state, image, likelihood_list, noise_list,obs_likelihood = self.forward(
                        inputs, train=False, params = params, environment_data=environment_data)
                    total_sup_eval_loss.append(loss_sup.detach().cpu().numpy())
                    total_sup_last_eval_loss.append(loss_sup_last.detach().cpu().numpy())
                    loss_buffer=torch.cat((loss_buffer, loss_alltime.float()),dim=0)
            mean_rmse = np.mean(np.sqrt(total_sup_last_eval_loss))
            total_rmse = np.sqrt(np.mean(total_sup_last_eval_loss))
            np.save(os.path.join('logs', run_id, "data", 'test_loss_epoch.npy'), total_sup_eval_loss)
            print(f"End-to-end loss testing: loss: {np.mean(total_sup_eval_loss)}, loss_last: {np.mean(total_sup_last_eval_loss)}, Mean RMSE: {mean_rmse}, Overall RMSE: {total_rmse}")
            np.savez(os.path.join('logs', run_id, "data",'test_result.npz'),
                     particle_list= particle_list.detach().cpu().numpy(),
                     particle_weight_list=particle_weight_list.detach().cpu().numpy(),
                     likelihood_list=likelihood_list.detach().cpu().numpy(),
                     state=state.detach().cpu().numpy(),
                     pred=predictions.detach().cpu().numpy(),
                     images=image.detach().cpu().numpy(),
                     noise=noise_list.detach().cpu().numpy(),
                     loss_buffer=loss_buffer.detach().cpu().numpy()),


class DPF_disk(DPF_base):

    def __init__(self, args):
        super().__init__(args)
        self.param = args
        self.NF = args.NF_dyn
        self.NFcond = args.NF_cond
        self.measurement = args.measurement
        self.hidden_size = args.hiddensize  # origin: 32
        self.state_dim = 2  # 4
        self.param.state_dim = self.state_dim
        self.lr = args.lr
        self.alpha = args.alpha
        self.seq_len = args.sequence_length
        self.num_particle = args.num_particles
        self.batch_size = args.batchsize

        self.labeledRatio = args.labeledRatio

        self.spring_force = 0.1  # 0.1 #0.05  # 0.1 for one object; 0.05 for five objects
        self.drag_force = 0.0075  # 0.0075

        self.pos_noise = args.pos_noise  # 0.1 #0.1
        self.vel_noise = args.vel_noise  # 2.
        self.NF_lr = args.NF_lr
        self.n_sequence = 2

        self.build_model()

        self.eps = args.epsilon
        self.scaling = args.scaling
        self.threshold = args.threshold
        self.max_iter = args.max_iter
        self.resampler = resampler(self.param)

        test_param=deepcopy(self.param)
        test_param.resampler_type = 'soft'
        test_param.alpha = 1.0
        self.resampler_test = resampler(test_param)

    def build_model(self):
        if self.measurement=='CGLOW':
            self.encoder = build_encoder_cglow_disk(self.hidden_size)
            self.decoder = build_encoder_cglow_disk(self.hidden_size)
            self.build_particle_encoder = build_particle_encoder_cglow_disk()
        else:
            self.encoder = build_encoder_disk(self.hidden_size)
            self.decoder = build_decoder_disk(self.hidden_size)
            self.build_particle_encoder = build_particle_encoder_disk
        self.encoder_flow = build_encoder_disk(2)

        self.obs_feature = self.encoder

        self.particle_encoder = self.build_particle_encoder(self.hidden_size, self.state_dim)
        self.transition_model = build_transition_model_disk(self.state_dim)
        self.motion_update=motion_update_disk

        # normalising flow dynamic initialisation
        self.nf_dyn = build_conditional_nf(self.n_sequence, 2 * self.state_dim, self.state_dim, init_var=0.01)
        self.cond_model = build_conditional_nf(self.n_sequence, 2 * self.state_dim + self.hidden_size, self.state_dim, init_var=0.01)

        if self.measurement=='CRNVP':
            self.cnf_measurement = build_conditional_nf(self.n_sequence, self.hidden_size, self.hidden_size,
                                                        init_var=0.01, prior_std=2.5)
            self.measurement_model = measurement_model_cnf(self.particle_encoder, self.cnf_measurement, self.param)
        elif self.measurement=='cos':
            self.measurement_model = measurement_model_cosine_distance(self.particle_encoder, self.param)
        elif self.measurement=='NN':
            self.likelihood_est = build_likelihood_disk(self.hidden_size, self.state_dim)
            self.measurement_model = measurement_model_NN(self.particle_encoder, self.likelihood_est, self.param)
        elif self.measurement=='gaussian':
            self.gaussian_distribution = torch.distributions.MultivariateNormal(torch.ones(self.hidden_size).to(device),
                                                                                100 * torch.eye(self.hidden_size).to(device))
            self.measurement_model = measurement_model_Gaussian(self.particle_encoder, self.gaussian_distribution, self.param)
        elif self.measurement=='CGLOW':
            self.cglow_measurement = build_conditional_glow(self.param).to(device)
            self.measurement_model = measurement_model_cglow(self.particle_encoder, self.cglow_measurement)

        self.prototype_density=compute_normal_density_disk(pos_noise=self.pos_noise, vel_noise= self.vel_noise)

        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[30*(1+x) for x in range(10)], gamma=1.0)

class DPF_maze(DPF_base):
    def __init__(self, args, parameter_input):
        super().__init__(args)
        self.param = args
        self.hidden_size = args.hiddensize
        self.hidden_size_flow = args.hiddensize
        self.state_dim =args.dim
        self.param.state_dim = self.state_dim
        # self.expanded_state_dim= self.state_dim
        self.action_dim=args.dim
        self.batch_size = args.batchsize
        self.num_particle = args.num_particles
        self.labeledRatio = args.labeledRatio
        self.n_sequence = args.n_sequence
        self.init_var = 0.001
        self.NF_lr = args.NF_lr
        self.NF = args.NF_dyn
        self.NFcond = args.NF_cond
        self.measurement = args.measurement
        self.lr = args.lr
        (
        self.initial_loc, self.initial_scale, self.init_transition_mult, self.transition_scale, self.init_emission_mult,
        self.emission_scale, self.init_proposal_scale_0, self.init_proposal_scale_t) = parameter_input
        self.build_model()

        self.resampler = resampler(self.param)
        test_param=deepcopy(self.param)
        test_param.resampler_type = 'soft'
        test_param.alpha = 1.0
        self.resampler_test = resampler(test_param)


    def build_model(self):

        self.initial = lgssm.Initial(self.initial_loc, self.initial_scale)
        self.transition = lgssm.Transition(self.init_transition_mult,
                                           self.transition_scale)
        self.emission = lgssm.Emission(self.init_emission_mult,
                                       self.emission_scale)
        self.proposal = lgssm.Proposal_cnf(initial=self.initial,
                                      transition=self.transition,
                                      scale_0=self.init_proposal_scale_0,
                                      scale_t=self.init_proposal_scale_t,
                                      k=1,
                                      type='nvp').to(device)

        self.encoder = build_encoder_maze(self.hidden_size, self.state_dim, self.param.dropout_keep_ratio)
        self.encoder_flow = build_encoder_maze_flow(self.hidden_size, self.hidden_size_flow)

        self.decoder = build_decoder_maze(self.hidden_size)

        self.map_encoder = build_encoder_maze(self.hidden_size, self.state_dim, self.param.dropout_keep_ratio)

        self.map_decoder = build_decoder_maze(self.hidden_size)

        self.particle_encoder = build_particle_encoder_maze(self.hidden_size, self.state_dim)

        self.obs_like_estimator = build_likelihood_maze(self.hidden_size, self.state_dim)

        self.mo_noise_generator = build_action_noise_maze(self.action_dim)
        self.state_noise = [self.param.std_x, self.param.std_y, self.param.std_t]

        self.motion_update = motion_update_maze(self.mo_noise_generator, self.state_noise)

        self.nf_dyn = build_conditional_nf(self.n_sequence, 2 * self.state_dim, self.state_dim, init_var=self.init_var) #, flow=Planar

        self.cond_model = build_conditional_nf(self.n_sequence, self.hidden_size_flow + 2*self.state_dim, self.state_dim, init_var=self.init_var)

        self.cnf_measurement = build_conditional_nf(self.n_sequence, self.hidden_size, self.hidden_size, init_var=0.01, prior_std=1.0) #init_var=0.1, prior_std=10.0

        if self.measurement == 'CRNVP':
            self.measurement_model = measurement_model_cnf(self.particle_encoder, self.cnf_measurement, self.param)
        elif self.measurement == 'NN':
            self.measurement_model = measurement_model_NN(self.particle_encoder, self.obs_like_estimator, self.param)
        elif self.measurement == 'cos':
            self.measurement_model = measurement_model_cosine_distance(self.particle_encoder, self.param)

        self.prototype_density = compute_normal_density_disk(pos_noise=self.param.std_x, vel_noise=self.param.std_t)

        self.optim = torch.optim.AdamW(self.parameters(), lr=self.lr) #, weight_decay=1e-2
        # self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[30*(1+x) for x in range(10)], gamma=1.0)

class DPF_house3d(DPF_base):
    def __init__(self, args):
        super().__init__(args)
        self.param = args
        self.hidden_size = args.hiddensize
        self.state_dim =3
        self.param.state_dim = self.state_dim
        self.expanded_state_dim=4
        self.spatial = args.spatial
        self.action_dim=3
        self.batch_size = args.batchsize
        self.num_particle = args.num_particles
        self.labeledRatio = args.labeledRatio
        self.n_sequence = args.n_sequence
        self.init_var = 0.001
        self.NF_lr = args.NF_lr
        self.NF = args.NF_dyn
        self.NFcond = args.NF_cond
        self.measurement = args.measurement
        self.std_x = args.std_x
        self.std_y = args.std_y
        self.std_t = args.std_t
        self.lr = args.lr
        self.build_model()

        self.resampler = resampler(self.param)
        test_param=deepcopy(self.param)
        test_param.resampler_type = 'soft'
        test_param.alpha = 1.0
        self.resampler_test = resampler(test_param)

    def build_model(self):

        self.encoder = build_obs_feature_house3d(self.hidden_size, self.param.dropout_keep_ratio)
        self.encoder_flow = build_encoder_house3d(2, self.param.dropout_keep_ratio)
        # self.obs_feature = build_obs_feature_house3d(self.hidden_size, self.param.dropout_keep_ratio)

        # self.decoder = build_decoder_house3d(self.hidden_size)
        self.decoder = build_obs_decoder_house3d(self.hidden_size)

        self.map_encoder = build_map_encoder_house3d(self.hidden_size, self.param.dropout_keep_ratio)

        self.local_map_encoder = build_local_map_encoder_house3d(self.hidden_size, self.param.dropout_keep_ratio)
        # self.map_decoder = build_decoder_house3d(self.hidden_size)

        self.particle_encoder = build_particle_encoder_house3d(self.hidden_size, self.expanded_state_dim)

        # self.obs_like_estimator = build_likelihood_house3d(self.hidden_size, self.expanded_state_dim)

        self.mo_noise_generator = build_action_noise_house3d(self.action_dim)
        self.state_noise = [self.param.std_x, self.param.std_y, self.param.std_t]

        self.motion_update = motion_update_house3d(self.mo_noise_generator, self.state_noise)

        self.nf_dyn = build_conditional_nf(self.n_sequence, 2 * self.state_dim, self.state_dim, init_var=self.init_var)

        self.cond_model = build_conditional_nf(self.n_sequence, 2 * self.state_dim + self.hidden_size, self.state_dim, init_var=self.init_var)

        self.cnf_measurement = build_conditional_nf(self.n_sequence, self.hidden_size, self.hidden_size, init_var=0.1, prior_std=10.0)

        if self.measurement == 'CRNVP':
            self.measurement_model = measurement_model_cnf(self.particle_encoder, self.cnf_measurement, self.param)
        elif self.measurement=='NN':
            if self.spatial:
                # self.localconv1 = LocallyConnected2D(24, 8, output_size=(11, 11), kernel_size=(3, 3))
                # self.localconv2 = LocallyConnected2D(24, 8, output_size=(11, 11), kernel_size=(3, 3))
                self.likelihood_est = build_likelihood_house3d_spatial(self.hidden_size, self.state_dim)
                self.measurement_model = measurement_model_NN(self.local_map_encoder, self.likelihood_est, self.param)
            else:
                self.likelihood_est = build_likelihood_house3d(self.hidden_size, self.state_dim)
                self.measurement_model = measurement_model_NN(self.particle_encoder, self.likelihood_est, self.param)
        else:
            self.measurement_model = measurement_model_cosine_distance(self.local_map_encoder, self.param)
            # self.measurement_model = measurement_model_cosine_distance(self.particle_encoder, self.param)

        self.prototype_density = compute_normal_density_disk(pos_noise=self.param.std_x, vel_noise=self.param.std_t)

        # self.optim = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=4e-6, alpha=0.9)
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=4e-6)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[30*(1+x) for x in range(10)], gamma=1.0)
        # self.optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.5)