import torch
from torch import nn
import numpy as np
from utils import wrap_angle, normalize_log_probs
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def autoencoder_loss(image, train, encoder, decoder, stats, params):
    mse = nn.MSELoss()

    batch, seq, c, h, w = image.shape
    image_input = torch.reshape(image, (batch * seq, c, h, w))

    feature = encoder(image_input)
    recon_img = decoder(feature)

    loss = mse(recon_img, image_input)
    return loss

def weights_loss(particle_list, particle_weight_list, true_state):
    diff_matrix = (particle_list - true_state[..., None, :])[..., :2]
    kernel = torch.distributions.MultivariateNormal(torch.tensor([0.,0.]).to(device), torch.tensor([[200.,0.],[0.,200.]]).to(device))
    density_log = kernel.log_prob(diff_matrix)
    density_normalise = normalize_log_probs(density_log)

    loss_weight =torch.mean(torch.sum( (density_normalise - particle_weight_list).abs(), dim=-1) )

    return loss_weight, density_normalise

def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi

def supervised_loss_maze(state_step_sizes, particle_list, particle_weight_list, true_state, mask, train, labeledRatio=1.0):

    # prediction[..., -1] = wrap_angle(prediction[..., -1].clone())
    # loss = torch.sqrt(torch.mean((prediction - true_state) ** 2))  # Rooted mean square error
    state_dim = particle_list.shape[-1]
    prediction = torch.sum(particle_list * particle_weight_list[:, :, :, None],
                          dim=2)  # the dataset has extra initial state

    # prediction[:, :, 2] = wrap_angle(prediction[:, :, 2])
    diff = prediction - true_state[:,:,:state_dim]
    diff[:,:,2] = wrap_angle(diff[:,:,2])
    diff = diff / state_step_sizes
    if train:
        if labeledRatio > 0:
            loss = torch.sqrt( torch.mean(  torch.sum(mask[:,:,None]*diff ** 2, dim=-1)  )/ (mask.sum()/(mask.shape[0]*mask.shape[1]))) # Rooted mean square error
            if mask[:, -1].sum() > 0:
                loss_last = torch.sqrt( torch.mean(  (torch.sum(mask[:,:,None]*diff ** 2, dim=-1))[:,-1]  )/ (mask[:,-1].sum()/mask.shape[0])  )
            else:
                loss_last = torch.zeros_like(loss)
            return None, loss, loss_last, prediction
        elif labeledRatio == 0:
            return 0
    else:
        loss = torch.sqrt( torch.mean(torch.sum(diff ** 2, dim=-1)) )
        # loss_alltime = torch.sqrt(torch.mean((prediction - true_state[:, :, :state_dim]) ** 2, dim=-1))
        loss_alltime = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        loss_last = torch.sqrt(torch.mean( (torch.sum(diff ** 2, dim=-1)[:,-1] )))
        return loss_alltime, loss, loss_last, prediction

def supervised_loss_house(particle_list, particle_weight_list, true_state, mask, train, labeledRatio=1.0):

    # prediction[..., -1] = wrap_angle(prediction[..., -1].clone())
    # loss = torch.sqrt(torch.mean((prediction - true_state) ** 2))  # Rooted mean square error
    state_dim = particle_list.shape[-1]
    prediction = torch.sum(particle_list * particle_weight_list[:, :, :, None],
                           dim=2)  # the dataset has extra initial state
    if train:
        if labeledRatio > 0:
            pred_coord = torch.sum(particle_list[:, :, :, :2] * particle_weight_list[:, :, :, None],
                                   dim=2)  # the dataset has extra initial state
            true_coord = true_state[:, :, :2]
            coord_diffs = mask[:,:,None]*(pred_coord - true_coord)
            coord_diffs *= 0.02
            loss_coord = torch.sum(coord_diffs ** 2, dim=2) / (mask.sum()/(mask.shape[0]*mask.shape[1]))

            true_ori = true_state[:, :, 2]
            ori_diff = mask[:,:,None]*(particle_list[:, :, :, 2] - true_ori[:, :, None])
            ori_diff = (ori_diff + np.pi) % (2 * np.pi) - np.pi
            loss_ori = ((torch.sum(ori_diff * particle_weight_list, dim=2)) ** 2) / (mask.sum()/(mask.shape[0]*mask.shape[1]))
            loss_total = torch.mean(loss_coord + 0.36 * loss_ori)
            loss_report = torch.mean(loss_coord)
            return None, loss_total, loss_report, prediction
        elif labeledRatio == 0:
            return 0
    else:
        pred_coord = torch.sum(particle_list[:,:,:,:2] * particle_weight_list[:, :, :, None],
                              dim=2)  # the dataset has extra initial state
        true_coord = true_state[:, :, :2]
        coord_diffs = pred_coord - true_coord
        coord_diffs *= 0.02
        loss_coord = torch.sum(coord_diffs ** 2, dim=2)

        true_ori = true_state[:, :, 2]
        ori_diff = particle_list[:,:,:,2] - true_ori[:,:,None]
        ori_diff = (ori_diff + np.pi) % (2 * np.pi) - np.pi
        loss_ori = (torch.sum(ori_diff * particle_weight_list, dim=2)) ** 2
        loss_total = torch.mean(loss_coord + 0.36 * loss_ori)
        loss_report = loss_coord
        loss_alltime = torch.abs(torch.sum((prediction - true_state[:, :, :state_dim]), dim=-1))
        return loss_alltime, loss_total, loss_report, prediction

def supervised_loss(learnType, particle_list, particle_weight_list, true_state, mask, train, labeledRatio=1.0):

    # prediction[..., -1] = wrap_angle(prediction[..., -1].clone())
    # loss = torch.sqrt(torch.mean((prediction - true_state) ** 2))  # Rooted mean square error
    state_dim = true_state.shape[-1]
    if state_dim == 3:
        particle_list=torch.cat([
            particle_list[..., :2],  # normalized pos
            torch.cos(particle_list[..., 2:3]),  # cos
            torch.sin(particle_list[..., 2:3])],  # sin
            dim=-1)
        true_state = torch.cat([
            true_state[..., :2],  # normalized pos
            torch.cos(true_state[..., 2:3]),  # cos
            torch.sin(true_state[..., 2:3])],  # sin
            dim=-1)

    prediction = torch.sum(particle_list * particle_weight_list[:, :, :, None],
                          dim=2)  # the dataset has extra initial state

    loss = torch.sqrt( torch.mean(torch.sum((prediction - true_state) ** 2, dim=-1))) # Rooted mean square error
    loss_last = torch.sqrt( torch.mean( (torch.sum((prediction - true_state) ** 2, dim=-1)[:,-1])))
    if learnType == 'offline':
        return None, loss, loss_last, prediction
    if learnType == 'online':
        loss_alltime = torch.abs(torch.sum((prediction - true_state), dim=-1))
        return loss_alltime, loss, loss_last, prediction

    # else:
    #     loss = torch.sqrt( torch.mean(torch.sum((prediction - true_state) ** 2, dim=-1)))
    #     loss_alltime = torch.abs(torch.sum((prediction - true_state), dim=-1))
    #     loss_last = torch.sqrt(torch.mean( (torch.sum((prediction - true_state) ** 2, dim=-1)[:,-1] )))
    #     return loss_alltime, loss, loss_last, prediction

# def supervised_loss(particle_list, particle_weight_list, true_state, mask, train, labeledRatio=1.0):
#
#     # prediction[..., -1] = wrap_angle(prediction[..., -1].clone())
#     # loss = torch.sqrt(torch.mean((prediction - true_state) ** 2))  # Rooted mean square error
#     state_dim = particle_list.shape[-1]
#     if state_dim == 3:
#         particle_list=torch.cat([
#             particle_list[..., :2],  # normalized pos
#             torch.cos(particle_list[..., 2:3]),  # cos
#             torch.sin(particle_list[..., 2:3])],  # sin
#             dim=-1)
#         true_state = torch.cat([
#             true_state[..., :2],  # normalized pos
#             torch.cos(true_state[..., 2:3]),  # cos
#             torch.sin(true_state[..., 2:3])],  # sin
#             dim=-1)
#         state_dim = 4
#     elif state_dim == 2:
#         true_state = true_state.clone()[..., :2]
#     prediction = torch.sum(particle_list * particle_weight_list[:, :, :, None],
#                           dim=2)  # the dataset has extra initial state
#
#     if train:
#         if labeledRatio > 0:
#             loss = torch.sqrt( torch.mean(  torch.sum(mask[:,:,None]*(prediction - true_state[:,:,:state_dim]) ** 2, dim=-1)  )/ (mask.sum()/(mask.shape[0]*mask.shape[1])) ) # Rooted mean square error
#             if mask[:, -1].sum() > 0:
#                 loss_last = torch.sqrt( torch.mean(  (torch.sum(mask[:,:,None]*(prediction - true_state[:,:,:state_dim]) ** 2, dim=-1))[:,-1]  )/ (mask[:,-1].sum()/mask.shape[0])  )
#             else:
#                 loss_last = torch.zeros_like(loss)
#             return None, loss, loss_last, prediction
#         elif labeledRatio == 0:
#             return 0
#     else:
#         loss = torch.sqrt( torch.mean(torch.sum((prediction - true_state[:,:,:state_dim]) ** 2, dim=-1)))
#         # loss_alltime = torch.sqrt(torch.mean((prediction - true_state[:, :, :state_dim]) ** 2, dim=-1))
#         loss_alltime = torch.abs(torch.sum((prediction - true_state[:,:,:state_dim]), dim=-1))
#         loss_last = torch.sqrt(torch.mean( (torch.sum((prediction - true_state[:,:,:state_dim]) ** 2, dim=-1)[:,-1] )))
#         return loss_alltime, loss, loss_last, prediction

def pseudolikelihood_loss_nf(particle_weight_list, noise_list, likelihood_list, index_list, jac_list, prior_list, block_len=10):

    return -1. * torch.mean(compute_block_density_nf(particle_weight_list, noise_list, likelihood_list, index_list, jac_list, prior_list, block_len))

def compute_block_density_nf(particle_weight_list, noise_list, likelihood_list, index_list, jac_list, prior_list, block_len=10):
    batch_size, seq_len, num_resampled = particle_weight_list.shape

    # log_mu_s shape: (batch_size, num_particle)
    # block index
    b =0
    # pseudo_likelihood
    Q =0
    logyita = 0
    log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi))
    for k in range(seq_len):
        if (k+1)% block_len==0:
            for j in range(k, k-block_len, -1):
                if j == k:
                    lik_log = likelihood_list[:,j,:]
                    index_a = index_list[:,j,:]
                    jac_log = jac_list[:,j,:]
                    prior_ = prior_list[:, j, :]
                else:
                    lik_log = likelihood_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]
                    jac_log = jac_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]
                    prior_ = prior_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]

                    index_pre = index_list[:, j, :]
                    index_a = index_pre.reshape((batch_size * num_resampled,))[index_a]

                log_prior = prior_

                logyita = logyita + log_prior + lik_log
            Q = Q + torch.sum(particle_weight_list[:, k, :] * logyita, dim=-1)
            b = b+1
    # Q shape: (batch_size,)
    return Q/b


def compute_block_density(particle_weight_list, noise_list, likelihood_list, index_list, block_len=10, std_pos=1.0, std_vel=1.0):
    batch_size, seq_len, num_resampled = particle_weight_list.shape

    # log_mu_s shape: (batch_size, num_particle)
    # block index
    b =0
    # pseudo_likelihood
    Q =0
    logyita = 0
    log_c = - 0.5 * torch.log(torch.tensor(2 * np.pi))
    for k in range(seq_len):
        if (k+1)% block_len==0:
            for j in range(k, k-block_len, -1):
                if j == k:
                    lik = likelihood_list[:,j,:]
                    index_a = index_list[:,j, :]
                    noise_pos = noise_list[:, j, :, :2]
                    noise_vel = noise_list[:, j, :, 2:]
                else:
                    lik = likelihood_list[:,j,:].reshape((batch_size * num_resampled,))[index_a]
                    noise_pos = noise_list[:, j, :, :2].reshape((batch_size * num_resampled,-1))[index_a,:]
                    noise_vel = noise_list[:, j, :, 2:].reshape((batch_size * num_resampled, -1))[index_a, :]
                    index_pre = index_list[:, j, :]
                    index_a = index_pre.reshape((batch_size * num_resampled,))[index_a]

                log_prior = (2 * log_c -  2*torch.log(torch.tensor(std_pos)) - torch.sum(
                    noise_pos ** 2 / (2 * torch.tensor(std_pos) ** 2), dim=-1)) +\
                            (2 * log_c -  2*torch.log(torch.tensor(std_vel)) - torch.sum(
                            noise_vel ** 2 / (2 * torch.tensor(std_vel) ** 2), dim=-1))

                logyita = logyita + log_prior + lik
            Q = Q + torch.sum(particle_weight_list[:, k, :] * logyita, dim=-1)
            b = b+1
    # Q shape: (batch_size,)
    return Q/b



def pseudolikelihood_loss(particle_weight_list, noise_list, likelihood_list, index_list, block_len=10, std_pos=1.0, std_vel=1.0):

    return -1. * torch.mean(compute_block_density(particle_weight_list, noise_list, likelihood_list, index_list, block_len, std_pos, std_vel))