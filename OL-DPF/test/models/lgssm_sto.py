import copy
import aesmc
import numpy as np
import pykalman
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import sys
sys.path.append('../test/')

class Initial(nn.Module):
    def __init__(self, loc, scale):
        super(Initial, self).__init__()
        self.loc = loc
        self.scale = scale
        self.dim = self.loc.shape[0] if len(self.loc.shape)!= 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal
    def forward(self):
            return self.dist(self.loc, self.scale)


class Transition(nn.Module):
    def __init__(self, init_mult, init_scale):
        super(Transition, self).__init__()
        self.mult = nn.Parameter(init_mult.squeeze().clone())#init_mult##
        self.scale = nn.Parameter(init_scale.squeeze().clone())#init_scale##
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
            self.mult = self.mult
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        # mult = torch.diag(self.mult if self.dim != 1 else self.mult.unsqueeze(-1)).squeeze()
        loc = 0.5 * previous_latents[-1] + 0.5 * self.mult
        return aesmc.state.set_batch_shape_mode(
            self.dist(loc, self.scale),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)


class Emission(nn.Module):
    def __init__(self, mult):
        super(Emission, self).__init__()
        self.mult = mult
        self.dim = self.mult.shape[0] if len(self.mult.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

    def forward(self, latents=None, time=None, previous_observations=None):
        loc = torch.diag(self.mult.unsqueeze(-1)).squeeze()
        scale = (torch.exp(latents[-1]))**0.5
        return aesmc.state.set_batch_shape_mode(
                self.dist(loc, scale),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)
class Proposal(nn.Module):
    def __init__(self, scale_0, scale_t, device):
        super(Proposal, self).__init__()
        self.scale_0_vector = nn.Parameter(scale_0.squeeze().clone())#scale_0#
        self.scale_t_vector = nn.Parameter(scale_t.squeeze().clone())#scale_t#

        self.dim = self.scale_0_vector.shape[0] if len(self.scale_0_vector.shape) != 0 else 1

        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

        self.lin_0 = nn.Linear(self.dim, self.dim, bias=False).to(device)
        self.lin_t = nn.Linear(2*self.dim, self.dim, bias=False).to(device)

    def forward(self, previous_latents=None, time=None, observations=None):
        self.scale_0 = torch.nn.functional.softplus(torch.diag(self.scale_0_vector if self.dim != 1 else self.scale_0_vector.unsqueeze(-1)).squeeze())
        # self.scale_t = torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze()
        self.scale_t = torch.nn.functional.softplus(torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze())

        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                self.dist(self.lin_0(observations[0].unsqueeze(-1) if self.dim ==1 else observations[0]).squeeze(-1), self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:
            num_particles = previous_latents[-1].shape[1]
            a=self.lin_t(torch.cat(
                            [previous_latents[-1].unsqueeze(-1) if self.dim ==1 else previous_latents[-1],
                             (observations[time].view(-1, 1, 1) if self.dim ==1 else observations[time].unsqueeze(1)).repeat(1, num_particles, 1)],
                            dim=2
                        ).view(-1, 2*self.dim)).squeeze(-1).view((-1, num_particles) if self.dim ==1 else (-1, num_particles, self.dim))
            # mask1 = (a < 0)
            # a[mask1] = 0.0
            return aesmc.state.set_batch_shape_mode(
                self.dist(
                        a,
                        self.scale_t),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)

class Proposal_rnn(nn.Module):
    def __init__(self, scale_0, scale_t, device):
        super(Proposal_rnn, self).__init__()
        self.scale_0_vector = nn.Parameter(scale_0.squeeze().clone())#scale_0#
        self.scale_t_vector = nn.Parameter(scale_t.squeeze().clone())#scale_t#

        self.dim = self.scale_0_vector.shape[0] if len(self.scale_0_vector.shape) != 0 else 1

        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

        self.lin_0 = nn.Linear(self.dim, self.dim, bias=False).to(device)
        self.rnn_cell = nn.GRUCell(self.dim, self.dim) #nn.Linear(2*self.dim, self.dim, bias=False).to(device)
        self.lin_t = nn.Linear(self.dim, self.dim, bias=False).to(device)
    def forward(self, previous_latents=None, time=None, observations=None):
        self.scale_0 = torch.diag(self.scale_0_vector if self.dim != 1 else self.scale_0_vector.unsqueeze(-1)).squeeze()
        self.scale_t = torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze()

        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                self.dist(self.lin_0(observations[0].unsqueeze(-1) if self.dim ==1 else observations[0]).squeeze(-1), self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:
            num_particles = previous_latents[-1].shape[1]

            previous_latent_input = previous_latents[-1].unsqueeze(-1).reshape(-1, self.dim) if self.dim == 1 else previous_latents[-1].reshape(-1, self.dim)
            observation_input = (observations[time].view(-1, 1, 1) if self.dim == 1 else observations[time].unsqueeze(1)).repeat(1,num_particles,1).reshape(-1, self.dim)
            rnn_output = self.rnn_cell(previous_latent_input,observation_input).squeeze(-1).view(-1, num_particles, self.dim)
            rnn_output = self.lin_t(rnn_output).reshape(-1, num_particles) if self.dim == 1 else self.lin_t(rnn_output)

            return aesmc.state.set_batch_shape_mode(
                self.dist(rnn_output,self.scale_t),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)

class Proposal_cnf(nn.Module):
    def __init__(self, transition, initial, scale_0, scale_t, device, type = 'planar', k=2):
        super(Proposal_cnf, self).__init__()
        self.scale_0_vector = nn.Parameter(scale_0.squeeze().clone())#scale_0#
        self.scale_t_vector = nn.Parameter(scale_t.squeeze().clone())#scale_t#

        self.transition = transition
        self.initial = initial

        self.dim = self.scale_0_vector.shape[0] if len(self.scale_0_vector.shape) != 0 else 1
        if self.dim == 1:
            self.dist = torch.distributions.Normal
        else:
            self.dist = torch.distributions.multivariate_normal.MultivariateNormal

        self.lin_0 = nn.Linear(self.dim, self.dim, bias=False).to(device)
        self.lin_t = nn.Linear(2*self.dim, self.dim, bias=False).to(device)

        self.type_flow = type
        if self.type_flow == 'planar':
            self.k = k
            if self.dim == 1:
                self.planar = Planar_1d_compose(k=self.k, device=device)
            else:
                self.planar = Planar_compose(k=self.k, dim=self.dim, device=device)
        elif self.type_flow == 'radial':
            self.radial_flow = Radial(dim = 1)
        elif self.type_flow == 'nvp':
            self.k = k
            self.nvp = RealNVP_cond_compose(k=k, dim = self.dim, hidden_dim= 2 * self.dim, obser_dim = self.dim, device=device)

    def sample(self, previous_latents=None, time=None, observations=None, batch_size = 10, num_particles = 100):
        self.scale_0 = torch.diag(self.scale_0_vector if self.dim != 1 else self.scale_0_vector.unsqueeze(-1)).squeeze()
        self.scale_t = torch.diag(self.scale_t_vector if self.dim != 1 else self.scale_t_vector.unsqueeze(-1)).squeeze()

        if self.type_flow != 'bootstrap':
            if time == 0:
                loc = self.lin_0(observations[0].unsqueeze(-1) if self.dim ==1 else observations[0]).squeeze(-1)

                dist_0 = aesmc.state.set_batch_shape_mode(self.dist(loc,self.scale_0),
                                                          aesmc.state.BatchShapeMode.BATCH_EXPANDED)
                samples = aesmc.state.sample(dist_0, batch_size, num_particles)
                proposal_log_prob = aesmc.state.log_prob(dist_0, samples)
                return samples, proposal_log_prob
            else:
                loc = self.lin_t(torch.cat(
                        [previous_latents[-1].unsqueeze(-1) if self.dim ==1 else previous_latents[-1],
                         (observations[time].view(-1, 1, 1) if self.dim ==1 else observations[time].unsqueeze(1)).repeat(1, num_particles, 1)],
                        dim=2
                    ).view(-1, 2*self.dim)).squeeze(-1).view((-1, num_particles) if self.dim ==1 else (-1, num_particles, self.dim))
                dist_t = aesmc.state.set_batch_shape_mode(self.dist(loc,self.scale_t),
                                                          aesmc.state.BatchShapeMode.FULLY_EXPANDED)
                proposal_samples = aesmc.state.sample(dist_t, batch_size, num_particles)
                proposal_log_prob = aesmc.state.log_prob(dist_t, proposal_samples)

                if self.type_flow == 'planar':
                    proposal_samples, log_det = self.planar(proposal_samples, observations,time, previous_latents=previous_latents)
                elif self.type_flow == 'nvp':
                    proposal_samples, log_det = self.nvp(proposal_samples,observations,time)
                elif self.type_flow == 'radial':
                    proposal_samples, log_det = self.radial_flow(proposal_samples)
                elif self.type_flow == 'normal':
                    log_det = 0.0
                else:
                    raise ValueError('Please select a type from {planar, radial, normal, bootstrap}.')
                proposal_log_prob = proposal_log_prob - log_det.squeeze(-1)
        elif self.type_flow == 'bootstrap':
            if time == 0:
                initial_samples = aesmc.state.sample(self.initial(), batch_size, num_particles)
                initial_log_prob = aesmc.state.log_prob(self.initial(), initial_samples)
                return initial_samples, initial_log_prob
            else:
                transition_dist = self.transition(previous_latents=previous_latents)
                proposal_samples = aesmc.state.sample(transition_dist, batch_size, num_particles)
                proposal_log_prob = aesmc.state.log_prob(
                    self.transition(previous_latents=previous_latents), proposal_samples)
        else:
            raise ValueError('Please select a type from {planar, radial, normal, bootstrap}.')
        return proposal_samples, proposal_log_prob



class FCNN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )


    def forward(self, x):
        return self.network(x.float())

functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor).to(x.device) + \
                            (x < 0).type(torch.FloatTensor).to(x.device) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor).to(x.device) + \
                     (x < 0).type(torch.FloatTensor).to(x.device) * torch.exp(x)
}

class Planar_1d_compose(nn.Module):
    def __init__(self, k=1, device='cuda'):
        super(Planar_1d_compose, self).__init__()
        self.flows = nn.Sequential(*[Planar_1d(device) for _ in range(k)])
    def forward(self, proposal_samples, observations, time, previous_latents = None):
        log_det = 0
        for flow in self.flows:
            proposal_samples, log_det_k = flow(proposal_samples, observations, time)
            log_det = log_det + log_det_k
        return  proposal_samples, log_det

class Planar_1d(nn.Module):
    def __init__(self, device='cuda'):
        super(Planar_1d, self).__init__()
        self.u = nn.Parameter(torch.tensor(0.1).to(device))
        self.b = nn.Parameter(torch.tensor(0.1).to(device))
        self.w = nn.Parameter(torch.tensor(0.1).to(device))
        self.reset_parameters()
    def reset_parameters(self):
        init.uniform_(self.w, 0.0, 0.25)
        init.uniform_(self.u, 0.0, 0.25)
        init.uniform_(self.b, 0.0, 0.25)
    def forward(self, proposal_samples, observations, time, previous_latents = None):
        num_particles = proposal_samples.shape[-1]
        proposal_samples = proposal_samples + self.u * torch.tanh(self.w * proposal_samples +
                                                                  self.b * observations[time].view(-1, 1).expand(-1,
                                                                                                                 num_particles))
        log_det = (1 + self.u * self.w * (1 - torch.tanh(self.w * proposal_samples +
                                                         self.b * observations[time].view(-1, 1).expand(-1,
                                                                                                        num_particles)) ** 2)).abs().log()
        return proposal_samples, log_det

class Planar_compose(nn.Module):
    def __init__(self, k=1, dim = 2, device='cuda'):
        super(Planar_compose, self).__init__()
        self.flows = nn.Sequential(*[Planar(dim=dim, device=device).to(device) for _ in range(k)])
    def forward(self, proposal_samples, observations, time, previous_latents = None):
        log_det = 0
        for flow in self.flows:
            proposal_samples, log_det_k = flow(proposal_samples, observations, time, previous_latents = previous_latents)
            log_det = log_det + log_det_k
        return proposal_samples, log_det

class Planar(nn.Module):
    def __init__(self, dim, nonlinearity=torch.tanh, device='cuda'):
        super().__init__()
        self.h = nonlinearity
        self.w = nn.Parameter(torch.Tensor(dim))
        self.u = nn.Parameter(torch.Tensor(dim))
        self.b = nn.Parameter(torch.Tensor(dim))

        self.lin_b = nn.Linear(dim, dim)

        self.reset_parameters(dim)
        self.dim = dim


    def reset_parameters(self, dim):
        # init.constant_(self.w, 0.1)
        # init.constant_(self.u, 0.1)
        # init.constant_(self.b, 0.1)
        init.uniform_(self.w, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.u, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.b, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x, observation, time, previous_latents =None):

        observation = torch.cat([observation[time].unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
        # if self.h in (F.elu, F.leaky_relu):
        #     u = self.u
        # elif self.h == torch.tanh:
        #     scal = torch.log(1+torch.exp(self.w @ self.u)) - self.w @ self.u - 1
        #     u = self.u + scal * self.w / torch.norm(self.w) ** 2
        # else:
        #     raise NotImplementedError("Non-linearity is not supported.")
        u = self.u
        lin = torch.unsqueeze( x @ self.w, -1) + (self.b * self.lin_b(observation)).sum(-1, keepdims=True)
        z = x + u * self.h(lin)
        phi = functional_derivatives[self.h](lin) * self.w
        log_det = torch.log(torch.abs(1 + phi @ u) + 1e-4)
        return z.squeeze(-1), log_det


class Radial(nn.Module):
    """
    Radial flow.
        z = f(x) = = x + β h(α, r)(z − z0)
    [Rezende and Mohamed 2015]
    """
    def __init__(self, dim):
        super().__init__()
        self.x0 = nn.Parameter(torch.Tensor(dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

    def reset_parameters(self, dim):
        init.uniform_(self.x0, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.log_alpha, -math.sqrt(1/dim), math.sqrt(1/dim))
        init.uniform_(self.beta, -math.sqrt(1/dim), math.sqrt(1/dim))

    def forward(self, x):
        """
        Given x, returns z and the log-determinant log|df/dx|.
        """
        m, n = x.shape
        r = torch.norm(x - self.x0)
        h = 1 / (torch.exp(self.log_alpha) + r)
        beta = -torch.exp(self.log_alpha) + torch.log(1 + torch.exp(self.beta))
        z = x + beta * h * (x - self.x0)
        log_det = (n - 1) * torch.log(1 + beta * h) + \
                  torch.log(1 + beta * h - \
                            beta * r / (torch.exp(self.log_alpha) + r) ** 2)
        return z, log_det

class RealNVP_cond_compose(nn.Module):
    def __init__(self, k, dim, hidden_dim, obser_dim, device):
        super(RealNVP_cond_compose, self).__init__()
        self.flows = nn.Sequential(*[RealNVP_cond_t(dim = dim,
                                                    hidden_dim = hidden_dim,
                                                    obser_dim = obser_dim).to(device) for _ in range(k)])
        for flow in self.flows:
            flow.zero_initialization(0.1)
    def forward(self, proposal_samples, observations, time):
        log_det = 0
        for flow in self.flows:
            proposal_samples, log_det_k = flow(proposal_samples, observations, time)
            log_det = log_det + log_det_k
        return proposal_samples, log_det

class RealNVP_cond_t(nn.Module):

    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, obser_dim=None):
        super().__init__()
        self.dim = dim
        self.dim_1 = self.dim - dim//2
        self.dim_2 = self.dim//2
        self.obser_dim=obser_dim
        self.t1 = base_network(self.dim_1+self.obser_dim, self.dim_2, hidden_dim)
        self.s1 = base_network(self.dim_1+self.obser_dim, self.dim_2, hidden_dim)
        self.t2 = base_network(self.dim_2+self.obser_dim, self.dim_1, hidden_dim)
        self.s2 = base_network(self.dim_2+self.obser_dim, self.dim_1, hidden_dim)
    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, obser, time):
        x_mean, x_std = x.mean(dim=1,keepdim=True), x.std(dim=1,keepdim=True)
        # x = (x-x_mean)/x_std
        obser = obser[time].unsqueeze(1).repeat(1, x.shape[1], 1)
        lower, upper = x[..., :self.dim_1], x[..., self.dim_1:]
        t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
        s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
        upper = t1_transformed + upper #* torch.exp(s1_transformed)
        t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
        s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
        lower = t2_transformed + lower #* torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=-1)
        # z = z*x_std+x_mean
        # log_det = torch.sum(s1_transformed, dim=-1) + \
        #           torch.sum(s2_transformed, dim=-1)
        log_det = torch.zeros_like(torch.sum(s2_transformed, dim=-1))
        return z, log_det

def lgssm_true_posterior(observations, initial_loc, initial_scale,
                         transition_mult, transition_bias, transition_scale,
                         emission_mult, emission_bias, emission_scale):
    ssm_parameter_lists = [initial_loc, initial_scale,
    transition_mult, transition_scale,
    emission_mult, emission_scale,
    transition_bias, emission_bias]

    initial_loc, initial_scale, \
    transition_mult, transition_scale, \
    emission_mult, emission_scale,\
    transition_bias, emission_bias = [ssm_parameter.cpu().detach().numpy() for
                                                    ssm_parameter in ssm_parameter_lists if torch.is_tensor(ssm_parameter)]
    dim = initial_loc.shape[0] if len(initial_loc.shape) != 0 else 1

    if dim == 1:
        kf = pykalman.KalmanFilter(
            initial_state_mean=[initial_loc],
            initial_state_covariance=[[initial_scale**2]],
            transition_matrices=[[transition_mult]],
            transition_offsets=[transition_bias],
            transition_covariance=[[transition_scale**2]],
            observation_matrices=[[emission_mult]],
            observation_offsets=[emission_bias],
            observation_covariance=[[emission_scale**2]])
    else:
        kf = pykalman.KalmanFilter(
            initial_state_mean=initial_loc,
            initial_state_covariance=initial_scale ** 2,
            transition_matrices=transition_mult,
            transition_offsets=transition_bias,
            transition_covariance=transition_scale ** 2,
            observation_matrices=emission_mult,
            observation_offsets=emission_bias,
            observation_covariance=emission_scale ** 2)

    return kf.smooth(torch.stack([observation[0].cpu().squeeze() for observation in observations], dim = 0).numpy())


class TrainingStats(object):
    def __init__(self, initial_loc, initial_scale, true_transition_mult,
                 true_transition_scale, emission_mult,
                 num_timesteps, num_test_obs, test_inference_num_particles,
                 saving_interval=100, logging_interval=100, algorithm='is',args=None, num_iterations=500):
        device = args.device
        self.dim = 1 if len(initial_loc.shape)==0 else initial_loc.shape[0]
        self.true_transition_mult = true_transition_mult
        self.true_transition_scale = true_transition_scale
        self.test_inference_num_particles = test_inference_num_particles
        self.saving_interval = saving_interval
        self.logging_interval = logging_interval
        self.p_l2_history = []
        self.normalized_log_weights_history = []
        self.iteration_idx_history = []
        self.loss_history = []
        self.initial = Initial(initial_loc, initial_scale).to(device)
        # self.true_transition = Transition(true_transition_mult,
        #                                   transition_scale).to(device)
        # self.true_emission = Emission(true_emission_mult, emission_scale).to(device)
        self.true_transition = Transition(torch.tensor(0.3),
                                          torch.tensor(0.8)).to(device)
        self.true_emission = Emission(emission_mult).to(device)
        dataloader = aesmc.train.get_synthetic_dataloader(self.initial,
                                                          self.true_transition,
                                                          self.true_emission,
                                                          num_timesteps,
                                                          num_test_obs)
        self.test_obs = next(iter(dataloader))
        for test_obs_idx in range(num_test_obs):
            latent = [[l[test_obs_idx]] for l in self.test_obs[0]]
            observations = [[o[test_obs_idx]] for o in self.test_obs[1]]

        #test stage
        self.true_transition_test = Transition(torch.tensor(0.3),
                                          torch.tensor(0.8)).to(device)
        self.true_emission_test = Emission(emission_mult).to(device)
        dataloader_test = aesmc.train.get_synthetic_dataloader(self.initial,
                                                          self.true_transition_test,
                                                          self.true_emission_test,
                                                          num_timesteps,
                                                          num_test_obs)
        self.test_obs_test = next(iter(dataloader_test))

        if algorithm == 'iwae':
            self.algorithm = 'is'
        else:
            self.algorithm = 'smc'

        self.args = args
        self.device = args.device
        self.num_iterations = num_iterations
    def __call__(self, epoch_idx, epoch_iteration_idx, loss, initial,
                 transition, emission, proposal, test=False, online=False):
        if test == True:
            latents_test=self.test_obs_test[0]
            latents_test = [latent_test.to(self.device).unsqueeze(-1) if len(latent_test.shape) == 1 else latent_test.to(self.device)
                       for latent_test in latents_test]
            inference_result = aesmc.inference.infer(
                self.algorithm, self.test_obs_test[1], self.initial,
                transition, emission, proposal,
                self.test_inference_num_particles, args=self.args, true_latents=latents_test,
                return_log_marginal_likelihood=True)
            loss_rmse_test=inference_result['loss_report'].cpu().detach().numpy()
            print('loss_rmse_test:', loss_rmse_test)
            return loss_rmse_test


        if epoch_iteration_idx % self.saving_interval == 0 or epoch_iteration_idx + 1 == self.num_iterations:
            print('theta1:', transition.mult.cpu().detach().numpy(), 'theta2:', transition.scale.cpu().detach().numpy())
            if online == False:
                self.p_l2_history.append(np.linalg.norm(
                    np.array([transition.mult.flatten().cpu().detach().numpy(), transition.scale.cpu().detach().numpy()],dtype=object)-np.array([self.true_transition_mult.cpu().detach().numpy(), self.true_transition_scale.cpu().detach().numpy()],dtype=object)
                ).squeeze())
            else:
                self.p_l2_history.append(np.linalg.norm(
                    np.array(
                        [transition.mult.flatten().cpu().detach().numpy(), transition.scale.cpu().detach().numpy()],
                        dtype=object) -
                                          np.array([0.3,
                                                    0.8], dtype=object)
                ).squeeze())
            latents = self.test_obs[0]
            latents = [latent.to(self.device).unsqueeze(-1) if len(latent.shape)==1 else latent.to(self.device)
                       for latent in latents]
            inference_result = aesmc.inference.infer(
                self.algorithm, self.test_obs[1], self.initial,
                transition, emission, proposal,
                self.test_inference_num_particles,args=self.args, true_latents=latents,return_log_marginal_likelihood=True)
            normalized_weights = aesmc.math.normalize_log_probs(torch.stack(inference_result['log_weights'],dim=0))+1e-8
            self.normalized_log_weights_history.append(normalized_weights.cpu().detach().numpy())
            self.loss_history.append(inference_result['loss_report'].cpu().detach().numpy())
            self.iteration_idx_history.append(epoch_iteration_idx)

        if epoch_iteration_idx % self.logging_interval == 0 or epoch_iteration_idx + 1 == self.num_iterations:
            print('Iteration {}:'
                  ' Loss = {:.3f},'
                  ' parameter error = {:.6f},'
                  .format(epoch_iteration_idx,inference_result['loss_report'],
                          self.p_l2_history[-1],))
