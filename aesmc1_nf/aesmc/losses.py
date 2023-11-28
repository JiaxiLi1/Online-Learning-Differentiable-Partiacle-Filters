from . import inference
import torch


def get_loss(online_data, training_stage, observations, num_particles, algorithm, initial, transition,
             emission, proposal, args=None, true_latents=None):
    """Returns a differentiable loss for gradient descent.

    Args:
        observations: list of tensors [batch_size, dim1, ..., dimN] or
            dicts thereof
        num_particles: int
        algorithm: iwae or aesmc
        initial: a callable object (function or nn.Module) which has no
            arguments and returns a torch.distributions.Distribution or a dict
            thereof
        transition: a callable object (function or nn.Module) with signature:
            Args:
                previous_latents: list of length time where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                previous_observations: list of length time where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof
        emission: a callable object (function or nn.Module) with signature:
            Args:
                true_latents: list of length (time + 1) where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                previous_observations: list of length time where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof
        proposal: a callable object (function or nn.Module) with signature:
            Args:
                previous_latents: list of length time where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                observations: list of length num_timesteps where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    if algorithm == 'iwae':
        inference_algorithm = 'is'
    else:
        inference_algorithm = 'smc'

    if training_stage == 'online':
        online_learning = True
    else:
        online_learning = False
    inference_result, data_pre_step = inference.infer(online_data=online_data,
                                        inference_algorithm=inference_algorithm,
                                       observations=observations,
                                       initial=initial,
                                       transition=transition,
                                       emission=emission,
                                       proposal=proposal,
                                       num_particles=num_particles,
                                       return_log_marginal_likelihood=True,
                                       return_latents=False,
                                       return_original_latents=True,
                                       return_log_weight=True,
                                       return_log_weights=True,
                                       return_ancestral_indices=False,
                                       args = args,
                                       true_latents= true_latents,
                                       online_learning=online_learning)
    elbo = inference_result['log_marginal_likelihood']
    loss_rmse = inference_result['loss_rmse']
    loss = -1e-2*torch.mean(elbo)
    # print(loss_rmse)
    if training_stage == 'offline':
        return loss_rmse, data_pre_step

    if args.trainType == 'DPF':
        return loss_rmse, data_pre_step
    elif args.trainType == 'SDPF_elbo':
        return loss, data_pre_step
    else:
        raise ValueError('Please select an algorithm from different DPF.')
