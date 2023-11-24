import torch
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def autoencoder_loss(encoder, decoder, image, means, stds):
    means_o = torch.tensor(means['o']).permute(0, 1, 4, 2, 3).to(device)
    stds_o = torch.tensor(stds['o']).permute(0, 1, 4, 2, 3).to(device)

    batch, seq, c, h, w = image.shape
    image_transform = (image - means_o) / stds_o
    image_input = torch.reshape(image_transform, (batch * seq, c, h, w))

    loss = torch.mean((decoder(encoder(image_input.float())) - image_input) ** 2)

    return loss