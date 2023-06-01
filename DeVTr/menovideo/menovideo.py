"""
Part of the code adapted from:
Almamon Rasool Abdali, DeVTr,
https://github.com/mamonraab/Data-efficient-video-transformer/

MIT License
Copyright (c) 2021
"""

import math
import timm
import torch

from torch import nn


class TimeWarp(nn.Module):
    def __init__(self, model):
        super(TimeWarp, self).__init__()
        self.model = model
 
    def forward(self, x):
        # [batch_size, time_steps, color_channels, height, width]
        _, time_steps, _, _, _ = x.size()
        output = []
        for frame in range(time_steps):
            x_t = self.model(x[:, frame, :, :, :])
            output.append(x_t)

        # Make output as (samples, time_steps, output_size)
        # Stacks the output list tensors along a new dimension at position 0
        # and swaps the position of dimensions 0 and 1
        x = torch.stack(output, dim=0).transpose_(0, 1)

        output = None
        x_t = None

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dimension, dropout=0.1, time_steps=40):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dimension = embedding_dimension
        self.time_steps = time_steps

    def do_pos_encode(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pe = torch.zeros(self.time_steps, self.embedding_dimension).to(device)
        for pos in range(self.time_steps):
            for i in range(0, self.embedding_dimension, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embedding_dimension)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embedding_dimension)))
        pe = pe.unsqueeze(0) # to make shape of (batch size, time steps, embding_dim)
        return pe

    def forward(self, x):
        # x here is embded data must be shape of (batch , time_steps , embding_dim)
        x = x * math.sqrt(self.embedding_dimension)
        pe = self.do_pos_encode()
        x += pe[:, :x.size(1)]   # pe will automatically be expanded with the same batch size as encoded_words
        x = self.dropout(x)
        return x


class memoTransormer(nn.Module):
    def __init__(self, dim, heads=8, layers=6, actv='gelu'):
        super(memoTransormer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, activation=actv)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x


def DeVTr(
    weights=None,                       # The path for pre-trained DeVTr model
    number_of_neurons=1024,             # Number of neurons of the first layer that goes after the output of the encoder
    classification_dropout_rate=0.4,    # Dropout rate of the classification network
    number_of_output_classes=1,         # Number of output classes (Violence or Non Violence)
    embedding_dimension=512,            # Number of output dimensions of the CNN network
    encoder_dropout_rate=0.1,           # Dropout rate of the transformer encoder
    number_of_frames=40,                # Number of frames of the input video
    encoder_layers=4,                   # Number of transformer encoder layers
    encoder_heads=8                     # Number of transformer encoder heads per layer
):
    # If the weights of the pre-trained DeVTr model are passed, default values will be used
    if weights:
        number_of_output_classes = 1
        encoder_dropout_rate = 0.1
        embedding_dimension = 512
        encoder_layers = 4
        encoder_heads = 8
        number_of_frames = 40

    # Creates the VGG-19 pre-trained network with batch normalization
    # The model is used to extract features of dimension 'embedding_dimension'
    vgg_19_model = timm.create_model('vgg19_bn.tv_in1k', pretrained=True, num_classes=embedding_dimension)

    # To freeze the first 40 layers of the model
    # This is because the initial layers usually contain more general and reusable
    # features that can be useful in various computer vision tasks
    # It seems that there are 53 layers
    i = 0
    for child in vgg_19_model.features.children():
        # To disable the calculation of gradients and freezes the layer parameters,
        # which means they will not be updated during training
        if i < 40:
            for param in child.parameters():
                param.requires_grad = False
        # Enables the calculation of gradients and allows the parameters of these layers
        # to be updated during training
        else:
            for param in child.parameters():
                param.requires_grad = True
        i += 1

    # Combines the VGG-19 network with a non-linear activation layer
    # ReLU(x) = max(0, x)
    embedding_network = nn.Sequential(vgg_19_model, nn.ReLU())

    final_model = nn.Sequential(
        TimeWarp(embedding_network, method='loop', flatn=False),
        PositionalEncoder(embedding_dimension, dropout=encoder_dropout_rate, time_steps=number_of_frames),
        memoTransormer(embedding_dimension, heads=encoder_heads, layers=encoder_layers, actv='gelu'),
        nn.Flatten(),
        #20480 is frame numbers * dim
        nn.Linear(number_of_frames * embedding_dimension, number_of_neurons),
        nn.Dropout(classification_dropout_rate),
        nn.ReLU(),
        nn.Linear(number_of_neurons, number_of_output_classes),
    )

    if weights != 'none':
        if torch.cuda.is_available():
            final_model.load_state_dict(torch.load(weights))
        else:
            final_model.load_state_dict(torch.load(weights, map_location ='cpu'))

    return final_model
