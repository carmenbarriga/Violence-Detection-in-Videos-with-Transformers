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
    def __init__(self, model, method='squeeze', flatn=True):
        super(TimeWarp, self).__init__()
        self.model = model
        self.method = method
        self.flatn = flatn
 
    def forward(self, x):
        batch_size, time_steps, color_channels, height, width = x.size()
        if self.method == 'loop':
            output = []
            for frame in range(time_steps):
                # Input one frame at a time into the model
                x_t = self.model(x[:, frame, :, :, :])
                # Flatten the output
                if self.flatn:
                    x_t = x_t.view(x_t.size(0), -1)
                output.append(x_t)

            # Make output as (samples, timesteps, output_size)
            x = torch.stack(output, dim=0).transpose_(0, 1)
            output = None
            x_t = None
        else:
            # reshape input  to be (batch_size * timesteps, input_size)
            x = x.contiguous().view(batch_size * time_steps, color_channels, height, width)
            x = self.model(x)
            if self.flatn:
                x = x.view(x.size(0), -1)
            #make output as  ( samples, timesteps, output_size)
            x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, embd_dim, dropout=0.1, time_steps=30):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embd_dim = embd_dim
        self.time_steps = time_steps

    def do_pos_encode(self):        
        device =  'cuda' if torch.cuda.is_available() else 'cpu'
        pe = torch.zeros(self.time_steps, self.embd_dim).to(device)
        for pos in range(self.time_steps):
            for i in range(0, self.embd_dim, 2):    # tow steps loop , for each dim in embddim
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embd_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embd_dim)))
        pe = pe.unsqueeze(0) #to make shape of (batch size , time steps ,embding_dim)
        return pe

    def forward(self, x):
        #x here is embded data must be shape of (batch , time_steps , embding_dim)
        x = x * math.sqrt(self.embd_dim)
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
    weights='none',
    embedding_layer='default',
    classifier='default',
    number_of_neurons=1024,
    classification_dropout_rate=0.4,
    number_of_output_classes=1,
    embedding_dimension=512,
    encoder_dropout_rate=0.1,
    number_of_frames=40,
    encoder_layers=4,
    encoder_heads=8
):
    """
    Args:
        weights: the path for pre-trained DeVTr model
        embedding_layer: CNN network to be used as input embedding layer.
        classifier: any nn.sequential network that receives the output from the transformer encoder
        number_of_neurons: number of neurons of the first layer that goes after the output of the encoder
        classification_dropout_rate: dropout rate of the classification network
        number_of_output_classes: number of output classes (Violence or Non Violence)
        embedding_dimension: number of output dimensions of the CNN network
        encoder_dropout_rate: dropout rate of the transformer encoder
        number_of_frames: number of frames of the input video
        encoder_layers: number of transformer encoder layers
        encoder_heads: number of transformer encoder heads per layer
    """
    # VGG-19 pre-trained network with batch normalization will be used as embedding layer
    if embedding_layer == 'default':
        # If the weights of the pre-trained DeVTr model are passed, default values will be used
        if weights != 'none':
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
            nn.Linear(number_of_frames * embedding_dimension, 1024),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(1024, number_of_output_classes),
        )

        if weights != 'none':
            if torch.cuda.is_available():
                final_model.load_state_dict(torch.load(weights))
            else:
                final_model.load_state_dict(torch.load(weights, map_location ='cpu'))
    # Another network will be used as embedding layer
    else:
        # Combines the network passed in the 'embedding_layer' parameter with a non-linear activation layer
        embedding_network = nn.Sequential(embedding_layer, nn.ReLU())

        # If a classification network is passed through the 'classifier' parameter
        if classifier != 'default':
            final_model = nn.Sequential(
                TimeWarp(embedding_network, method='loop', flatn=False),
                PositionalEncoder(embedding_dimension, dropout=encoder_dropout_rate, time_steps=number_of_frames),
                memoTransormer(embedding_dimension, heads=encoder_heads, layers=encoder_layers, actv='gelu'),
                nn.Flatten(),
                #20480 is frame numbers * dim
                nn.Linear(number_of_frames * embedding_dimension, number_of_neurons),
                nn.Dropout(classification_dropout_rate),
                nn.ReLU(),
                classifier,
            )
        # The default classification network will be used
        else:
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

    return final_model
