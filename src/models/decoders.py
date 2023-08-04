from abc import ABC, abstractmethod

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from utils.model_utils import get_linear_layers, get_1d_cnn_layers, Permute, get_2d_cnn_layers, Reshape, Unsqueeze, \
    ExtractTensor, get_cnn1d_output_dim, get_cnn2d_output_dim, get_parameters_list, get_parameters_list_tuples, \
    SelfAttention, Sum, Mean, Max

from torch.distributions.multinomial import Multinomial


class Decoder(nn.Module, ABC):

    def __init__(self, num_samples, output_size):
        """ Decoder parent class with no specified output distribution
        """
        super(Decoder, self).__init__()
        self.num_samples = num_samples
        self.output_size = output_size

    def get_probs(self, logits, is_ensemble):
        if is_ensemble:
            predictions_per_sample = torch.argmax(logits, -1)
            pre_instance_occurrences = F.one_hot(predictions_per_sample, self.output_size).sum(1)
            probs = pre_instance_occurrences / self.num_samples
        else:
            logits = logits.mean(1)
            probs = F.softmax(logits, dim=-1)
        return probs

    @abstractmethod
    def __call__(self, x, is_train=False):
        pass


# region Bernoulli
class BernoulliDecoder(Decoder):

    def __init__(self,
                 num_samples,
                 output_size,
                 z_dim,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3
                 ):
        super().__init__(num_samples, output_size)
        input_size = z_dim
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def __call__(self, x, is_train=False):
        logits = self.net(x)
        probs = self.get_probs(logits, is_ensemble=False)
        return Bernoulli(probs=probs)


# endregion


# region Multinomial

class MultinomialDecoder(Decoder):
    """
    Uses MLP layers to calculate the logits of the multinomial distribution of p(y|z).
    z is of shape z_dim X time_length which means that z is a matrix. 
    The MLP runs over the last therefore we flatten the matrix which means that the first MLP size is of 
    z_dim X time_length.  
    """

    def __init__(self,
                 z_dim,
                 num_samples,
                 output_size,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3
                 ):
        super(MultinomialDecoder, self).__init__(num_samples, output_size)
        input_size = z_dim
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def __call__(self, z, is_ensemble=False):
        logits = self.net(z)
        probs = self.get_probs(logits, is_ensemble)
        return Multinomial(probs=probs)


# region TimeSeriesMultinomial

class FlattenMultinomialDecoder(Decoder):
    """
    Uses MLP layers to calculate the logits of the multinomial distribution of p(y|z).
    z is of shape z_dim X time_length which means that z is a matrix.
    The MLP runs over the last therefore we flatten the matrix which means that the first MLP size is of
    z_dim X time_length.
    """

    def __init__(self,
                 z_dim,
                 z_dim_time_length,
                 output_size,
                 num_samples,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 **kwargs):
        super(FlattenMultinomialDecoder, self).__init__(num_samples, output_size)
        input_size = z_dim * z_dim_time_length
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def _get_logits(self, z):
        z = z.flatten(-2)
        logits = self.net(z)
        logits = logits.reshape(-1, z.shape[1], logits.shape[-1])
        return logits

    def __call__(self, z, is_ensemble=False):
        logits = self._get_logits(z)
        probs = self.get_probs(logits, is_ensemble)

        return Multinomial(probs=probs)


class Cnn1DMultinomialDecoder(Decoder):
    def __init__(self,
                 z_dim,
                 z_dim_time_length,
                 n_cnn_layers,
                 out_channels_1,
                 out_channels_2,
                 out_channels_3,
                 kernel_size_1,
                 kernel_size_2,
                 kernel_size_3,
                 padding_1,
                 padding_2,
                 padding_3,
                 dropout_1,
                 dropout_2,
                 dropout_3,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 num_samples,
                 output_size,
                 ):
        """ Decoder parent class with no specified output distribution
            :param hidden_sizes: tuple of hidden layer sizes. The tuple length sets the number of hidden layers.
        """
        super(Cnn1DMultinomialDecoder, self).__init__(num_samples, output_size)

        in_channels = z_dim
        in_features = z_dim_time_length
        kernels_list = get_parameters_list(kernel_size_1, kernel_size_2, kernel_size_3, length=n_cnn_layers)
        padding_list = get_parameters_list(padding_1, padding_2, padding_3, length=n_cnn_layers)
        dropout_list = get_parameters_list(dropout_1, dropout_2, dropout_3, length=n_cnn_layers)
        cnn1d_sizes = get_parameters_list(in_channels, out_channels_1, out_channels_2, out_channels_3,
                                          length=n_cnn_layers + 1)

        assert all([len(kernels_list) == len(cnn1d_sizes) - 1, len(padding_list) == len(cnn1d_sizes) - 1]), \
            "kernels list or padding list are not in the same size of the cnn list"
        output_dim = get_cnn1d_output_dim(in_features, kernels_list, padding_list)

        layers = [Reshape((-1, z_dim_time_length, z_dim)),
                  Permute((0, 2, 1))]
        layers += get_1d_cnn_layers(cnn1d_sizes, kernels_list, padding_list, dropout_list)
        # flatten over the time channel dimension
        layers += [nn.Flatten(-2)]

        input_size = cnn1d_sizes[-1] * output_dim

        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers += get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def _get_logits(self, z):
        logits = self.net(z)
        logits = logits.reshape(-1, z.shape[1], logits.shape[-1])
        return logits

    def __call__(self, z, is_voting=False):
        logits = self._get_logits(z)
        probs = self.get_probs(logits, is_voting)
        return Multinomial(probs=probs)


class Cnn2DMultinomialDecoder(Decoder):
    def __init__(self,
                 z_dim,
                 z_dim_time_length,
                 n_cnn_layers,
                 out_channels_1,
                 out_channels_2,
                 out_channels_3,
                 h_kernel_size_1,
                 h_kernel_size_2,
                 h_kernel_size_3,
                 h_padding_1,
                 h_padding_2,
                 h_padding_3,
                 w_kernel_size_1,
                 w_kernel_size_2,
                 w_kernel_size_3,
                 w_padding_1,
                 w_padding_2,
                 w_padding_3,
                 dropout_1,
                 dropout_2,
                 dropout_3,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 num_samples,
                 output_size
                 ):
        """ Decoder parent class with no specified output distribution
            :param hidden_sizes: tuple of hidden layer sizes. The tuple length sets the number of hidden layers.
        """
        super(Cnn2DMultinomialDecoder, self).__init__(num_samples, output_size)

        in_channels = 1
        out_channels_end = 1
        w_kernel_size_end = h_kernel_size_end = 1
        padding_end = 0

        cnn2d_sizes = get_parameters_list(in_channels, out_channels_1, out_channels_2, out_channels_3,
                                          length=n_cnn_layers + 1)
        cnn2d_sizes += [out_channels_end]

        kernels = get_parameters_list_tuples((h_kernel_size_1, w_kernel_size_1),
                                             (h_kernel_size_2, w_kernel_size_2),
                                             (h_kernel_size_3, w_kernel_size_3),
                                             length=n_cnn_layers)
        kernels += [(h_kernel_size_end, w_kernel_size_end)]

        paddings = get_parameters_list_tuples((h_padding_1, w_padding_1),
                                              (h_padding_2, w_padding_2),
                                              (h_padding_3, w_padding_3),
                                              length=n_cnn_layers)

        paddings += [(padding_end, padding_end)]

        dropouts = get_parameters_list(dropout_1, dropout_2, dropout_3, length=n_cnn_layers)
        dropouts += [0]

        output_dims = get_cnn2d_output_dim(feature_dim=(z_dim_time_length, z_dim), kernels=kernels,
                                           padding_list=paddings)

        layers = [Reshape((-1, in_channels, z_dim_time_length, z_dim))]
        layers += get_2d_cnn_layers(cnn2d_sizes, kernels, paddings, dropouts)
        # flatten over the channel dimension
        layers += [nn.Flatten(-3)]

        input_size = cnn2d_sizes[-1] * output_dims[0] * output_dims[1]

        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)

        hidden_sizes.append(output_size)
        layers += get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def _get_logits(self, z):
        # if self.is_demographics:
        #     # demographic_encoding = z[1]
        #     # demographic_encoding = torch.tile(demographic_encoding.unsqueeze(1), (1, self.num_samples, 1))
        #     # z = z[0].unsqueeze(2)
        #     # z = torch.concat([demographic_encoding, z], -1)
        #     # Todo concat after cnn layers
        #     pass
        # else:
        #     z = z.unsqueeze(2)
        logits = self.net(z)
        logits = logits.reshape(-1, z.shape[1], logits.shape[-1])
        return logits

    def __call__(self, z, is_voting=False):
        logits = self._get_logits(z)
        probs = self.get_probs(logits, is_voting)
        return Multinomial(probs=probs)


class MultiHeadAttentionDecoder(Decoder):
    def __init__(self,
                 z_dim,
                 z_dim_time_length,
                 nhead,
                 dim_feedforward,
                 attention_dropout,
                 aggregation,
                 output_size,
                 num_samples,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 positional_encoder,
                 **kwargs):
        super(MultiHeadAttentionDecoder, self).__init__(num_samples, output_size)
        layers = [Reshape((-1, z_dim_time_length, z_dim))]
        if positional_encoder is not None:
            positional_encoder = positional_encoder(max_len=z_dim_time_length)
            layers += [positional_encoder]
        nhead = 1 if nhead == 0 else nhead
        layers += [nn.TransformerEncoderLayer(d_model=z_dim,
                                              nhead=nhead,
                                              dim_feedforward=dim_feedforward,
                                              dropout=attention_dropout)]
        input_size = z_dim
        if aggregation == 'sum':
            layers += [Sum(dim=-2)]
        if aggregation == 'mean':
            layers += [Mean(dim=-2)]
        if aggregation == 'max':
            layers += [Max(dim=-2)]
        if aggregation == 'flatten':
            input_size = z_dim * z_dim_time_length
            layers += [nn.Flatten(start_dim=-2)]
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers += get_linear_layers(hidden_sizes)
        layers += [Reshape((-1, self.num_samples, hidden_sizes[-1]))]
        self.net = nn.Sequential(*layers)

    def _get_logits(self, z):
        logits = self.net(z)
        return logits

    def __call__(self, z, is_voting=False):
        logits = self._get_logits(z)
        probs = self.get_probs(logits, is_voting)
        try:
            return Multinomial(probs=probs)
        except:
            logits = self._get_logits(z)
            probs = self.get_probs(logits, is_voting)
            return Multinomial(probs=probs)


class AdditiveAttentionDecoder(Decoder):
    def __init__(self,
                 z_dim,
                 attention_hidden_size,
                 z_dim_time_length,
                 output_size,
                 num_samples,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 positional_encoder,
                 **kwargs):
        super(AdditiveAttentionDecoder, self).__init__(num_samples, output_size)
        layers = []
        self.query_proj = nn.Linear(z_dim, attention_hidden_size, bias=True)
        self.key_proj = nn.Linear(z_dim, attention_hidden_size, bias=True)
        self.bias = nn.Parameter(torch.rand(attention_hidden_size).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(attention_hidden_size, 1)

        hidden_sizes = [z_dim]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers += get_linear_layers(hidden_sizes)
        layers += [Reshape((-1, self.num_samples, hidden_sizes[-1]))]
        self.net = nn.Sequential(*layers)

    def _get_logits(self, z):
        z = z.reshape(-1, z.shape[-2], z.shape[-1])
        key = query = value = z
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias))
        attn = F.softmax(score, dim=1)
        # attn = score
        context = torch.bmm(value.permute(0, 2, 1), attn)
        # context = (value.permute(0, 2, 1) @ attn)
        out = self.net(context.squeeze())
        return out

    def __call__(self, z, is_voting=False):
        logits = self._get_logits(z)
        probs = self.get_probs(logits, is_voting)
        try:
            return Multinomial(probs=probs)
        except:
            logits = self._get_logits(z)
            probs = self.get_probs(logits, is_voting)
            return Multinomial(probs=probs)


class RNNMultinomialDecoder(Decoder):
    """
    Uses MLP layers to calculate the logits of the multinomial distribution of p(y|z).
    z is of shape z_dim X time_length which means that z is a matrix.
    The MLP runs over the last therefore we flatten the matrix which means that the first MLP size is of
    z_dim X time_length.
    """

    def __init__(self,
                 z_dim,
                 num_samples,
                 output_size,
                 time_length,
                 rnn_n_layers,
                 lstm_hidden_size,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3
                 ):
        super(RNNMultinomialDecoder, self).__init__(num_samples, output_size)
        input_size = z_dim
        lstm_layers = [
            Reshape((-1, time_length, input_size)),
            nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=rnn_n_layers,
                    batch_first=True),
            ExtractTensor()
        ]
        hidden_sizes = [lstm_hidden_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        layers = lstm_layers + layers + [Reshape((-1, self.num_samples, output_size))]
        self.net = nn.Sequential(*layers)

    def __call__(self, z, is_voting=False):
        probs = self.get_probs(z, is_voting)
        return Multinomial(probs=probs)

# endregion

# endregion
