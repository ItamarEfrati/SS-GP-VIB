import io
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

from utils.gp_kernels import rbf_kernel, diffusion_kernel, matern_kernel, cauchy_kernel


# region Custom Layers
class Permute(nn.Module):
    def __init__(self, perm):
        super(Permute, self).__init__()
        self.perm = perm

    def forward(self, x):
        return x.permute(self.perm)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class Sum(nn.Module):
    def __init__(self, dim):
        super(Sum, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(self.dim)


class Mean(nn.Module):
    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(self.dim)


class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.sum(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2, need_weights=False):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                               nhead=num_heads,
                                                               dim_feedforward=32,
                                                               dropout=0.4)
        self.need_weights = need_weights

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-2], x_shape[-1])
        out = self.multi_head_attention(x)
        if not self.need_weights:
            out = out.reshape(x_shape)
        return out


# LSTM() returns tuple of (tensor, (recurrent state))
class ExtractTensor(nn.Module):
    @staticmethod
    def forward(x):
        # Output shape (batch, features, hidden)
        output, _ = x
        # Reshape shape (batch, hidden)
        return output[:, -1, :]


# endregion

# region Layers utils

def get_linear_layers(hidden_sizes):
    layers = []
    for i in range(len(hidden_sizes) - 2):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], dtype=torch.float32))
        layers.append(nn.ReLU(), )
    layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1], dtype=torch.float32))
    return layers


def get_1d_cnn_layers(cnn_channels, kernel_size, padding, dropout):
    layers = []
    for i in range(len(cnn_channels) - 1):
        layers.append(nn.Conv1d(in_channels=cnn_channels[i],
                                out_channels=cnn_channels[i + 1],
                                kernel_size=kernel_size[i],
                                padding=padding[i],
                                dtype=torch.float32))
        # layers.append(nn.BatchNorm1d(cnn_channels[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout[i] / 100))
    return layers


def get_2d_cnn_layers(cnn_channels, kernel_size, padding_list, dropout):
    layers = []
    for i in range(len(cnn_channels) - 1):
        padding = 'same' if 'same' in padding_list[i] else padding_list[i]
        layers.append(nn.Conv2d(in_channels=cnn_channels[i],
                                out_channels=cnn_channels[i + 1],
                                kernel_size=kernel_size[i],
                                padding=padding,
                                dtype=torch.float32,
                                ))
        # layers.append(nn.BatchNorm2d(cnn_channels[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout[i] / 100))
    return layers


def get_cnn1d_output_dim(feature_dim, kernels, padding_list):
    output_size = feature_dim
    for i in range(len(kernels)):
        if padding_list[i] == 'same':
            output_size = feature_dim
        else:
            output_size = feature_dim + 2 * padding_list[i] - kernels[i] + 1
        feature_dim = output_size
    return output_size


def get_cnn2d_output_dim(feature_dim, kernels, padding_list):
    h_out = get_cnn1d_output_dim(feature_dim[0], list(map(lambda x: x[0], kernels)),
                                 list(map(lambda x: x[0], padding_list)))
    w_out = get_cnn1d_output_dim(feature_dim[1], list(map(lambda x: x[1], kernels)),
                                 list(map(lambda x: x[1], padding_list)))
    output_size = (h_out, w_out)
    return output_size


def get_parameters_list(*args, length=-1):
    parameters = []
    for i in range(len(args)):
        if any([args[i] == -1, i == length]):
            break
        parameters.append(args[i])
    return parameters


def get_parameters_list_tuples(*args, length=-1):
    parameters = []
    for i in range(len(args)):
        if any([args[i] == -1, i == length]):
            break
        if 'same' in args[i]:
            # the sign for padding same
            parameters.append(('same', 'same'))
        else:
            parameters.append(args[i])
    return parameters


# endregion

# region GP

def get_gp_prior(kernel, kernel_scales, time_length, sigma, length_scale, z_dim, device):
    """

    :param kernel: the type of the kernel to use
    :param kernel_scales: when constructing the prior of z each latent dimension of z ha its own prior. When
                          the kernel scale is bigger than 1 than we will use more than 1 prior.
    :param time_length:
    :param sigma:
    :param length_scale: scaling factor when calculating the kernel. The smaller the more weight closer items have
    :param z_dim:
    :param device:
    :return:
    """
    # Compute kernel matrices for each latent dimension
    kernel_matrices = []
    scaling = 1
    for i in range(kernel_scales):
        if kernel == "rbf":
            kernel_matrices.append(rbf_kernel(time_length, length_scale / scaling))
        elif kernel == "diffusion":
            kernel_matrices.append(diffusion_kernel(time_length, length_scale / scaling))
        elif kernel == "matern":
            kernel_matrices.append(matern_kernel(time_length, length_scale / scaling))
        elif kernel == "cauchy":
            kernel_matrices.append(cauchy_kernel(time_length, sigma, length_scale / scaling))
        scaling += 2 if scaling > 1 else 1

    # Combine kernel matrices for each latent dimension
    tiled_matrices = []
    total = 0
    for i in range(kernel_scales):
        if i == kernel_scales - 1:
            multiplier = z_dim - total
        else:
            multiplier = int(z_dim / kernel_scales)
            total += multiplier
        tiled_matrices.append(torch.tile(torch.unsqueeze(kernel_matrices[i], 0), [multiplier, 1, 1]))
    kernel_matrix_tiled = np.concatenate(tiled_matrices)
    assert len(kernel_matrix_tiled) == z_dim
    kernel_matrix_tiled = torch.tensor(kernel_matrix_tiled, device=device)

    return torch.distributions.MultivariateNormal(
        loc=torch.zeros([z_dim, time_length], dtype=torch.float32, device=device),
        covariance_matrix=kernel_matrix_tiled)


# endregion

# region Evaluation

def get_confusion_matrix_image(confusion_matrix, num_classes):
    df_cm = pd.DataFrame(confusion_matrix.detach().cpu().numpy().astype(int),
                         index=range(num_classes),
                         columns=range(num_classes))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)

    ax.legend(
        df_cm.index,
        df_cm.columns,
        handler_map={int: IntHandler()},
        loc='upper left',
        bbox_to_anchor=(1.2, 1)
    )
    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im


def get_precision_recall_curve_image(target, probabilities):
    positive_class_probabilities = probabilities.detach().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(y_true=target.detach().cpu().numpy(),
                                                           probas_pred=positive_class_probabilities)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    ax.plot(recall, precision, color='purple')

    # add axis labels to plot
    ax.axis('equal')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im


def get_roc_curve_image(target, probabilities):
    positive_class_probabilities = probabilities.detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true=target.detach().cpu().numpy(),
                                     y_score=positive_class_probabilities)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    ax.plot(fpr, tpr, color='purple')

    # add axis labels to plot
    ax.axis('equal')
    ax.set_title('ROC Curve')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    buf = io.BytesIO()

    plt.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    return im


class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


# endregion


if __name__ == '__main__':
    get_cnn2d_output_dim(feature_dim=(3, 49), kernels=[(5, 7), (3, 3), (3, 3)], padding_list=[(-2, -2), (1, 1), (1, 1)])
