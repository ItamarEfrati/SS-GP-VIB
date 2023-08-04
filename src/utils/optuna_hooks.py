import math

from omegaconf import DictConfig
from optuna import Trial


# region Private Methods


def _suggest_gp_parameters(z_dim, trial: Trial):
    trial.suggest_float('model.sigma', low=1, high=10, step=0.1)
    kernel_scales = trial.suggest_categorical(f"model.kernel_scales", [1, 2, 4, 8])
    trial.suggest_float(f"model.length_scale", 0.05, 10, step=0.05)


def _suggest_hidden_layers(n_layers, part, trial: Trial):
    choices = [2 ** i for i in range(3, 10)]
    for i in range(n_layers):
        trial.suggest_categorical(f"model.{part}.hidden_size_{i + 1}", choices=choices)


def _suggest_time_series_hidden_layers(n_layers, part, trial: Trial):
    choices = [2 ** i for i in range(3, 10)]
    for i in range(n_layers):
        trial.suggest_categorical(f"model.{part}.time_series_hidden_{i + 1}", choices=choices)


def _suggest_cnn_paddings(n_cnn_layers, part, trial: Trial):
    paddings = []
    for i in range(n_cnn_layers):
        paddings.append(trial.suggest_categorical(f"model.{part}.padding_{i + 1}", [0, 1, 2, 3, 'same']))
    return paddings


def _suggest_cnn_kernels(n_cnn_layers, part, trial):
    kernels = []
    for i in range(n_cnn_layers):
        kernels.append(trial.suggest_int(f"model.{part}.kernel_size_{i + 1}", 3, 7, step=2))
    return kernels


def _suggest_cnn_dropouts(n_cnn_layers, part, trial):
    dropouts = []
    for i in range(n_cnn_layers):
        dropouts.append(trial.suggest_int(f"model.{part}.dropout_{i + 1}", low=0, high=40, step=10))
    return dropouts


def _suggest_cnn_out_channels(n_cnn_layers, part, trial):
    choices = [2 ** i for i in [3, 4, 5, 6]]
    for i in range(n_cnn_layers):
        trial.suggest_categorical(f"model.{part}.out_channels_{i + 1}", choices)


def _suggest_cnn1d_parameters(n_cnn_layers, part, trial, is_padding_same=False):
    _suggest_cnn_out_channels(n_cnn_layers, part, trial)
    for i in range(n_cnn_layers):
        trial.suggest_int(f"model.{part}.kernel_size_{i + 1}", 3, 11, step=2)
        paddings = ['same'] if is_padding_same else [0, 1, 2, 'same']
        trial.suggest_categorical(f"model.{part}.padding_{i + 1}", paddings)
    _suggest_cnn_dropouts(n_cnn_layers, part=part, trial=trial)


def _suggest_cnn2d_parameters(n_cnn_layers, part, trial):
    _suggest_cnn_out_channels(n_cnn_layers, part, trial)
    for i in range(n_cnn_layers):
        trial.suggest_int(f"model.{part}.h_kernel_size_{i + 1}", 3, 9, step=2)
        if part != 'decoder':
            trial.suggest_int(f"model.{part}.w_kernel_size_{i + 1}", 3, 9, step=2)
        if part == 'timeseries_encoder' and i >= 2:
            continue
        trial.suggest_categorical(f"model.{part}.h_padding_{i + 1}", [0, 1, 2, 'same'])
    _suggest_cnn_dropouts(n_cnn_layers, part=part, trial=trial)


def _suggest_2d_encoder(trial):
    ts_embedding_size = trial.params["model.timeseries_encoder.ts_embedding_size"]
    high = min(9, int(2 ** math.log2(ts_embedding_size)))
    step = 1 if high == 4 else 2
    trial.suggest_int("model.timeseries_encoder.w_kernel_size_1", low=3, high=high, step=step)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)


def _suggest_self_attention(hidden_size, part, trial: Trial):
    max_nhead = min(hidden_size, 4)
    trial.suggest_int(f"model.{part}.nhead", 0, max_nhead, step=2)


# endregion

# region Physionet 2012

# region Old

def physionet_2012_gpvib_e_cnn1d_d_flatten(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    _suggest_cnn1d_parameters(n_cnn_layers, part='timeseries_encoder', trial=trial)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)


def physionet_2012_gpvib_e_cnn2d_d_attention(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    _suggest_2d_encoder(trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_self_attention(hidden_size=z_dim, part='decoder', trial=trial)
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)


# endregion


def physionet_2012_gpvib_e_multi_cnn1d_d_flatten(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    _suggest_cnn1d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial)

    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)


def physionet_2012_gpvib_e_multi_cnn2d_d_flatten(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    _suggest_cnn2d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)

    # model
    if not trial.params["datamodule.use_sample_weight"]:
        trial.suggest_float('model.alpha', low=0.5, high=1.5, step=0.01)


def physionet_2012_gpvib_e_multi_cnn2d_d_cnn1d(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    try:
        n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    except:
        n_cnn_layers = cfg["model"]["timeseries_encoder"]["n_cnn_layers"]
    _suggest_cnn2d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_cnn_layers = trial.params["model.decoder.n_cnn_layers"]
    _suggest_cnn1d_parameters(n_cnn_layers, part='decoder', trial=trial)
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)


def physionet_2012_gpvib_e_multi_cnn2d_d_cnn2d(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    try:
        n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    except:
        n_cnn_layers = cfg["model"]["timeseries_encoder"]["n_cnn_layers"]
    _suggest_cnn2d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_cnn_layers = trial.params["model.decoder.n_cnn_layers"]
    _suggest_cnn2d_parameters(n_cnn_layers, part='decoder', trial=trial)
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)


def physionet_2012_gpvib_e_attention_d_flatten(cfg: DictConfig, trial: Trial) -> None:
    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)


# endregion

# region Mimic3

def mimic3_gp_vib(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    _suggest_2d_encoder(trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)


def mimic3_gpvib_e_multi_cnn2d_d_flatten(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    _suggest_cnn2d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)

    # model
    if not trial.params["datamodule.use_sample_weight"]:
        trial.suggest_float('model.alpha', low=0.5, high=1.5, step=0.01)


# endregion

# region HMnist

def ss_hmnist_gp_vib_e_multi_cnn1d_d_flatten(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    _suggest_cnn1d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial, is_padding_same=True)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # label decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)

    # data decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='data_decoder', trial=trial)


def hmnist_layers(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_layers = trial.params["n_encoder_layers"]
    low, high, step = 512, 1024, 128
    for i in range(3):
        if i < n_layers:
            trial.suggest_int(f"model.encoder.hidden_size_{i + 1}", low, high, step=step)
            high = low
            low /= 2
            step /= 2
        else:
            trial.suggest_int(f"model.encoder.hidden_size_{i + 1}", -1, -1)

    # decoder
    low, high, step = 512, 1024, 128
    n_layers = trial.params["n_decoder_layers"]
    for i in range(n_layers):
        if i < n_layers:
            trial.suggest_int(f"model.decoder.hidden_size_{i + 1}", low, high, step=step)
            high = low
            low /= 2
            step /= 2
        else:
            trial.suggest_int(f"model.decoder.hidden_size_{i + 1}", -1, -1)


# endregion

def ss_hmnist_gp_vib_e_multi_cnn1d_d_flatten(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    _suggest_cnn1d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial, is_padding_same=True)
    n_layers = trial.params["+n_encoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # gp_prior
    z_dim = trial.params["model.timeseries_encoder.encoding_size"]
    _suggest_gp_parameters(z_dim, trial)

    # label decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)

    # data decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='data_decoder', trial=trial)


# region UCR
def ucr(cfg: DictConfig, trial: Trial) -> None:
    # encoder
    n_cnn_layers = trial.params["model.timeseries_encoder.n_cnn_layers"]
    _suggest_cnn1d_parameters(n_cnn_layers=n_cnn_layers, part='timeseries_encoder', trial=trial)

    n_layers = trial.params["+n_encoder_layers"]
    _suggest_time_series_hidden_layers(n_layers=n_layers, part='timeseries_encoder', trial=trial)

    # decoder
    n_layers = trial.params["+n_decoder_layers"]
    _suggest_hidden_layers(n_layers=n_layers, part='decoder', trial=trial)

# endregion
