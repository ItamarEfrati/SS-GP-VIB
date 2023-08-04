import torch
import pytorch_lightning as pl

from models.basics.ss_vib import SemiSupervisedVIB
from models.encoders import TimeSeriesDataEncoder
from utils.model_utils import get_gp_prior


class SemiSupervisedGPVIB(SemiSupervisedVIB, pl.LightningModule):

    def __init__(self,
                 kernel,
                 sigma,
                 length_scale,
                 kernel_scales,
                 timeseries_encoder: TimeSeriesDataEncoder,
                 is_ensemble=False,
                 timeseries_addition=0,
                 **kwargs):
        kwargs['is_ensemble'] = is_ensemble
        kwargs['ignore'] = ['timeseries_encoder'] if 'ignore' not in kwargs.keys() else kwargs['ignore'] + [
            'timeseries_encoder']
        super().__init__(**kwargs)
        self.timeseries_encoder = timeseries_encoder
        self.z_dim_time_length = self.timeseries_encoder.encoding_series_size + timeseries_addition
        self.encoder = self.encoder(z_dim_time_length=self.z_dim_time_length)
        self.label_decoder = self.label_decoder(z_dim_time_length=self.z_dim_time_length)

    def r_z(self):
        if self.prior is None:
            self.prior = get_gp_prior(kernel=self.hparams.kernel, kernel_scales=self.hparams.kernel_scales,
                                      time_length=self.z_dim_time_length, sigma=self.hparams.sigma,
                                      length_scale=self.hparams.length_scale, z_dim=self.encoder.encoding_size,
                                      device=self.device)
        return self.prior

    def forward(self, x, is_ensemble=False, is_sample=True, is_train=True):
        pz_x = self.encode(x)
        if is_sample:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        # transpose features and time dimensions to run cnn over time per feature
        # transpose time and z_dim dimensions
        # z is now a latent series of shape (batch, num_samples, time_length, z_dim)
        z = z.permute(1, 0, 3, 2)
        qy_z, px_z = self.decode(z, is_ensemble, is_train)
        return pz_x, qy_z, px_z

    def encode(self, x):
        # transpose features and time dimensions to run cnn over time per feature
        # x = x.transpose(-2, -1)
        x = self.timeseries_encoder(x)
        pz_x = self.encoder(x)
        return pz_x

    def compute_kl_divergence(self, pz_x):
        kl = torch.distributions.kl.kl_divergence(pz_x, self.r_z())
        kl = torch.where(torch.torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl.sum(-1)

    def get_x_y(self, batch, is_train=True):
        if is_train:
            labeled_data, unlabeled_data = batch[0], batch[1]
            x_unlabeled = unlabeled_data[0]
            x, y = labeled_data[0], labeled_data[1]
        else:
            x_unlabeled = None
            x, y = batch[0], batch[1]
        if self.hparams.is_ensemble:
            y = torch.tile(y.reshape(-1, 1), (1, x.shape[1]))
        return x, y, x_unlabeled
