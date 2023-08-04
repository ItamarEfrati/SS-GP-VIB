import torch
import pytorch_lightning as pl

from models.basics.vib import VIB
from models.encoders import TimeSeriesDataEncoder
from utils.model_utils import get_gp_prior


class GPVIB(VIB, pl.LightningModule):

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

    def forward(self, x, is_ensemble=False, is_sample=True):
        pz_x = self.encode(x)
        if is_sample:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        # transpose features and time dimensions to run cnn over time per feature
        # transpose time and z_dim dimensions
        # z is now a latent series of shape (batch, num_samples, time_length, z_dim)
        z = z.permute(1, 0, 3, 2)
        qy_z = self.decode(z, is_ensemble)
        return pz_x, qy_z

    def encode(self, x):
        x = self.timeseries_encoder(x)
        pz_x = self.encoder(x)
        return pz_x

    def decode(self, z, is_ensemble=False):
        return self.label_decoder(z, is_ensemble)

    def compute_kl_divergence(self, pz_x):
        kl = torch.distributions.kl.kl_divergence(pz_x, self.r_z())
        kl = torch.where(torch.torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl.sum(-1)

    def compute_log_likelihood(self, qy_z, y):
        nll = torch.nn.NLLLoss(reduction='none', weight=self.class_weight)
        log_likelihood = -nll(qy_z.logits, y.long())
        return log_likelihood.sum(-1) if self.hparams.is_ensemble else log_likelihood

    def get_x_y(self, batch):
        x = batch[0]
        y = batch[-1]
        if self.hparams.is_ensemble:
            y = torch.tile(y.reshape(-1, 1), (1, x.shape[1]))
        return x, y
