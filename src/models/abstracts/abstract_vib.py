import torch

from abc import ABC, abstractmethod

from src.models.decoders import Decoder
from src.models.encoders import Encoder


class AbstractVIB(ABC):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 beta: float,
                 num_samples: int
                 ):
        super(AbstractVIB, self).__init__()
        self.encoder = encoder
        self.label_decoder = decoder
        self.beta = beta
        self.num_samples = num_samples

        self.prior = None

    @abstractmethod
    def r_z(self):
        pass

    @abstractmethod
    def step(self, batch, stage):
        pass

    @abstractmethod
    def compute_log_likelihood(self, px_z, x):
        pass

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, is_ensemble=False):
        return self.label_decoder(z, is_ensemble)

    def compute_kl_divergence(self, q):
        kl = torch.distributions.kl.kl_divergence(q, self.r_z())
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        return kl

    def forward(self, x, is_sample, is_ensemble):
        pz_x = self.encode(x)
        if is_sample:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        z = z.transpose(0, 1)
        qy_z = self.decode(z, is_ensemble)
        return pz_x, qy_z

    # endregion

    def get_latent_vectors(self, data_loader):
        labels = []
        latent_vectors = []
        for batch in data_loader:
            x, y = batch
            q_z = self.encoder(x)
            z = q_z.sample()
            latent_vectors.append(z)
            labels.append(y)
        return torch.concat(latent_vectors), torch.concat(labels)
