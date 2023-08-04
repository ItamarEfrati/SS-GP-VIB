import torch

from models.basics.gpvib import GPVIB
from models.encoders import Encoder


class MedicalGPVIB(GPVIB):

    def __init__(self,
                 demographic_encoder: Encoder,
                 is_demographics=False,
                 **kwargs):
        kwargs['ignore'] = ['demographic_encoder', 'positional_encoder']
        kwargs['timeseries_addition'] = int(is_demographics)
        super().__init__(**kwargs)
        self.demographic_encoder = demographic_encoder

    def encode(self, x):
        ts_data, demo_data, indications = x
        ts_data = torch.concat([ts_data, indications], dim=-1)
        ts_encoding = self.timeseries_encoder(ts_data)
        if self.hparams.is_demographics:
            demographic_encoding = self.demographic_encoder(demo_data)
            ts_encoding = torch.hstack([demographic_encoding.unsqueeze(1), ts_encoding])
        pz_x = self.encoder(ts_encoding)
        return pz_x

    def decode(self, z, is_ensemble=False):
        return self.label_decoder(z, is_ensemble)

    def get_x_y(self, batch):
        ts_data, demo_data, indications, y = batch
        x = [ts_data, demo_data, indications]
        return x, y

