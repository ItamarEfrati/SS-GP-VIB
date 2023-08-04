import os

import torch
import torchmetrics
import pytorch_lightning as pl

from torch.distributions import MultivariateNormal

from models.basics.vib import VIB
from models.decoders import Decoder
from src.models.abstracts.abstract_vib import AbstractVIB


class SemiSupervisedVIB(VIB, pl.LightningModule):
    """
    VIB working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self,
                 data_decoder: Decoder,
                 ignore=None,
                 **kwargs):
        ignore = ['encoder', 'decoder', 'data_decoder'] if ignore is None else ignore + ['encoder',
                                                                                         'decoder',
                                                                                         'data_decoder']
        kwargs['ignore'] = ignore
        super().__init__(**kwargs)

        self.data_decoder = data_decoder

    def decode(self, z, is_ensemble=False, is_train=True):
        if is_train:
            z_labeled, z_unlabeled = z[:z.shape[0] // 2], z[z.shape[0] // 2:]
            px_z = self.data_decoder(z_unlabeled, is_ensemble)
            qy_z = self.label_decoder(z_labeled, is_ensemble)
        else:
            qy_z = self.label_decoder(z, is_ensemble)
            px_z = None

        return qy_z, px_z

    def forward(self, x, is_ensemble=False, is_sample=True, is_train=True):
        pz_x = self.encode(x)
        if is_sample:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        # transpose batch shape with num samples shape (B, num_samples, z_dim)
        z = z.transpose(0, 1)
        qy_z, px_z = self.decode(z, is_ensemble, is_train)
        return pz_x, qy_z, px_z

    def run_forward_step(self, batch, is_sample, stage):
        is_train = stage is 'train'
        x, y, x_unlabeled = self.get_x_y(batch, is_train)
        if is_train:
            x_temp = torch.concat([x, x_unlabeled])
        else:
            x_temp = x
        pz_x, qy_z, px_z = self.forward(x_temp, self.hparams.is_ensemble, is_sample, is_train)

        log_likelihood = self.compute_log_likelihood(qy_z, y)
        if is_train:
            data_log_likelihood = self.compute_log_likelihood(px_z, x_unlabeled, is_multinomial=False)
            data_log_likelihood = data_log_likelihood.sum(dim=[1, 2])
            reconstruction_error = torch.concat([log_likelihood, data_log_likelihood])
            data_log_likelihood = data_log_likelihood.mean()
            log_values = {'mean_label_negative_log_likelihood': (-log_likelihood).mean(),
                          'mean_data_negative_log_likelihood': data_log_likelihood}
        else:
            reconstruction_error = log_likelihood
            log_values = {'mean_label_negative_log_likelihood': (-log_likelihood).mean()}
        kl = self.compute_kl_divergence(pz_x)

        elbo = reconstruction_error - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo
        probabilities, y_pred = self.get_y_pred(qy_z)
        log_values['loss'] = loss
        log_values['kl_mean'] = kl.mean()
        return {'log': log_values,
                'preds': y_pred,
                'probs': probabilities,
                'target': y}

    def get_x_y(self, batch, is_train=True):
        if is_train:
            labeled_data, unlabeled_data = batch[0], batch[1]
            x_unlabeled = unlabeled_data[0]
            x, y = labeled_data[0], labeled_data[1]
        else:
            x_unlabeled = None
            x, y = batch[0], batch[1]
        if not self.hparams.is_ensemble:
            # if not ensemble that the model is expecting (B, input_dim)
            x = torch.flatten(x, 1)
        else:
            y = torch.tile(y.reshape(-1, 1), (1, x.shape[1]))
        return x, y, x_unlabeled

    def get_y_pred(self, qy_z):
        probabilities = qy_z.mean if not self.hparams.is_ensemble else qy_z.mean.mean(1)
        y_pred = torch.argmax(probabilities, dim=1)
        probabilities = probabilities[:, 1] if self.hparams.num_classes == 2 else probabilities
        return probabilities, y_pred

    # region Loss computations

    def compute_log_likelihood(self, prob, target, class_weight=None, is_multinomial=True):
        if class_weight is None:
            if is_multinomial:
                target = torch.nn.functional.one_hot(target.long(), num_classes=prob.event_shape[-1])
            log_likelihood = prob.log_prob(target)
        else:
            nll = torch.nn.NLLLoss(reduction='none', weight=class_weight)
            log_likelihood = -nll(prob.logits, target.long())
        return log_likelihood

    # endregion
