import os

import torch
import torchmetrics
import pytorch_lightning as pl

from torch.distributions import MultivariateNormal

from src.models.abstracts.abstract_vib import AbstractVIB


class VIB(AbstractVIB, pl.LightningModule):
    """
    VIB working with images of shape (Batch size, Sequence size, Height, Width)
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 monitor_metric,
                 num_classes,
                 alpha,
                 is_ensemble,
                 class_weight_file,
                 use_class_weight,
                 sample_during_evaluation,
                 ignore=None,
                 **kwargs):
        super().__init__(**kwargs)
        ignore = ['encoder', 'decoder'] if ignore is None else ignore + ['encoder', 'decoder']
        self.save_hyperparameters(ignore=ignore)
        self.train_outputs = []

        # metrics
        task = "binary" if num_classes == 2 else "multiclass"

        metrics = torchmetrics.MetricCollection({
            'ACC': torchmetrics.Accuracy(num_classes=num_classes, task=task),
            'BALACC': torchmetrics.Accuracy(num_classes=num_classes, task=task, average='macro'),
            'RECALL': torchmetrics.Recall(num_classes=num_classes, task=task, average='macro'),
            'PRECISION': torchmetrics.Precision(num_classes=num_classes, task=task, average='macro'),
            'AUROC': torchmetrics.AUROC(num_classes=num_classes, task=task, average='macro'),
            'AUPRC': torchmetrics.AveragePrecision(num_classes=num_classes, task=task, average='macro')
        })

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def r_z(self):
        if self.prior is None:
            self.prior = MultivariateNormal(loc=torch.zeros(self.encoder.encoding_size, device=self.device),
                                            covariance_matrix=torch.eye(self.encoder.encoding_size, device=self.device))
        return self.prior

    def forward(self, x, is_sample, is_ensemble):
        pz_x = self.encode(x)
        if self.hparams.sample_during_evaluation:
            z = pz_x.rsample((self.num_samples,))  # (num_samples, B, z_dim)
        else:
            z = pz_x.loc.unsqueeze(0)
        # transpose batch shape with num samples shape (B, num_samples, z_dim)
        z = z.transpose(0, 1)
        qy_z = self.decode(z, is_ensemble)
        return pz_x, qy_z

    def run_forward_step(self, batch, is_sample, stage):
        x, y = self.get_x_y(batch)

        pz_x, qy_z = self.forward(x, is_ensemble=self.hparams.is_ensemble, is_sample=is_sample)

        log_likelihood = self.compute_log_likelihood(qy_z, y)
        kl = self.compute_kl_divergence(pz_x)

        elbo = log_likelihood - self.hparams.beta * kl
        elbo = elbo.mean()
        loss = -elbo
        probabilities, y_pred = self.get_y_pred(qy_z)
        return {'log': {'loss': loss,
                        'kl_mean': kl.mean(),
                        'mean_negative_log_likelihood': (-log_likelihood).mean()},
                'preds': y_pred,
                'probs': probabilities,
                'target': y,
                'latent': pz_x.mean}

    def step(self, batch, stage):
        is_sample = any([stage is 'train',
                         all([stage is not 'train', self.hparams.sample_during_evaluation])])
        forward_outputs = self.run_forward_step(batch, is_sample, stage)
        log_dict = forward_outputs.pop('log')
        forward_outputs['loss'] = log_dict['loss']
        log_dict = {f'{stage}_{k}': v for k, v in log_dict.items()}
        self.log_dict(log_dict, prog_bar=False)
        return forward_outputs

    def get_x_y(self, batch):
        x, y = batch[0], batch[1]
        if not self.hparams.is_ensemble:
            # if not ensemble that the model is expecting (B, input_dim)
            x = torch.flatten(x, 1)
        else:
            y = torch.tile(y.reshape(-1, 1), (1, x.shape[1]))
        return x, y

    def get_y_pred(self, qy_z):
        probabilities = qy_z.mean if not self.hparams.is_ensemble else qy_z.mean.mean(1)
        y_pred = torch.argmax(probabilities, dim=1)
        probabilities = probabilities[:, 1] if self.hparams.num_classes == 2 else probabilities
        return probabilities, y_pred

    # region Loss computations

    def compute_log_likelihood(self, qy_z, y):
        nll = torch.nn.NLLLoss(reduction='none', weight=self.class_weight)
        log_likelihood = -nll(qy_z.logits, y.long())
        return log_likelihood

    # endregion

    # region Pytorch lightning overwrites

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler:
            lr_scheduler = {
                "scheduler": self.hparams.scheduler(optimizer=optimizer),
                "monitor": self.hparams.monitor_metric,
                "interval": 'epoch',
                "frequency": 1
            }
            return [optimizer], [lr_scheduler]
        return optimizer

    def setup(self, stage=None):
        if self.hparams.use_class_weight:
            class_weight = torch.load(self.hparams.class_weight_file, map_location=self.device).float()
            class_weight[1] *= self.hparams.alpha
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def training_step(self, batch, batch_idx):
        step_output = self.step(batch, stage='train')
        self.train_metrics.update(step_output['probs'], step_output['target'])
        self.log_dict(self.train_metrics, on_epoch=True, prog_bar=False)
        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'], on_step=True, prog_bar=True)
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = self.step(batch, stage='val')
        self.val_metrics.update(step_output['probs'], step_output['target'])
        self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        return step_output

    def test_step(self, batch, batch_idx):
        step_output = self.step(batch, stage='test')
        self.test_metrics.update(step_output['probs'], step_output['target'])
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True)
        return step_output

    # endregion
