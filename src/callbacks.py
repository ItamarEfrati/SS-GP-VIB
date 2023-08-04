import pytorch_lightning as pl
import torch
import torchmetrics

from torch import nn
from typing import Optional
from torch.optim import Optimizer
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping

from utils.model_utils import get_confusion_matrix_image, get_precision_recall_curve_image, get_roc_curve_image


class MyEarlyStopping(Callback):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for m, v in trainer.lightning_module.val_metrics.items():
            if 'Accuracy' in m:
                d = v

        if d.compute().item() == 1:
            trainer.should_stop = True


class LogEvaluationMetrics(Callback):
    def __init__(self, num_classes):
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        self.num_classes = num_classes

    def log_metrics_images(self, outputs, trainer, pl_module, split):
        predictions = torch.cat([tmp['predictions'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        probabilities = torch.cat([tmp['probabilities'] for tmp in outputs])

        confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.num_classes).to(pl_module.device)
        confusion_matrix = confusion_matrix(predictions, targets)

        cm_im = get_confusion_matrix_image(confusion_matrix, self.num_classes)
        prc_im = get_precision_recall_curve_image(targets, probabilities)
        rocc_im = get_roc_curve_image(targets, probabilities)

        trainer.logger.experiment.add_image(f"{split} Confusion Matrix", cm_im, trainer.current_epoch)
        trainer.logger.experiment.add_image(f"{split} Precision Recall Curve", prc_im, trainer.current_epoch)
        trainer.logger.experiment.add_image(f"{split} ROC Curve", rocc_im, trainer.current_epoch)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if len(self.train_outputs) > 0:
            self.log_metrics_images(self.train_outputs, trainer, pl_module, 'Train')
            self.train_outputs = []

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_metrics_images(self.val_outputs, trainer, pl_module, 'Validation')
        self.val_outputs = []

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_metrics_images(self.test_outputs, trainer, pl_module, 'Test')
        self.test_outputs = []

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.train_outputs.append(outputs)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.val_outputs.append(outputs)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.test_outputs.append(outputs)


class WarmupStart(Callback):

    def __init__(self, warmup_steps, lr):
        self.warmup_steps = warmup_steps
        self.lr = lr

    def on_before_optimizer_step(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer, opt_idx: int
    ) -> None:
        if self.warmup_steps == 0:
            return
        if trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_step == self.warmup_steps:
            trainer.progress_bar_callback.print("Finish warmup")


class XaviarInit(Callback):

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        for module in pl_module._modules.values():
            if 'net' in dir(module):
                for layer in list(module.net):
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
                        continue
                    if 'weight' in dir(layer):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        layer.bias.data.zero_()


class ParameterInitialization(Callback):

    def __init__(self, init_method):
        self.init_method = init_method

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if self.init_method is None:
            return
        for module in pl_module._modules.values():
            if 'net' in dir(module):
                for layer in list(module.net):
                    if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                        nn.init.constant_(layer.weight, 1)
                        nn.init.constant_(layer.bias, 0)
                        continue
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                        if self.init_method == 'kaiming':
                            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                        elif self.init_method == 'xavier':
                            nn.init.xavier_uniform_(layer.weight)
                        else:
                            raise Exception(f"Wrong init name for {self.init_method}")
                        layer.bias.data.zero_()


class LatentSpaceSaver(Callback):
    def __init__(self):
        self.train_outputs = []
        self.train_predictions = []
        self.train_target = []
        self.test_outputs = []
        self.test_predictions = []
        self.test_target = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        train_outputs = torch.cat(self.train_outputs)
        train_predictions = torch.cat(self.train_predictions)
        train_target = torch.cat(self.train_target)
        torch.save(train_outputs, 'train_z.pt')
        torch.save(train_predictions, 'train_pred.pt')
        torch.save(train_target, 'train_target.pt')
        self.train_outputs = []
        self.train_predictions = []
        self.train_target = []

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.test_outputs = torch.cat(self.test_outputs)
        self.test_predictions = torch.cat(self.test_predictions)
        self.test_target = torch.cat(self.test_target)
        torch.save(self.test_outputs, 'test_z.pt')
        torch.save(self.test_predictions, 'test_pred.pt')
        torch.save(self.test_target, 'test_target.pt')

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.train_outputs.append(outputs['latent'])
        self.train_predictions.append(outputs['preds'])
        self.train_target.append(outputs['target'])

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        self.test_outputs.append(outputs['latent'])
        self.test_predictions.append(outputs['preds'])
        self.test_target.append(outputs['target'])
