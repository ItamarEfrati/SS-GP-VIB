import os
from typing import List, Tuple

import hydra
import lightning.pytorch as pl
import torch

from omegaconf import DictConfig
from lightning.fabric.utilities.seed import seed_everything, reset_seed
from lightning.pytorch.loggers import Logger

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(config: DictConfig) -> dict:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    torch.set_float32_matmul_precision('high')
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed = config.seed
        seed_everything(config.seed, workers=True)
    else:
        rand_bytes = os.urandom(4)
        seed = int.from_bytes(rand_bytes, byteorder='little', signed=False)
        seed_everything(seed, workers=True)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    if os.name != 'nt':
        torch.compile(model)

    log.info("Instantiating callbacks...")
    callbacks: List[pl.Callback] = utils.instantiate_callbacks(config.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(config.get("logger"))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    if logger:
        log.info("Logging hyperparameters!")
        hparams = {
            'seed': seed,
            'config': config.copy(),
            'datamodule': datamodule,
            'model': model,
            'callbacks': callbacks,
            'trainer': trainer,
        }
        utils.log_hyperparameters(hparams=hparams, metrics=dict(config.metrics.metrics))

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        log.warning("Best ckpt not found! Using current weights for testing...")
        ckpt_path = None

    metric_dict = {config.get("optimized_metric"): trainer.checkpoint_callback.best_model_score}

    trainer.logger.log_metrics(
        {f'optimize_{config.get("optimized_metric")}': trainer.checkpoint_callback.best_model_score})

    if config.get("run_test"):
        reset_seed()
        log.info("Starting testing!")
        trainer.test(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
        metric_dict.update(trainer.callback_metrics)

    return metric_dict
