import os
from typing import List

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from lightning.fabric.utilities.seed import seed_everything, reset_seed
from lightning.pytorch.loggers import Logger


from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(config: DictConfig) -> dict:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        dict: Dict with metrics
    """

    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    else:
        rand_bytes = os.urandom(4)
        config['seed'] = int.from_bytes(rand_bytes, byteorder='little', signed=False)
        seed_everything(config.seed, workers=True)

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
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)

    ckpt_path = config.get("ckpt_path")

    if ckpt_path is None:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

    metrics_dict = {'seed': config['seed']}

    log.info("Last validation!")
    reset_seed()
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    metrics_dict.update(trainer.callback_metrics)

    log.info("Starting testing!")
    reset_seed()
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    metrics_dict.update(trainer.callback_metrics)

    return metrics_dict
