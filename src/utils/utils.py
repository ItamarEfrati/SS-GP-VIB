import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List

import hydra
from omegaconf import DictConfig
from lightning import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(config: DictConfig, *args):

        # apply extra utilities
        extras(config)

        # execute the task
        n_errors = 0

        metric_dict = {config.get("optimized_metric"): 0}
        while n_errors < 3:
            metric_dict = task_func(config, *args)
            try:
                start_time = time.time()

                n_errors = 4
            except ValueError as ex:
                log.error("value error")  # save exception to `.log` file
                n_errors += 1
                continue
            except Exception as ex:
                log.exception("")  # save exception to `.log` file
                raise ex
            finally:
                path = Path(config.paths.output_dir, "exec_time.log")
                content = f"'{config.datamodule_name}_{config.model_name}' execution time: {time.time() - start_time} (s)"
                save_file(path, content)  # save task execution time (even if exception occurs)
                close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {config.paths.output_dir}")

        return metric_dict

    return wrap


def extras(config: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not config.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if config.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if config.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(config, save_to_file=True)

    # pretty print config tree using Rich library
    if config.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(config, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_config: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_config:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_config.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(hparams: dict, metrics: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """
    new_hparams = {}
    config = hparams['config']
    model = hparams['model']
    trainer = hparams['trainer']

    new_hparams['seed'] = hparams['seed']
    new_hparams['batch_size'] = config['datamodule']['batch_size']
    # new_hparams['aggregation'] = config['datamodule']['aggregation_minutes']
    new_hparams['model'] = config['model']

    # save number of model parameters
    new_hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    new_hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    new_hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    if 'timeseries_encoder' in new_hparams['model'].keys():
        paddings_keys = [i for i in new_hparams['model']['timeseries_encoder'].keys() if "padding" in i]
        for k in paddings_keys:
            new_hparams['model']['timeseries_encoder'][k] = str(new_hparams['model']['timeseries_encoder'][k])

    paddings_keys = [i for i in new_hparams['model']['decoder'].keys() if "padding" in i]
    for k in paddings_keys:
        new_hparams['model']['decoder'][k] = str(new_hparams['model']['decoder'][k])

    if 'init_parameters' in config['callbacks']:
        new_hparams['init_method'] = config['callbacks']['init_parameters']['init_method']

    # send hparams to all loggers
    trainer.logger.log_hyperparams(new_hparams, metrics)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name]
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
