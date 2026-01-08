import argparse
import logging

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pydantic import BaseModel, Field

from mlx_timeseries_workbench import callbacks, data
from mlx_timeseries_workbench.mlx_models import autoencoders, base

logger = logging.getLogger(__name__)


class Params(BaseModel):
    cpu: bool = Field(False, description="use cpu")
    seed: int = Field(123, description="random seed")
    batch_size: int = Field(256, description="batch size")
    learning_rate: float = Field(1e-3, description="learning rate")
    epochs: int = Field(40, description="number of epochs")
    log_dir: str = Field(description="log directory")


def add_model(parser: argparse.ArgumentParser, model: type[BaseModel]) -> None:
    fields = model.model_fields
    for name, field_info in fields.items():
        if field_info.annotation is bool:
            parser.add_argument(
                f"--{name}",
                dest=name,
                action="store_true",
                required=field_info.is_required(),
                default=field_info.default,
                help=field_info.description,
            )
        else:
            parser.add_argument(
                f"--{name}",
                dest=name,
                type=field_info.annotation,  # type: ignore
                required=field_info.is_required(),
                default=field_info.default,
                help=field_info.description,
            )


def __data_normalization(
    train_data: np.ndarray, test_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    min_val = np.min(train_data)
    max_val = np.max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data.astype(np.float32)
    test_data.astype(np.float32)

    return train_data, test_data


def __split_dataset(
    data: np.ndarray, validation_fraction: float = 0.02, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    index: np.ndarray = np.arange(len(data))
    rng.shuffle(index)

    t = int(validation_fraction * len(data))
    validation_set = data[:t]
    train_set = data[t:]

    assert len(data) == len(train_set) + len(validation_set)
    return train_set, validation_set


def main(pars: Params) -> None:
    logger = logging.getLogger(__name__)
    train_data, test_data, train_labels, test_labels = data.get_tensorflow_ECG5000()

    logger.debug("Normalizing data")
    train_data, test_data = __data_normalization(train_data, test_data)

    normal_train_data = train_data[train_labels == 0]

    # split train set to get a validation
    logger.info("Splitting train/validation set")
    np_train_set, np_validation_set = __split_dataset(normal_train_data)

    logger.info(
        f"train size: {len(np_train_set)} - validation size: {len(np_validation_set)}"
    )

    train_set: mx.array = mx.array(np_train_set)
    validation_set: mx.array = mx.array(np_validation_set)

    logger.info("Creating autoencoder")
    autoencoder = autoencoders.FixedLengthFullyConnectedAutoEncoder()
    mx.eval(autoencoder.parameters())

    logger.info(autoencoder)

    t = base.Trainer(autoencoder)
    t.compile(optimizer=optim.Adam(learning_rate=pars.learning_rate), loss=loss_fn)

    t.fit(
        train_set,
        train_set,
        epochs=pars.epochs,
        batch_size=pars.batch_size,
        validation_set=(validation_set, validation_set),
        verbose=1,
        callbacks=[callbacks.TensorBoardLogger(log_dir=pars.log_dir)],
    )


def loss_fn(model: nn.Module, X: mx.array, y: mx.array) -> mx.array:
    t: mx.array = nn.losses.l1_loss(model(X), y, reduction="mean")
    return t


if __name__ == "__main__":
    logging.basicConfig(
        format="%(filename)s: %(levelname)s: %(funcName)s(): %(lineno)d:\t%(message)s"
    )

    data.logger.setLevel(logging.DEBUG)
    autoencoders.logger.setLevel(logging.DEBUG)
    base.logger.setLevel(logging.INFO)
    callbacks.logger.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_model(parser, Params)

    args = parser.parse_args()
    pars = Params(**vars(args))
    logger.info(pars)

    if pars.cpu:
        mx.set_default_device(mx.cpu)  # type: ignore

    np.random.seed(pars.seed)
    mx.random.seed(pars.seed)

    main(pars)
