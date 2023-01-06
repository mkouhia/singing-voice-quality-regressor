"""Train fast.ai model."""

import argparse
import json
import logging
from collections.abc import Collection
from os import PathLike
from pathlib import Path
from matplotlib import pyplot as plt

import pandas as pd
from fastai.data.block import DataBlock, RegressionBlock
from fastai.data.transforms import ColReader, ColSplitter
from fastai.losses import L1LossFlat
from fastai.vision.all import vision_learner
from fastai.vision.learner import Learner
from fastai.vision.models import resnet18
from fastaudio.augment.preprocess import RemoveSilence, Resample
from fastaudio.augment.signal import DownmixMono, ResizeSignal
from fastaudio.core.config import AudioBlock, AudioConfig
from fastaudio.core.spectrogram import AudioToSpec


logger = logging.getLogger(__name__)


def gather_data(
    train: Path,
    valid: Path,
    segment_summary: Path,
    targets: Collection[str],
    target_astype: str = None,
) -> pd.DataFrame:
    """Form a dataframe for model training.

    Args:
        train: Path to train dataframe Parquet file.
        valid: Path to test dataframe Parquet file.
        segment_summary: Path to segment summary Parquet file.
        targets: Names of target columns to be included in the result.
        target_astype: Convert targets to this type.

    Returns:
        Joined dataframe, with columns `target`, 'path' and 'is_valid'.
    """
    logger.info("Gathering data")
    segments = pd.read_parquet(segment_summary)
    segments = segments[~segments["path"].isna()][["tag", "num", "path"]]

    def _add_paths(data):
        return pd.merge(
            data,
            segments,
            left_on=["tag", "seg_num"],
            right_on=["tag", "num"],
            how="inner",
        ).drop(columns=["num"])

    ret = pd.concat(
        [
            _add_paths(pd.read_parquet(data_path)).assign(is_valid=is_valid_)
            for is_valid_, data_path in [(False, train), (True, valid)]
        ],
        axis=0,
        ignore_index=True,
    )[list(targets) + ["path", "is_valid"]]

    if target_astype is not None:
        for target_ in targets:
            ret[target_] = ret[target_].astype(target_astype)

    return ret


def create_learner(
    learn_df,
    targets: Collection[str],
    sample_rate: int,
    batch_duration_ms: int,
    batch_size: int,
    n_fft: int = 400,
) -> Learner:
    """Create fast.ai learner for training.

    Args:
        learn_df: Input dataframe, containing columns `target`, 'path'
          and 'is_valid'.
        targets: Names of target columns, containing target values.
        sample_rate: Audio sample rate to use in learner.
        batch_duration_ms: Audio sample duration for batches.
        batch_size: Number of samples in a batch.
        n_fft: Size of FFT

    Returns:
        Fast.ai learner.
    """
    logger.info("Creating learner")

    cfg = AudioConfig.BasicMelSpectrogram(sample_rate=sample_rate, n_fft=n_fft)
    a2s = AudioToSpec.from_cfg(cfg)

    dblock = DataBlock(
        blocks=(AudioBlock, RegressionBlock(n_out=len(targets))),
        splitter=ColSplitter("is_valid"),
        get_x=ColReader("path"),
        get_y=ColReader(targets),
        item_tfms=[
            RemoveSilence(),
            ResizeSignal(duration=batch_duration_ms),
            DownmixMono(),
            Resample(sample_rate),
        ],
        batch_tfms=[a2s],
    )

    dls = dblock.dataloaders(learn_df, bs=batch_size)  # verbose=True

    # TODO add output clamping, e.g. like https://forums.fast.ai/t/image-regression-using-fastai/27784/22?u=mosscoder
    return vision_learner(
        dls,
        resnet18,
        n_in=1,  # <- Number of audio channels
        loss_func=L1LossFlat(),
    )


def train_learner(
    train: Path,
    valid: Path,
    segment_summary: Path,
    targets: Collection[str],
    model: Path,
    epochs: int,
    sample_rate: int,
    batch_duration_ms: int,
    batch_size: int,
    n_fft: int,
    metrics: Path = None,
    lr_plot: Path = None,
    loss_plot: Path = None,
):
    """Train fast.ai learner and save it to disk.

    Args:
        train: Path to train dataframe Parquet file.
        valid: Path to test dataframe Parquet file.
        segment_summary: Path to segment summary Parquet file.
        targets: Names of target columns to be included in the result.
        model: Model output file location.
        epochs: Number of epochs for the model.
        sample_rate: Audio sample rate to use in learner.
        batch_duration_ms: Audio sample duration for batches.
        batch_size: Number of samples in a batch.
        metrics: Model metrics output file location.
        lr_plot: Learning rate plot output file location.
        lr_plot: Loss plot output file location.
    """
    logger.info("Train learner with validation data")
    learn_data = gather_data(train, valid, segment_summary, targets, "float")
    learner = create_learner(
        learn_data, targets, sample_rate, batch_duration_ms, batch_size, n_fft
    )

    logger.info("Find learning rate")
    lr_suggested = learner.lr_find(show_plot=bool(lr_plot))
    logger.info("Suggested learning rate: %.2E", lr_suggested)

    if lr_plot:
        plt.savefig(lr_plot)
        plt.clf()

    logger.info("Perform fine tuning")
    learner.fine_tune(epochs, lr_suggested.valley)

    if loss_plot:
        learner.recorder.plot_loss()
        plt.savefig(loss_plot)
        plt.clf()

    if metrics:
        logger.info("Save metrics")
        metric_values = {i.name: float(i.value) for i in learner.metrics}
        with metrics.open("w", encoding="utf-8") as file_:
            json.dump(metric_values, file_, indent=2)
            file_.write("\n")

    logger.info("Export learner to %s", model)
    learner.export(model)

    logger.info("Ready!")


def _path_ensure_parent(path_like: PathLike) -> Path:
    ret = Path(path_like)
    ret.parent.mkdir(parents=True, exist_ok=True)
    return ret


if __name__ == "__main__":

    log_fmt = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-duration-ms", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--segment_summary", type=Path, required=True)
    parser.add_argument("--metrics", type=_path_ensure_parent)
    parser.add_argument("--lr-plot", type=_path_ensure_parent)
    parser.add_argument("--loss-plot", type=_path_ensure_parent)
    parser.add_argument("--targets", type=str, nargs="+", required=True)
    parser.add_argument("model", type=_path_ensure_parent)

    namespace = parser.parse_args()
    train_learner(**vars(namespace))
