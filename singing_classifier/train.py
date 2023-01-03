"""Train fast.ai model."""

import argparse
from pathlib import Path

import pandas as pd
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import ColReader, ColSplitter
from fastai.metrics import RocAuc, RocAucBinary, accuracy
from fastai.vision.all import vision_learner
from fastai.vision.learner import Learner
from fastai.vision.models import resnet18
from fastaudio.augment.preprocess import RemoveSilence, Resample
from fastaudio.augment.signal import DownmixMono, ResizeSignal
from fastaudio.core.config import AudioBlock, AudioConfig
from fastaudio.core.spectrogram import AudioToSpec


def gather_data(
    train: Path, valid: Path, segment_summary: Path, target: str
) -> pd.DataFrame:
    """Form a dataframe for model training.

    Args:
        train: Path to train dataframe Parquet file.
        valid: Path to test dataframe Parquet file.
        segment_summary: Path to segment summary Parquet file.
        target: Name of target column to be included in the result.

    Returns:
        Joined dataframe, with columns `target`, 'path' and 'is_valid'.
    """
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

    return pd.concat(
        [
            _add_paths(pd.read_parquet(data_path)).assign(is_valid=is_valid_)
            for is_valid_, data_path in [(False, train), (True, valid)]
        ],
        axis=0,
        ignore_index=True,
    )[[target, "path", "is_valid"]]


def create_learner(
    learn_df,
    target: str,
    sample_rate: int,
    batch_duration_ms: int,
    batch_size: int,
) -> Learner:
    """Create fast.ai learner for training.

    Args:
        learn_df: Input dataframe, containing columns `target`, 'path'
          and 'is_valid'.
        target: Target column, containing classification label.
        sample_rate: Audio sample rate to use in learner.
        batch_duration_ms: Audio sample duration for batches.
        batch_size: Number of samples in a batch.

    Returns:
        Fast.ai learner.
    """
    cfg = AudioConfig.BasicMelSpectrogram(sample_rate=sample_rate, n_fft=512)
    a2s = AudioToSpec.from_cfg(cfg)

    dblock = DataBlock(
        blocks=(AudioBlock, CategoryBlock),
        splitter=ColSplitter("is_valid"),
        get_x=ColReader("path"),
        get_y=ColReader(target),
        item_tfms=[
            RemoveSilence(),
            ResizeSignal(duration=batch_duration_ms),
            DownmixMono(),
            Resample(sample_rate),
        ],
        batch_tfms=[a2s],
    )

    dls = dblock.dataloaders(learn_df, bs=batch_size)  # verbose=True

    num_classes = len(learn_df[target].unique().tolist())
    metrics = [accuracy] + [RocAucBinary() if num_classes == 2 else RocAuc()]

    return vision_learner(
        dls,
        resnet18,
        n_in=1,  # <- Number of audio channels
        #     loss_func=CrossEntropyLossFlat(),
        metrics=metrics,
    )


def train_learner(
    train: Path,
    valid: Path,
    segment_summary: Path,
    target: str,
    model: Path,
    epochs: int,
    sample_rate: int,
    batch_duration_ms: int,
    batch_size: int,
):
    """Train fast.ai learner and save it to disk.

    Args:
        train: Path to train dataframe Parquet file.
        valid: Path to test dataframe Parquet file.
        segment_summary: Path to segment summary Parquet file.
        target: Name of target column to be included in the result.
        model: Model output file location.
        epochs: Number of epochs for the model.
        sample_rate: Audio sample rate to use in learner.
        batch_duration_ms: Audio sample duration for batches.
        batch_size: Number of samples in a batch.

    """
    learn_data = gather_data(train, valid, segment_summary, target)
    learner = create_learner(
        learn_data, target, sample_rate, batch_duration_ms, batch_size
    )

    lr_suggested = learner.lr_find(show_plot=False)
    learner.fine_tune(epochs, lr_suggested.valley)
    learner.export(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-duration-ms", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--segment_summary", type=Path, required=True)
    parser.add_argument("target", type=str)
    parser.add_argument("model", type=Path)

    namespace = parser.parse_args()
    train_learner(**vars(namespace))
