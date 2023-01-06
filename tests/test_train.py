"""Test model training helpers."""

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from fastai.vision.learner import Learner
from pandas.testing import assert_frame_equal

from singing_classifier.train import gather_data, create_learner
from .util import create_wavfile_at


@pytest.fixture(name="learn_df")
def fx_learn_df() -> pd.DataFrame:
    """Returns dataframe for learner."""
    return pd.DataFrame(
        {
            "Chest": [0, 3, 1, 2],
            "path": ["d/a_0.m4a", "d/a_1.m4a", "d/b_0.m4a", "d/b_1.m4a"],
            "is_valid": [False, False, True, True],
        }
    )


def test_gather_data(tmp_path: Path, learn_df: pd.DataFrame):
    """Gathered data contains columns `target`, 'path' and 'is_valid'."""
    train = tmp_path / "train.parquet"
    valid = tmp_path / "valid.parquet"
    target = "Chest"

    train_content = """tag,seg_num,Head,Chest,song,path
a,0,2,0,foo,d/a_0.m4a
a,1,1,3,foo,d/a_1.m4a
"""
    valid_content = """tag,seg_num,Head,Chest,song,path
b,0,0,1,bar,d/b_0.m4a
b,1,3,2,bar,d/b_1.m4a
"""

    for text, dest in [
        (train_content, train),
        (valid_content, valid),
    ]:
        data = pd.read_csv(StringIO(text))
        data.to_parquet(dest)

    received = gather_data(train, valid, target)

    assert_frame_equal(received, learn_df)


def test_create_learner(tmp_path: Path, learn_df: pd.DataFrame):
    """A learner instance is returned"""
    learn_df["path"] = f"{tmp_path}/" + learn_df["path"]
    (tmp_path / "d").mkdir()
    for target_path in learn_df["path"]:
        create_wavfile_at(target_path)

    received = create_learner(
        learn_df=learn_df,
        target="Chest",
        sample_rate=32000,
        batch_duration_ms=1000,
        batch_size=12,
    )

    assert isinstance(received, Learner)
