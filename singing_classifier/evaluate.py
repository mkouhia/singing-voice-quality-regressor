"""Test how well model performs on test set."""

import argparse
import json
from collections.abc import Collection, Sequence, Generator
from pathlib import Path

import fastai.metrics
import numpy as np
import pandas as pd
import torch
from fastai.learner import Metric
from fastai.torch_core import TensorBase
from fastai.vision.learner import Learner, load_learner


def evaluate(
    test: Path,
    model: Path,
    out_json: Path,
    targets: Sequence[str],
    metrics: Collection[str] = None,
    out_csv_root: Path = None,
):
    """Evaluate model performance, write output to `out_json`.

    Args:
        test: Path to test parquet file.
        model: Path to trained model file.
        out_json: Metrics json output path.
        targets: Names of target columns in test. This is required to
          correspond to model output columns.
        metrics: Metric names to use in evaluation. If empty or None,
          the metrics are acquired from the `learn` object.
        out_csv_root: Path to output csv directory, where for each
          target, data will be saved in file `{target}.csv`
          under columns 'predicted' and 'actual'. If None, do not create
          this output file.
    """
    learn = load_learner(model)
    data = pd.read_parquet(test)

    actual = data[list(targets)].astype(float).to_numpy()
    preds = get_predictions(learn, paths=data["path"].tolist())

    result = evaluate_metrics(
        actual_labels=actual,
        predicted=preds,
        metrics=metrics or learn.metrics,
    )

    with out_json.open("w", encoding="utf-8") as file_:
        json.dump(result, file_)
        file_.write("\n")

    if out_csv_root is not None:
        out_csv_root.mkdir(parents=True, exist_ok=True)
        for label, summary_df in zip(targets, to_comparison(preds, actual)):
            summary_df.to_csv(out_csv_root / f"{label}.csv", index=False)


def get_predictions(
    learn: Learner,
    paths: list,
) -> TensorBase:
    """Get predictions from model.

    Args:
        learn: Fast.ai learner object.
        paths: List of audio file locations, on which to create
          predictions.

    Returns:
        Received predictions
    """
    test_dl = learn.dls.test_dl(paths)
    preds, _ = learn.get_preds(dl=test_dl)
    return preds


def evaluate_metrics(
    predicted: TensorBase,
    actual_labels: np.ndarray,
    metrics: Collection[str] | Collection[Metric] = None,
) -> dict[str, float]:
    """Evaluate model performance

    Args:
        actual_labels: Expected labels that correspond to `paths`.
        metrics: Metrics to use in evaluation.

    Returns:
        Dictionary of output metrics and their values.metrics
    """
    actual_tensor = torch.from_numpy(actual_labels)

    met_objects = []
    for name in metrics or []:
        name_stem = name if "(" not in name else name[: name.index("(")]
        if not hasattr(fastai.metrics, name_stem):
            raise UserWarning("Unknown metric: " + name_stem)

        met_obj = getattr(fastai.metrics, name_stem)

        if name_stem != name:
            if not name.endswith("()"):
                raise UserWarning("Metric parameter parsing not implemented")
            met_obj = met_obj()

        met_objects.append(met_obj)

    return {met.name: float(met(predicted, actual_tensor)) for met in met_objects}


def to_comparison(
    predictions: torch.Tensor | np.ndarray,
    actual: torch.Tensor | np.ndarray,
) -> Generator[pd.DataFrame, None, None]:
    """Generate comparison dataframe

    Args:
        predictions: Predicted values.
        actual: Actual values.

    Yields:
        Dataframes formed from predictions and actual. Each dataframe
        contains two columns: 'predicted' and 'actual'.
    """
    for i in range(predictions.shape[1]):
        yield pd.DataFrame({"predicted": predictions[:, i], "actual": actual[:, i]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        required=True,
        help="Names of target columns in test dataset.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        help="List of metric names to evaluate. If not specified, use loaded model metrics.",
    )
    parser.add_argument(
        "--out-json", type=Path, help="Output metrics json path", required=True
    )
    parser.add_argument("--out-csv-root", type=Path, help="Path to output csv root.")
    parser.add_argument("test", type=Path, help="Test Parquet file")
    parser.add_argument("model", type=Path, help="Model file")

    evaluate(**vars(parser.parse_args()))
