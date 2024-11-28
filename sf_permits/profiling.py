import json
from typing import Callable, TypeVar
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from sf_permits.config import PROFILING_DATA_DIR, RAW_DATA_DIR


def original_dtypes(df: pd.DataFrame) -> dict:
    return df.dtypes.astype("string").to_dict()


def inferred_dtypes(df: pd.DataFrame) -> dict:
    return df.convert_dtypes().dtypes.astype("string").to_dict()


def correlation(df: pd.DataFrame) -> dict:
    return df.corr(numeric_only=True).to_dict()


def size(input_: pd.Series | pd.DataFrame) -> np.ndarray:
    return np.prod(input_.shape)


def duplication(input_: pd.Series | pd.DataFrame) -> int | None:
    """Assess number of duplicated tuples."""
    input_ = input_.dropna()
    if input_.empty:
        return None
    return (input_.duplicated().sum() / size(input_)).item()


def completeness(input_: pd.Series | pd.DataFrame) -> int:
    """Assess number of missing values."""
    return 1 - (input_.isna().sum().sum() / size(input_)).item()


def interestingness(input_: pd.Series | pd.DataFrame) -> float | None:
    """
    Assess variability of attribute.

    We define variability as the ration between the frequency
    of the most common value and the frequency of the least
    common value.
    """
    value_counts = input_.value_counts(sort=True, ascending=False)
    # TODO Investigate why `df.value_counts()` returns an empty Series
    if value_counts.empty:
        return None
    return (value_counts.iloc[0] / value_counts.iloc[-1]).item()


# def accuracy():
#     """Assess syntactic accuracy."""
#     ...


def uniqueness(series: pd.Series, cutoff: int = 9) -> dict[str, int]:
    """Assess number of unique values."""
    counts = series.value_counts(normalize=True, sort=True, ascending=False, dropna=True)
    other_count = counts.iloc[cutoff:].sum()
    counts = counts.iloc[:cutoff]
    counts["other"] = other_count
    return counts.to_dict()


T = TypeVar("T")


def numeric_guard(func: Callable[..., T]) -> T | None:
    """Decorator to check if attribute is numeric."""

    def wrapper(series: pd.Series, *args, **kwargs):
        if not pd.api.types.is_numeric_dtype(series.dtype):
            logger.warning(f"Attribute {series.name} is not numeric.")
            return None
        return func(series, *args, **kwargs)

    wrapper.__name__ = func.__name__
    return wrapper


@numeric_guard
def distribution(series: pd.Series) -> dict[str, float]:
    """Assess distribution of numeric attribute."""
    return series.describe().to_dict()


@numeric_guard
def inter_quantile_outliers(series: pd.Series, factor: float = 1.5) -> list[int]:
    """Return indices of outliers according to interquartile range."""
    quantiles = series.quantile([0.25, 0.5, 0.75])
    inter_quantile_range = quantiles[0.75] - quantiles[0.25]
    return np.nonzero(np.abs(series - quantiles[0.5]) > factor * inter_quantile_range)[
        0
    ].tolist()


@numeric_guard
def standard_deviation_outliers(series: pd.Series, factor: float = 3) -> list[int]:
    """Return indices of outliers according to standard deviation."""
    return np.nonzero(np.abs(series - series.mean()) > factor * series.std())[
        0
    ].tolist()


DATAFRAME_METRICS = {
    original_dtypes,
    inferred_dtypes,
    correlation,
    duplication,
    completeness,
    interestingness,
}

SERIES_METRICS = {
    duplication,
    completeness,
    interestingness,
    uniqueness,
    distribution,
    inter_quantile_outliers,
    standard_deviation_outliers,
}

ALL_METRICS = DATAFRAME_METRICS | SERIES_METRICS

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "building_permits.csv",
    output_dir: Path = PROFILING_DATA_DIR,
):
    logger.info("Starting data profiling")
    logger.debug("Loading data from {} and saving to {}", input_path, output_dir)
    df = pd.read_csv(input_path)

    for metric in tqdm(ALL_METRICS, desc="Metric"):
        metric_name = metric.__name__
        logger.info("Computing metric {}", metric_name)
        result = {}

        if metric in DATAFRAME_METRICS:
            logger.debug("Computing over entire dataset")
            if (table_result := metric(df)) is not None:
                result["table"] = table_result

        if metric in SERIES_METRICS:
            attribute_results = {}
            for column in tqdm(df.columns, desc="Attribute"):
                logger.debug("Computing over attribute {}", column)
                if (attribute_result := metric(df[column])) is not None:
                    attribute_results[column] = attribute_result
            result["attributes"] = attribute_results

        output_path = output_dir / f"{metric_name}.json"
        if output_path.exists():
            logger.warning("File {} already exists, overwriting", output_path)
        logger.debug("Saving results to {}", output_path)
        with output_path.open("w") as f:
            json.dump(result, f)

        logger.success(f"Done computing metric {metric_name}")


if __name__ == "__main__":
    app()
