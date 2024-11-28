from pathlib import Path

import pandas as pd
import geopandas as gpd
import typer
from loguru import logger

from sf_permits.config import RAW_DATASET_PATH, CLEAN_DATASET_PATH

app = typer.Typer()


def string_to_datetime(
    df: pd.DataFrame,
    like: str = "Date",
    format: str = r"%m/%d/%Y",
) -> pd.DataFrame:
    date_df = df.filter(like=like)
    logger.debug("Identified date columns: {}", date_df.columns)
    date_df = date_df.apply(pd.to_datetime, errors="raise", format=format)
    for date_column in date_df:
        df[date_column] = date_df[date_column]
    logger.success("Date columns converted to datetime")
    return df


def coordinates_to_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
    location_series = df.pop("Location")
    coordinates = (
        location_series.str.replace("(", "")
        .str.replace(")", "")
        .str.split(",", expand=True)
        .dropna()
    )
    location_geoseries = gpd.GeoSeries.from_xy(coordinates[0], coordinates[1])
    gdf = gpd.GeoDataFrame(df, geometry=location_geoseries)
    logger.success("Coordinates converted to geometry")
    logger.info("Returning GeoDataFrame")
    return gdf


def to_categorical(
    series: pd.Series,
    categories: list | None = None,
    ordered: bool = False,
) -> pd.Series:
    categorical_dtype = pd.CategoricalDtype(categories=categories, ordered=ordered)
    return series.astype(categorical_dtype)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame: ...


@app.command()
def main(
    input_path: Path = RAW_DATASET_PATH,
    output_path: Path = CLEAN_DATASET_PATH,
):
    logger.info("Starting data cleaning")
    logger.debug("Loading from {}", input_path)
    raw_df = pd.read_csv(input_path)
    clean_df = raw_df.convert_dtypes()

    # Normalisation
    clean_df = string_to_datetime(clean_df)
    clean_df = coordinates_to_geometry(clean_df)

    # Error correction
    # TODO

    # Outlier detection
    # TODO

    # Duplicate removal
    # TODO

    logger.success("Data cleaning complete")
    logger.info("Saving cleaned data")
    logger.debug("Saving to {}", output_path)
    clean_df.to_parquet(output_path)


if __name__ == "__main__":
    app()
