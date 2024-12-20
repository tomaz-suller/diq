import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
import typer
from tqdm import tqdm

from sf_permits.config import (
    logger,
    RAW_DATASET_PATH,
    INTERIM_DATASET_PATH,
    NEIGHBOURHOOD_SHAPEFILE_PATH,
    ZIP_CODE_SHAPEFILE_PATH,
)

app = typer.Typer()

STREET_SUFFIX_MAP = {
    "av": ["avenue", "ave"],
    "wy": ["via"],
    "bl": ["blvd"],
}
MISSING_AS_FALSE_COLUMNS = [
    "Fire Only Permit",
    "Structural Notification",
    "Site Permit",
    "Voluntary Soft-Story Retrofit",
]
MISSING_STREET_NAME = [
    "situs to be assigned",
    "unknown",
]


def decode_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    coordinates = (
        df["Location"]
        .str.replace("(", "")
        .str.replace(")", "")
        .str.split(",", expand=True)
        .astype("float")
    )
    df["latitude"] = coordinates[0]
    df["longitude"] = coordinates[1]
    return df


def remove_street_suffix_from_name(df: pd.DataFrame) -> pd.DataFrame:
    for index, street_name in tqdm(
        df["Street Name"].items(), total=df.shape[0], desc="Street"
    ):
        street_name: str

        if not pd.isna(df.loc[index, "Street Suffix"]):
            continue

        logger.debug(
            "Index {} and street name '{}' has null suffix", index, street_name
        )
        for suffix, aliases in STREET_SUFFIX_MAP.items():
            for alias in aliases:
                if alias in street_name:
                    logger.debug("Street name matches alias '{}'", alias)
                    df.loc[index, "Street Suffix"] = suffix
                    df.loc[index, "Street Name"] = (
                        street_name.replace(alias, "").replace("  ", " ").strip()
                    )

        return df


def assign_na_to_missing_street_name(df: pd.DataFrame) -> pd.DataFrame:
    df[df["Street Name"].isin(MISSING_STREET_NAME)] = pd.NA
    return df


def assign_na_completion_to_incomplete_permit(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["Current Status"] != "complete", "Completed Date"] = pd.NA
    return df


def string_to_lower_case(df: pd.DataFrame) -> pd.DataFrame:
    for string_column in df.select_dtypes("string"):
        df[string_column] = df[string_column].str.lower()
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "Neighborhoods - Analysis Boundaries": "Neighborhood",
        }
    )


def match(base: gpd.GeoDataFrame, target: gpd.GeoSeries) -> pd.DataFrame:
    """
    Match points in target geometry to regions in base geometry.

    Assumes that `base` contains a single column (other than the geometry)
    specifying a label which should be associated with each point
    in `target`.
    """
    match_df = pd.DataFrame(index=target.index)
    for id_, geometry in tqdm(
        base.itertuples(index=False), total=base.shape[0], desc="Geometry"
    ):
        match_df[id_] = target.within(geometry)
    return pd.from_dummies(match_df, default_category=pd.NA)


def replace_matching_geometry_values(
    df: pd.DataFrame, column: str, base: gpd.GeoDataFrame
) -> pd.DataFrame:
    geometry = gpd.GeoSeries.from_xy(df["longitude"], df["latitude"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        match_df = match(base, geometry)
    df[column] = match_df.iloc[:, 0].combine_first(df[column]).str.lower()
    return df


def impute(
    df: pd.DataFrame, mean_columns: list[str] = [], mode_columns: list[str] = []
) -> pd.DataFrame:
    imputed_df = df.copy()
    for column in mean_columns:
        imputed_df.loc[:, column] = imputed_df[column].fillna(
            imputed_df[column].mean(skipna=True)
        )
    for column in mode_columns:
        imputed_df.loc[:, column] = imputed_df[column].fillna(
            imputed_df[column].mode(dropna=True)
        )
    return imputed_df


def impute_group(
    df: pd.DataFrame,
    group: str | list[str],
    mean_columns: list[str] = [],
    mode_columns: list[str] = [],
) -> pd.DataFrame:
    if isinstance(group, list):
        na_mask = df[group].isna().any(axis="columns")
    else:
        na_mask = df[group].isna()
    group_df_list = [df[na_mask]]
    for _, group_df in tqdm(df.groupby(group), desc="Group"):
        group_df_list.append(impute(group_df, mean_columns, mode_columns))
    group_df = pd.concat(group_df_list)
    return group_df


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


def string_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    for column in MISSING_AS_FALSE_COLUMNS:
        df[column] = df[column].map({"Y": True, pd.NA: False}).astype("bool")
    return df


def report_missing_value_count(df: pd.DataFrame) -> None:
    logger.debug("Missing value count \n{}", df.isna().sum())

@app.command()
def main(
    input_path: Path = RAW_DATASET_PATH,
    output_path: Path = INTERIM_DATASET_PATH,
):
    logger.info("Starting data cleaning")
    logger.debug("Loading from {}", input_path)
    raw_df = pd.read_csv(input_path)
    logger.debug("Initial data has shape {}", raw_df.shape)
    # We start from deleting completely empty rows
    clean_df = raw_df.dropna(how="all", axis="index")
    logger.debug("Shape after dropping empty rows is {}", clean_df.shape)

    logger.info("Starting normalisation")
    # # Normalisation
    clean_df = clean_df.convert_dtypes()
    clean_df = decode_coordinates(clean_df)
    clean_df = string_to_lower_case(clean_df)
    clean_df = rename_columns(clean_df)
    clean_df = remove_street_suffix_from_name(clean_df)
    clean_df = assign_na_to_missing_street_name(clean_df)
    clean_df = string_to_datetime(clean_df)

    logger.success("Normalisation complete")
    logger.info("Starting error correction")
    # # Error correction

    # Some permits are not assigned the "Complete" status yet
    # are assigned a completion date
    # We treat these as errors and so assign NA to them
    clean_df = assign_na_completion_to_incomplete_permit(clean_df)

    # ## Using external location-based data
    neighbourhood_gdf: gpd.GeoDataFrame = gpd.read_file(NEIGHBOURHOOD_SHAPEFILE_PATH)
    logger.info("Matching neighbourhood geometries")
    clean_df = replace_matching_geometry_values(
        clean_df, "Neighborhood", neighbourhood_gdf
    )

    logger.info("Matching zipcode geometries")
    zipcode_gdf: gpd.GeoDataFrame = gpd.read_file(
        ZIP_CODE_SHAPEFILE_PATH, columns=["zip"]
    )
    clean_df = replace_matching_geometry_values(clean_df, "Zipcode", zipcode_gdf)

    logger.success("Error correction complete")
    logger.info("Starting missing value imputation")
    # # Missing value imputation
    report_missing_value_count(clean_df)

    clean_df = string_to_boolean(clean_df)
    report_missing_value_count(clean_df)

    # ## Fill location using `Block` and `Lot`
    logger.info("Imputing based on block and lot")
    clean_df = impute_group(
        clean_df,
        ["Block", "Lot"],
        mean_columns=("latitude", "longitude"),
        mode_columns=(
            "Street Name",
            "Street Suffix",
            "Supervisor District",
        ),
    )
    report_missing_value_count(clean_df)

    # ## Fill location using `Street Name`
    logger.info("Imputing based on street name")
    clean_df = impute_group(
        clean_df,
        "Street Name",
        mean_columns=("latitude", "longitude"),
        mode_columns=(
            "Street Suffix",
            "Supervisor District",
        ),
    )
    report_missing_value_count(clean_df)

    # After imputing the location, we are able to use it to correct
    # and impute `Neighborhood` and `Zipcode` as we did before
    # so we apply the same function we did for error correction again
    logger.info("Reapplying neighbourhood matching")
    clean_df = replace_matching_geometry_values(
        clean_df, "Neighborhood", neighbourhood_gdf
    )
    logger.info("Reapplying zipcode matching")
    clean_df = replace_matching_geometry_values(clean_df, "Zipcode", zipcode_gdf)
    report_missing_value_count(clean_df)

    logger.success("Error correction complete")
    # Outlier detection
    # TODO

    # Duplicate removal
    # TODO

    logger.success("Data cleaning complete")
    logger.info("Saving cleaned data")
    logger.debug("Saving to {}", output_path)
    clean_df.to_parquet(output_path)
    logger.success("Saved clean data")


if __name__ == "__main__":
    app()
