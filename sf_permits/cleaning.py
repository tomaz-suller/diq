from collections import defaultdict
from pathlib import Path
from string import punctuation
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from sf_permits.config import (
    INTERIM_DATASET_PATH,
    NEIGHBOURHOOD_SHAPEFILE_PATH,
    RAW_DATASET_PATH,
    STREET_NAMES_PATH,
    ZIP_CODE_SHAPEFILE_PATH,
    logger,
)
from sf_permits.utils.string_similarity import (
    get_matching_strings,
    jaccard,
    jaro_winkler,
)

app = typer.Typer()

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
DUPLICATE_IDENTIFIERS = [
    "Permit Number",
    "Street Name",
    "Street Number",
    "Street Number Suffix",
    "Unit",
    "Unit Suffix",
]
PUNCTUATION_REGEX = r"[{}]".format(punctuation)
STREET_NAME_JACCARD_JARO_SIMILARITY_THRESHOLD = 0.70
STREET_NAME_DIRECT_JARO_SIMILARITY_THRESHOLD = 0.93
STREET_NAME_REVERSE_JARO_SIMILARITY_THRESHOLD = 0.89


@app.command()
def main(
    input_path: Path = RAW_DATASET_PATH,
    output_path: Path = INTERIM_DATASET_PATH,
):
    logger.info("Starting data cleaning")
    logger.debug("Loading from {}", input_path)
    raw_df = pd.read_csv(input_path)
    logger.debug("Initial data has shape {}", raw_df.shape)
    # We start by deleting completely empty rows
    clean_df = raw_df.dropna(how="all", axis="index")
    logger.debug("Shape after dropping empty rows is {}", clean_df.shape)

    logger.info("Starting normalisation")
    # # Normalisation
    clean_df = clean_df.convert_dtypes()
    clean_df = decode_coordinates(clean_df)
    clean_df = string_to_lower_case(clean_df)
    clean_df = rename_columns(clean_df)
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

    # ## Using external street name data
    logger.info("Matching street names")
    clean_df = fix_street_name_spelling(clean_df)

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

    # ## Exploit approximate functional dependency between `Neighborhood` and `Supervisor District`
    clean_df = fill_district_based_on_neighbourhood(clean_df)
    report_missing_value_count(clean_df)

    logger.success("Missing value imputation complete")
    logger.info("Starting outlier detection")
    # Outlier detection
    # TODO
    logger.warning("Outlier detection not yet implemented")

    logger.success("Outlier detection complete")
    logger.info("Starting duplicate removal")
    # Duplicate removal
    clean_df = drop_duplicate_position_permits(clean_df)

    logger.success("Duplicate removal complete")

    logger.success("Data cleaning complete")
    logger.info("Saving clean data")
    logger.debug("Saving clean data to {}", output_path)
    clean_df.to_parquet(output_path)
    logger.success("Saved clean data")


def report_missing_value_count(df: pd.DataFrame) -> None:
    logger.debug("Missing value count \n{}", df.isna().sum())


def string_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    for column in MISSING_AS_FALSE_COLUMNS:
        df[column] = df[column].map({"Y": True, pd.NA: False}).astype("bool")
    return df


def drop_duplicate_position_permits(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=DUPLICATE_IDENTIFIERS)


def assign_na_to_missing_street_name(df: pd.DataFrame) -> pd.DataFrame:
    df[df["Street Name"].isin(MISSING_STREET_NAME)] = pd.NA
    return df


def assign_na_completion_to_incomplete_permit(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["Current Status"] != "complete", "Completed Date"] = pd.NA
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "Neighborhoods - Analysis Boundaries": "Neighborhood",
        }
    )


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


def fill_district_based_on_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    neighbourhood_district_records = (
        df[["Neighborhood", "Supervisor District"]]
        .dropna()
        .drop_duplicates("Neighborhood")  # Keep only a single neighbourhood per district
        .to_dict("records")
    )
    neighbourhood_district_map = {
        record["Neighborhood"]: record["Supervisor District"]
        for record in neighbourhood_district_records
    }
    for neighbourhood, district in neighbourhood_district_map.items():
        df.loc[df["Neighborhood"] == neighbourhood, "Supervisor District"] = district
        df.loc[df["Supervisor District"] == district, "Neighborhood"] = neighbourhood
    return df


def street_names_similar(base: str, target: str) -> bool:
    return (
        jaccard(base, target, normalise=False) >= 1
        and jaro_winkler(base, target) > STREET_NAME_JACCARD_JARO_SIMILARITY_THRESHOLD
    ) or (
        jaro_winkler(base, target) > STREET_NAME_DIRECT_JARO_SIMILARITY_THRESHOLD
        and jaro_winkler(base[::-1], target[::-1])
        > STREET_NAME_REVERSE_JARO_SIMILARITY_THRESHOLD
    )


def fix_street_name_spelling(df: pd.DataFrame) -> pd.DataFrame:
    street_names = df["Street Name"].str.replace(PUNCTUATION_REGEX, "", regex=True)

    logger.debug("Loading external street names from {}", STREET_NAMES_PATH)
    external_street_df = string_to_lower_case(
        pd.read_csv(STREET_NAMES_PATH).convert_dtypes()
    )
    normalised_external_street_df = external_street_df.copy()
    for string_column in normalised_external_street_df.select_dtypes("string"):
        normalised_external_street_df[string_column] = normalised_external_street_df[
            string_column
        ].str.replace(PUNCTUATION_REGEX, "", regex=True)
    # Python cannot infer the type, so we add the type hint
    normalised_external_street_df: pd.DataFrame

    normalised_external_street_df["StreetNameDirection"] = (
        normalised_external_street_df["StreetName"]
        + " "
        + normalised_external_street_df["PostDirection"].fillna("")
    )

    # Match external names with existing ones
    # based on the full street name (name, type and direction)...
    full_name_match_df = normalised_external_street_df.reset_index(
        names="base_index"
    ).merge(
        street_names.reset_index(), left_on="FullStreetName", right_on="Street Name"
    )
    # ... only the street name...
    street_name_match_df = normalised_external_street_df.reset_index(
        names="base_index"
    ).merge(street_names.reset_index(), left_on="StreetName", right_on="Street Name")
    # ... and the street name with direction
    street_name_direction_match_df = normalised_external_street_df.reset_index(
        names="base_index"
    ).merge(
        street_names.reset_index(),
        left_on="StreetNameDirection",
        right_on="Street Name",
    )
    match_indices = pd.concat(
        [
            full_name_match_df["index"],
            street_name_match_df["index"],
            street_name_direction_match_df["index"],
        ]
    ).unique()
    logger.debug("{} street names match exactly", len(match_indices))
    non_match_indices = df.index.difference(match_indices)
    logger.debug("{} street names did not match exactly", len(non_match_indices))
    # Streets which are not matched are assumed to be incorrectly spelled
    # so we use string similarity to match them
    wrong_street_names = street_names.loc[non_match_indices]

    # Dataset street names mix name and direction,
    # so we concatenate them in the external dataset before comparing
    external_street_names = (
        normalised_external_street_df["StreetName"]
        + " "
        + normalised_external_street_df["PostDirection"].fillna("")
    ).str.strip()

    logger.debug("Matching street names with string similarity")
    matching_indices, _ = get_matching_strings(
        external_street_names,
        wrong_street_names,
        street_names_similar,
        block_length=0,
    )

    for match_df in (
        full_name_match_df,
        street_name_direction_match_df,
        street_name_match_df,
    ):
        for base_index, index in match_df[["base_index", "index"]].itertuples(
            index=False
        ):
            try:
                matching_indices[base_index].append(index)
            except KeyError:
                matching_indices[base_index] = [index]

    # We build matches as a map from base index to target index
    # so we need to reverse it before continuing
    reversed_matching_indices = defaultdict(list)
    for base_index, target_indices in matching_indices.items():
        for target_index in target_indices:
            reversed_matching_indices[target_index].append(base_index)

    # Each target may be matched to multiple base indices
    # so we only keep the one with the highest similarity
    unique_reversed_matching_indices: dict[int, int] = {}
    for target_index, base_indices in reversed_matching_indices.items():
        # Skip computing similarity if there is only one base index
        if len(base_indices) == 1:
            unique_reversed_matching_indices[target_index] = base_indices[0]
            continue
        jaro_similarities = [
            jaro_winkler(
                external_street_names.loc[base_index],
                street_names.loc[target_index],
            )
            for base_index in base_indices
        ]
        unique_reversed_matching_indices[target_index] = base_indices[
            np.argmax(jaro_similarities)
        ]

    final_match_df = (
        pd.Series(unique_reversed_matching_indices, name="base_index")
        .reset_index()
        .rename(columns={"index": "target_index"})
    )

    # Once we have all the matches, we replace the values of
    # `Street Name` and `Street Type` with those from the
    # external dataset
    clean_street_df = final_match_df.join(
        external_street_df[["StreetName", "StreetType"]], on="base_index"
    ).join(df, on="target_index", how="right")
    clean_street_df["Street Name"] = clean_street_df["StreetName"].combine_first(
        clean_street_df["Street Name"]
    )
    clean_street_df["Street Suffix"] = clean_street_df["StreetType"].combine_first(
        clean_street_df["Street Suffix"]
    )
    clean_street_df = clean_street_df.drop(columns=["StreetName", "StreetType"])

    return clean_street_df


def replace_matching_geometry_values(
    df: pd.DataFrame, column: str, base: gpd.GeoDataFrame
) -> pd.DataFrame:
    geometry = gpd.GeoSeries.from_xy(df["longitude"], df["latitude"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        match_df = match(base, geometry)
    df[column] = match_df.iloc[:, 0].combine_first(df[column]).str.lower()
    return df


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


def string_to_lower_case(df: pd.DataFrame) -> pd.DataFrame:
    for string_column in df.select_dtypes("string"):
        df[string_column] = df[string_column].str.lower()
    return df


if __name__ == "__main__":
    app()
