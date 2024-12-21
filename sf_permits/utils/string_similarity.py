from collections import defaultdict

import pandas as pd
from strsimpy.levenshtein import Levenshtein
from strsimpy.jaro_winkler import JaroWinkler
from tqdm import tqdm

from sf_permits.config import logger


def get_matching_strings(
    base: pd.Series,
    target: pd.Series,
    similarity,
    block_length: int = 0,
) -> tuple[dict[int, list], dict[str, list]]:
    logger.info("Starting string matching")

    matching_indices: dict[int, list] = defaultdict(list)
    matching_values: dict[str, list] = defaultdict(list)  # Only for visualisation

    for base_index, base_value in tqdm(base.items(), total=len(base), desc="String matching"):
        logger.debug("Processing base value {}", base_value)
        block_key = base_value[:block_length]
        block = target[target.str.startswith(block_key)]
        logger.debug("{} candidates in block '{}'", len(block), block_key)
        for i, value in block.items():
            if value == base_value:
                continue
            if similarity(base_value, value):
                matching_indices[base_index].append(i)
                matching_values[base_value].append(value)
        number_matches = len(matching_indices[base_value])
        if number_matches == 0:
            logger.debug("No matches found for '{}'", base_value)
        else:
            logger.debug(
                "{} matches in for '{}'", len(matching_indices[base_value]), base_value
            )

    matching_indices = {key: value for key, value in matching_indices.items() if value}
    matching_values = {key: value for key, value in matching_values.items() if value}

    logger.success("String matching complete")

    return matching_indices, matching_values


def jaccard(base: str, target: str, normalise: bool = True) -> float:
    base_words = base.split()
    target_words = target.split()
    intersection = sum(1 for target_word in target_words if target_word in base_words)
    result = intersection
    if normalise:
        union = len(set((*base_words, *target_words)))
        result /= union
    return result


def levenshtein(base: str, target: str) -> float:
    distance = Levenshtein().distance(base, target)
    combined_length = len(base) + len(target)
    return 1 - distance / combined_length

def jaro_winkler(base: str, target: str) -> float:
    return JaroWinkler().similarity(base, target)
