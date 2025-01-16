# sf-permits

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Politecnico di Milano Data and Information Quality 2024-2025 project

**Group members**
| Name                  | Person Code |
|-----------------------|-------------|
Federica Maria Laudizi  | 10724111    |
Satvik Bisht            | 10886954    |
Tomaz Maia Suller       | 10987566    |

**Project information**
|            |    |
|------------|----|
| Project ID | 67 |
| Dataset ID | 6  |

## Usage

### Dependencies
This project uses [uv](https://github.com/astral-sh/uv) for managing dependencies and the Python virtual environment, so it must be
installed.
It also uses Python 3.12.

### Data loading
The raw dataset file `building_permits.csv` available in [Kaggle](https://www.kaggle.com/datasets/aparnashastry/building-permit-applications-data) must be placed in `data/raw`.
External data must be downloaded from specified sources, moved to `data/external`
and renamed according to the following table:

| Name | Source | Format |
|------|--------|--------|
| `analysis-neighborhoods/` | https://data.sfgov.org/-/Analysis-Neighborhoods/j2bu-swwd/about_data                          | Shapefile |
| `bay-area-zip-codes/`     | https://data.sfgov.org/dataset/Bay-Area-ZIP-Codes/4kz9-76pb/about_data                        | Shapefile |
| `street-names.csv`        | https://data.sfgov.org/Geographic-Locations-and-Boundaries/Street-Names/6d9h-4u5v/about_data  | CSV       |

### Pipeline execution
A list of available commands is provided by `make help`.

In particular:
* `make data-cleaning` execute data profiling and outputs results to `data/profiling`;
* `make data-cleaning` execute data cleaning and outputs results to `data/clean/dataset.parquet`;
* `make requirements` creates a virtual environment and installs Python dependencies; it is automatically executed by the
    previous commands and so should not need to be manually executed.

## Project Organization

```
├── Makefile                    <- Makefile with convenience commands: `make data-profiling` or `make data-cleaning`.
├── README.md                   <- The top-level README for developers using this project.
├── data
│   ├── clean                   <- Final, canonical clean data for modelling
│   ├── external                <- Third-party data.
│   ├── interim                 <- Intermediate data that has been transformed (only used for development).
│   ├── profiling               <- Profiling results.
│   └── raw                     <- The original, immutable data dump.
│
├── notebooks                   <- Jupyter notebooks for profiling analysis and modelling.
│
├── pyproject.toml              <- Project configuration file with package metadata for
│                                  sf_permits and configuration for tools like ruff
│
├── references
|   └── data-dictionary.xlsx    <- Data dictionary.
│
├── uv.lock                     <- The requirements file for reproducing the analysis environment.
│
└── sf_permits                  <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sf_permits a Python module
    │
    ├── cleaning.py             <- Data cleaning.
    │
    ├── config.py               <- Store useful variables and configuration.
    │
    ├── profiling.py            <- Raw data profiling.
    │
    └── utils
        ├── __init__.py
        └── string_similarity.py
```

--------

