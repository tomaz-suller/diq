---
title: Data Cleaning and Analysis over the San Francisco Building Permits Dataset
author:
- Federica
- Satvik
- Tomaz Maia Suller, 10987566
abstract: |
    The abstract goes here in case we decide to add one.
header-includes:
- \usepackage[toc, page]{appendix}
include-before: |
    |                |                        |
    |----------------|------------------------|
    | **Project ID** | 67                     |
    | **Dataset ID** | 6                      |
...

<!--

Instructions

    PROJECT REPORT
    PROJECT ID
    ASSIGNED DATASET
    STUDENTS (NAME SURNAME ID)
    1. SETUP CHOICES
    Describe the setup choices made: libraries, data preparation techniques used, etc.
    2. PIPELINE IMPLEMENTATION
    Describe all the pipeline steps in detail: what did you find from the data exploration? How did you decide to use it
    in the data preparation phase? Why did you used specific that data preparation technique?
    3. RESULTS
    Discuss the main results obtained: verify the desired quality level has been achieved, compare the data analysis
    results [only for 3-people groups]
    Very important Justify your choices! (for example, why you have chosen a specific data preparation technique for a
    specific column than all those seen in the lectures?)
-->

# Introduction

The remainder of this report is organised as follows:
Section \ref{profiling} describes preliminary analysis approaches
employed to assess the quality of the dataset and to direct points for
improvement;
these points are addressed by data cleaning techniques described
in Section \ref{data-cleaning}.
Finally, Section \ref{data-analysis} describes the selected machine
learning problem under investigation, as well as further data preparation
steps specifically required by the employed model.

# Data Profiling and Quality Assessment {#profiling}

# Data Cleaning

## Data Normalisation
As no major formatting issues were found in the dataset, only minor
data normalisation was required.
Namely,
attributes encoding dates whose data type was inadequately
inferred by Pandas were encoded as `datetime` objects;
the attribute name on the neighbourhood associated with a permit was
renamed for easier handling;
and latitude and longitude added as separate floating-point attributes
by splitting the permit location.

Importantly, all string values were set to lowercase with the aim of
simplifying comparisons between tuples in the dataset and with tuples in
external datasets employed for missing value imputation and error
correction. In these specific cases, further normalisation was performed
to facilitate string matching (e.g. punctuation removal), but it was
deemed that applying further normalisation to the entire dataset would
excessively modify the underlying data.

Finally, placeholders for missing values found during preliminary analysis (e.g. "unknown" and "situs to be assigned" for the street name)
were set to missing
^[This was done by assigning `pd.NA` to these values. \label{foot:na}].

## Error Correction
Various errors were discovered in the dataset during manual data
exploration and analysis of the profiling metrics computed as
described in Section \ref{profiling}.

### Location-based Error Correction
The presence of geospatial attributes in the dataset provides an
opportunity to match ground-truth external dataset tuples with those
from the dataset using their coordinates as matching key.
Matching may be performed not only by exact value matching, but also
by geospatial matching, considering a maximum distance between matching
tuples, or other geometric relationships (e.g. if the geometry of a
tuple is within the geometry of another).

### Remaining Error Correction

#### Non-complete permits with completion date
Only permits whose status is "Complete" should be associated with a
completion date, yet so are permits with different statuses.
These values were deemed errors, and the completion date for permits with
a status different from "Complete" were set to missing
^[See Footnote \ref{foot:na}.].

# Data Analysis

<!--
Some pandoc black magic to write markdown inside the LaTeX appendices
environment
-->
```{=tex}
\begin{appendices}
```

# Example Appendix {#app:a}

```{=tex}
\end{appendices}
```
