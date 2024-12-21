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
    |               |       |
    |---------------|-------|
    | Project ID    | 67    |
    | Dataset ID    | 6     |
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
employed to assess the quality of the permits dataset and to direct
points for improvement;
these points are addressed by data cleaning techniques described
in Section \ref{data-cleaning}.
Finally, Section \ref{data-analysis} describes the selected machine
learning problem under investigation, as well as further data preparation
steps specifically required by the employed model and the results of the
analysis.

# Data Profiling and Quality Assessment {#profiling}

# Data Cleaning

## Data Normalisation
Two string attributes, `Description` and `Existing Use`, provide
semi-structured natural language values which were processed in search
of recurring patterns for encoding these attributes as categorical
when possible, and so required special handling.
No other attributes presented similar challenges, so minor normalisation
was applied to them.

### `Description` Attribute

### `Existing Use` Attribute

### Miscellaneous Data Normalisation
Only minor data normalisation was required for the remaining attributes.
Namely,
attributes encoding dates whose data type was inadequately
inferred by Pandas were encoded as `datetime` objects;
the attribute name on the neighbourhood associated with a permit was
renamed for easier handling;
and latitude and longitude added as separate floating-point attributes
by splitting the permit location.

Importantly, all string values were set to lowercase with the aim of
simplifying comparisons between tuples in the permits dataset and with
tuples in external datasets employed for missing value imputation and
error correction. In these specific cases, further normalisation was
performed to facilitate string matching (e.g. punctuation removal), but
it was deemed that applying further normalisation to the entire dataset
would excessively modify the underlying data.

Furthermore, placeholders for missing values found during preliminary analysis (e.g. "unknown" and "situs to be assigned" for `Street Name`)
were set to
missing^[This was done by assigning `pd.NA` to these values.
\label{foot:na}].

## Error Correction
Various errors were discovered in the permits dataset during manual data
exploration and analysis of the profiling metrics computed as
described in Section \ref{profiling}.

### Location-based Error Correction
The presence of geospatial attributes in the permits dataset provides an
opportunity to match ground-truth external dataset tuples with those
from the permits dataset using their coordinates as matching key.
Matching may be performed not only by exact value matching, but also
by geospatial matching, considering a maximum distance between matching
tuples, or other geometric relationships (e.g. if the geometry of a
tuple is within the geometry of another).

Error correction was performed assuming location information is always
accurate, that is, inconsistencies between geospatial ground-truth
external data and some attribute would imply an error in the attribute
and not in the location.

#### Neighbourhood correction
An external
dataset^[<https://data.sfgov.org/-/Analysis-Neighborhoods/j2bu-swwd/>]
made available by the San Francisco Office of the City Administrator
provides non-overlapping geospatial boundaries associated with each
neighbourhood in the city of San Francisco. The `Neighborhood` value of
each tuple was therefore replaced with their associated coordinate
values.

#### ZIP code correction
Analogously, a second external
dataset^[<https://data.sfgov.org/dataset/Bay-Area-ZIP-Codes/4kz9-76pb/>]
made available by the San Francisco Office of the City Administrator
provides boundaries for regions associated with the same ZIP code, whose
value was used to replace the ZIP code of each tuple based on its
coordinates.

### String-based Error Correction
String attributes are prone to errors due to typos which need to be
corrected. Given the replacement of neighbourhoods with their
ground-truth equivalents, the `Street Name` attribute required
special treatment.

A ground-truth external
dataset^[<https://data.sfgov.org/Geographic-Locations-and-Boundaries/Street-Names/6d9h-4u5v/>]
made available by San Francisco Public Works
provided full street names and their respective components (name, type,
and post direction) which were matched with tuples in the permits
dataset, whose values were then replaced with those provided by the
ground-truth data.

Matching was performed in two stages.
First, exact matches between the external dataset and the permits dataset
were computed by considering normalised street names, lower-case and with
all punctuation removed. Matches were computed between the permit dataset
`Street Name` attribute and the `FullStreetName`, `StreetName` and the concatenation of the `StreetName` and `Post Direction` attributes in the
external dataset to ensure as many exact matches would be found during
the initial matching stage.
In total, 197743 matches were found, leaving 1157 tuples unmatched.

Second, approximate matches were computed between the external dataset
and the 1157 unmatched tuples from initial matching. To that end, a
hybrid similarity measure was used, combining Jaro-Winkler and a
Jaccard-like similarity in two components. This approach resulted in
992 further matches, leaving 165 unmatched and so unmodified.

Formally, two strings $x, y$ are said to _match_ if $m(x, y)$ evaluates
to a logical true value, with
\begin{subequations}
\begin{align}
m(x, y)     &= m_1(x, y) \lor m_2(x, y) & \\
m_1(x, y)   &= (| x \cap y | \ge 1)     \land (sim_j(x, y) > \alpha)    \label{eq:sim:1} \\
m_2(x, y)   &= (sim_j(x, y) > \beta_1)  \land (sim_j(-x, -y) > \beta_2) \label{eq:sim:2}
\end{align}
\end{subequations}
in which, $x \cap y$ returns the set of words contained in both $x$ and
$y$, $sim_j$ is the Jaro-Winkler similarity measure, and given a
sequence $x = (x_1, \dots, x_m)$ of characters, we denote its reverse
sequence by $-x = (x_m, \dots, x_1)$.

The intuition behind this definition is the following:
$m_1$ matches strings which have at least one word in common, which
allows us to reduce the Jaro-Winkler similarity threshold $\alpha$
without making wrong associations. This component aims to match
strings with significantly different content (e.g. "embarcadero center"
and "the embarcadero") but still share an (ideally) meaningful word
between them^[This idea could be further extended adding TF-IDF weights
to each word, but the resulting matches were deemed sufficiently accurate
to go without this addition.]
$m_2$ matches strings which are similar and have both similar prefixes
and suffixes. Given the similarity bonus provided by Jaro-Winkler
to strings with matching prefixes, analysing only the direct sequence
proved to inaccurately match strings with identical prefixes prefixes
(e.g. "lake merced hill so" and "lake merced hill no"), so an additional
similarity threshold was introduced on the reverse sequence to also
account for the suffix.
Similarity thresholds $\alpha = 0.7$, $\beta_1 = 0.93$ and
$\beta_2 = 0.89$ were empirically tuned to balance coverage and
accuracy.

To generate a bijective map between permit and external dataset tuples,
in case a street name from the permit dataset matched multiple
street names from the external dataset, the most similar one according
to the Jaro-Winkler similarity was selected.

Once tuples were matched, `Street Name` and `Street Type` in the
permits dataset were replaced with `StreetName` and `StreetType` from
the external dataset.

### Miscellaneous Error Correction

#### Non-complete permits with completion date
Only permits whose status is "Complete" should be associated with a
completion date, yet so are permits with different statuses.
These values were deemed errors, and the completion date for permits with
a status different from "Complete" were set to
missing^[See Footnote \ref{foot:na}.].

## Missing Value Imputation

### Location-based Imputation
Intuitively, approximate functional dependencies should hold between
hierarchical location elements: a specific block and lot correspond
to a single street, which corresponds (usually) to a single supervisor
district and to a single neighbourhood. These domain-based relationships
enable us to impute values in tuples iteratively, following the described
hierarchy.

In this case, it was decided that imputation would be performed first by
block and lot, and then by street name. No further elements in the
hierarchy were considered since they were deemed possibly too distant to
provide useful data for imputation.

For both block and lot and for street name, a similar imputation
procedure was performed: `latitude` and `longitude` were imputed by
taking the mean over tuples with the same grouping key (either block and
lot or street name), and `Street Name`, `Street Suffix` and
`Supervisor District`, by taking the mode.
Neither `Neighborhood` nor `Zipcode` were imputed since the imputed
coordinates were used to match them to external datasets as described in
Section \ref{location-based-error-correction}.

## Outlier Detection and Removal

## Duplicate Detection and Removal
Given previously performed error correction, and the use of external
ground-truth datasets in particular, duplicate detection and removal was
performed by exact matching between subsets of the attributes of tuples
in the permits dataset.
Selected attributes included the permit number and those associated with
the location of the building permit, except its coordinates, as
coordinates were found not to always match exactly, but rather be a few
meters apart for duplicated permits.
Therefore, only one tuple for each combination of
`Permit Number`,
`Street Name`,
`Street Number`,
`Street Number Suffix`,
`Unit` and
`Unit Suffix`
was kept in the dataset.

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
