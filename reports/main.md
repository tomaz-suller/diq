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

### Voluntary Soft-Story Retrofit and TIDF Compliance 
- **Observation:** Columns had 99.98% missing values.
- **Action:**
  - Dropped these columns as they where not adding significant information due to the high number of null values.

###  Permit Number
- **Observation:** Identified multiple records for the same `Permit Number`.
- Observed that the samples having the same permit number differed in location-related fields (e.g., street, block, lot), indicating permits related to different buildings in the same condominium.
- **Action:**
   - No changes made; differences deemed valid.

### Binary Features (e.g., `Fire Only Permit`, `Structural Notification`)
- **Observation:** Binary features had values "Y" and `NaN`.
- **Action:** Transformed values:
  - "Y" → `True`
  - `NaN` → `False`

### Completed Date
- **Observation:** For this feature we did expect some of the null values due to the fact that if the `current status` of the permit is not "complete" the permit cannot have a "complete date". Therefore, after analysing the data we found that all entries with "complete" status have a `Completed Date`, whereas some of them with status different form complete had a non-null `Completed Date`.
- **Action:**
  - Set `Completed Date` to `NaN` for non-complete statuses with incorrect values.

### Permit Expiration Date
- **Observation:** After looking at the data description we discovered that `Expiration date` is related only to permits that have been issued. Therefore, all permits that are in a status previous to the "issuing" stage must not have an expiration date. Here's a typical sequential order based on how permits are commonly processed:
1. Filed: The permit application has been submitted to the relevant authority.
2. Incomplete: The application is reviewed, and if information is missing, the status is marked as incomplete.
3. Plancheck: The application is under review to ensure compliance with building codes and regulations.
4. Disapproved: The application does not meet the requirements and needs revision.
5. Approved: The application has been reviewed and meets all necessary standards.
6. Appeal: If there’s a dispute over the decision (e.g., disapproval), the applicant can appeal.
7. Issued: The permit has been granted, and construction can commence.
8. Reinstated: If a previously suspended or withdrawn permit is reactivated.
9. Suspend: The permit's activity is temporarily halted due to issues like non-compliance or pending information.
10. Revoked: The permit is permanently invalidated due to significant violations or changes in conditions.
11. Withdrawn: The applicant has chosen to withdraw the permit application.
12. Cancelled: The permit is nullified, typically by the issuing authority or applicant agreement.
13. Complete: All work and inspections are finished, and the project is officially closed.
14. Expired: The permit has exceeded its valid period without necessary action or completion.

- **Action:** Upon reviewing the data, no inconsistencies were identified. While some values were missing, we determined that imputing them would likely produce results significantly different to their true values, given the limited domain knowledge and the variability in expiration dates across permits.

### Construction Type (Existing & Proposed)
- **Observation:** The correlation between `Existing Construction Type` and `Proposed Construction Type` was found to be 1. When both fields are non-null, their values are always identical. If the Proposed Construction Type is null, it indicates that Existing Construction Type already holds a value, making it unnecessary to specify the proposed one. Conversely, when Existing Construction Type is null, the Proposed Construction Type provides the relevant information.
- **Action:** No further action needed as missing values aligned with expected behavior.

### Existing and Proposed Stories
- **Observation:** There is a perfect correlation (1.0) between `Number of Existing Stories` and `Number of Proposed Stories` and this comes with all the implications outlined previously. Moreover, this feature is not relevant for all permit types, and the presence of missing values is valid for specific cases, such as OTC Alterations.
- **Action:**
  - We found that for permit types 1,2,5,7 this feature is not applicable so the null values are legit
  - For permit type 3/4/6 missing values are a very very small number (< 4%) so we can proceed deleting this rows as they might be outliers. 
  - For permit type 8 (OTC Alterations), there is a relatively high percentage of missing values (approximately 22%). These permits are typically issued for minor modifications or repairs to existing buildings, which usually do not involve adding new stories or altering the number of existing ones but in some cases we can have new stories. As a result, the missing values are valid and should not be imputed.

### Existing and Proposed Use
- **Observation:** Rows where both `Proposed Use` and `Existing Use` were null were checked for permit type relevance. We observed that 99% of missing values were form permit type 8. We assuemd that this is indicative of something meaningful about the permit type itself.Perhaps the 'Proposed Use' and 'Existing Use' fields are not as relevant or required for this specific permit type. Therefore, we decided to not impute them.
- **Action:**
  - According to the data, permit types 1, 2, 4, 5, and 6 are expected to include this attribute. However, a very small number of records for these permit types doesn't contain this attribute. These records are considered outliers and have been dropped.
  
### Estimated Cost
- **Observation:** Permit type 6 should not have `Estimated Cost` but had 597 rows with values.
- **Action:** Dropped these rows as outliers.

### Neighborhoods - Analysis Boundaries
- **Observation:** 0.87% of values were missing. According to the data description, this column provides neighborhood to which the building location belongs to so this information has a direct link to the zip code.
- **Action:**
  - Imputed missing values using the most frequent `Neighborhoods - Analysis Boundaries` value for the respective zip codes.
  - We used the zip code to fill in the missing data. There are 10 rows that have missing `Neighborhoods - Analysis Boundaries` but not null `Zipcode`. We filled in the missing data with the most frequent `Neighborhoods - Analysis Boundaries` value of the respective Zipcode. Because the amount of missing data is not substantial, drop the remaining missing rows.

### Supervisor District
- **Observation:** 0.86% of values were missing. This column provides information about the `Supervisor District` to which the building location belongs to. This information can be based on the column `Neighborhoods - Analysis Boundaries` to fill in the missing values. However, after cleaning column `Neighborhoods - Analysis Boundaries`, there are only 3 rows that have null `Supervisor District` also have not null `Neighborhoods - Analysis Boundaries`.
- **Action:**
  - Imputed missing values using known relationships with `Neighborhoods - Analysis Boundaries` and `Zipcode`.
  - Dropped remaining rows with missing values.

### Correlation Analysis
1. **Construction Types and Descriptions:**
   - Verified consistency between types and descriptions for both `Existing` and `Proposed Construction Type`.
   - No inconsistencies found.
2. **Numeric Features:**
   - Heatmap showed expected correlations, confirming data integrity.


### Conclusion
The cleaning process ensured the dataset is consistent, reliable, and ready for downstream applications, such as predictive modeling or descriptive analysis of permit-related activities.




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
