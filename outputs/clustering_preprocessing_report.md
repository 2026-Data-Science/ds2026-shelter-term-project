# Clustering Preprocessing Verification Report

Generated at (UTC): 2026-05-21T12:29:26+00:00
Input file: `data/train.csv`
Entrypoint: `main_clustering.py`

## Input shape

- X.shape = (26729, 7)
- OutcomeType / OutcomeSubtype: **not used** (unsupervised pipeline)

## Cardinality flow (primary_breed)

- Raw `Breed` unique values: **1380**
- After rule-based simplification (`primary_breed`): **220**
- After Top-K encoding (`top_k=30`): **30** kept + `Other`

## Pipeline output

- Transformed shape: (26729, 109)
- Dtype: float64
- Numeric columns: 8
- OneHot expanded columns: 101
- Total columns: 109

## Categorical columns sent to OneHot

- `animal_type`
- `outcome_season`
- `sex`
- `neuter_status`
- `primary_breed`
- `primary_color`

> `primary_breed` is collapsed to Top-K + Other before OneHot (no `breed_te_*` target encoding).

## Leakage guard (feature names)

- No feature name contains `Outcome` or `breed_te` (checked substrings: ('Outcome', 'breed_te')).

## Transformed sample (first 5 rows, first 10 columns)

| has_name | age_days | outcome_year | outcome_month | outcome_dayofweek | outcome_hour | outcome_is_weekend | is_mixed_breed | animal_type_Cat | animal_type_Dog |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0000 | 0.0000 | -0.8333 | -0.2500 | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 0.0000 | 0.0000 | -1.0000 | 0.5000 | 0.7500 | -0.6000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0.0000 | 0.3527 | 1.0000 | -1.0000 | 0.5000 | -0.6000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| -1.0000 | -0.3324 | 0.0000 | 0.0000 | 0.2500 | 0.8000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| -1.0000 | 0.3527 | -1.0000 | 0.6667 | 0.2500 | -0.6000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
