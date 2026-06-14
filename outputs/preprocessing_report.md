# Preprocessing Verification Report

Generated at (UTC): 2026-05-30T10:01:50+00:00
Input file: `data/train.csv`

## Input shape

- X.shape = (26064, 7)
- y.shape = (26064,)
- Target classes = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']

## Class proportions

| Class | Count | Ratio |
|-------|-------|-------|
| Adoption | 10381 | 39.83% |
| Died | 189 | 0.73% |
| Euthanasia | 1529 | 5.87% |
| Return_to_owner | 4749 | 18.22% |
| Transfer | 9216 | 35.36% |

## Cardinality flow (primary_breed)

- Raw `Breed` unique values: **1371**
- After rule-based simplification (`primary_breed`): **219**
- After target encoding: **5 numeric columns** (`breed_te_*`)

## Pipeline output

- Transformed shape: (26064, 83)
- Dtype: float64
- Numeric columns (including target encoded): 13
- OneHot expanded columns: 70
- Total columns: 83

## Categorical columns sent to OneHot

- `animal_type`
- `intake_season`
- `sex`
- `neuter_status`
- `primary_color`

> `primary_breed` is intentionally absent — it is replaced by the five `breed_te_*` columns.

## Transformed sample (first 5 rows, first 10 columns)

| has_name | age_days | intake_year | intake_month | intake_dayofweek | intake_hour | intake_is_weekend | is_mixed_breed | breed_te_Adoption | breed_te_Died |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.5610 | -0.3857 | 2.2069 | -1.4136 | 1.5309 | 0.4819 | 1.5851 | 0.2355 | -0.5372 | 1.0015 |
| 0.6406 | 0.2886 | 2.2069 | -1.4136 | 1.5309 | -0.4968 | 1.5851 | -4.2456 | -0.3312 | -0.3404 |
| 0.6406 | -0.3857 | 2.2069 | -1.4136 | 1.5309 | -0.4968 | 1.5851 | 0.2355 | 0.2370 | -0.5128 |
| 0.6406 | 1.9745 | 2.2069 | -1.4136 | 1.0292 | 1.4606 | 1.5851 | 0.2355 | 0.7498 | -1.2424 |
| 0.6406 | -0.0485 | 2.2069 | -1.4136 | 1.0292 | 1.1344 | 1.5851 | 0.2355 | 1.2485 | -0.6062 |
