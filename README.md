# FAIDM Diabetes (binary_merge12) - FAST Project

## Target (binary_merge12)
- y=1 if Diabetes_012 == 2 (diabetes)
- y=0 if Diabetes_012 in {0,1} (no diabetes + prediabetes merged)

## Setup
pip install -r requirements.txt

## Data
Put CSV here:
data/raw/CDC Diabetes Dataset.csv

## Run
python -m src.main

## Speed controls (in src/faidm/config.py)
- clustering_sample_n (default 50000)
- run_dbscan (default False)
- enable_grid_search (default False)

