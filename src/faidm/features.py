from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def get_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric = ["BMI", "MentHlth", "PhysHlth"]
    ordinal = ["GenHlth", "Age", "Education", "Income"]
    binary = [c for c in df.columns if c not in (["Diabetes_012"] + numeric + ordinal)]
    return numeric, ordinal, binary

def build_preprocessor(numeric, ordinal, binary) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    ord_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    bin_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("ord", ord_pipe, ordinal),
            ("bin", bin_pipe, binary),
        ],
        remainder="drop",
    )
