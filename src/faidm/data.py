import pandas as pd

EXPECTED_COLS = [
    "Diabetes_012","HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump",
    "AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk",
    "Sex","Age","Education","Income"
]

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Diabetes_012"] = df["Diabetes_012"].round().astype("Int64")
    return df

def make_target_binary_merge12(df: pd.DataFrame) -> pd.Series:
    return (df["Diabetes_012"].astype(int) == 2).astype(int)
