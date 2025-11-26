import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    
    df = df.copy()

    # Drop rows with all missing values
    df.dropna(how="all", inplace=True)

    # Label Encoding for categorical columns
    cat_cols = ["carrier", "carrier_name", "airport", "airport_name"]
    encoder = LabelEncoder()

    for col in cat_cols:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def create_delay_label(df):

    df["delayed"] = df["arr_del15"].apply(lambda x: 1 if x >= 1 else 0)
    return df
