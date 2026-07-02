import numpy as np
import pandas as pd


def create_features(df):
    df = df.copy()

    df["size_score"] = np.log1p(df["company_size"]) / np.log1p(5000)

    return df