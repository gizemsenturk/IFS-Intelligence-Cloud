import pandas as pd

FEATURES = [
    "size_score",
    "multi_site",
    "legacy_dependency",
    "erp_hiring_signal"
]


def predict(df, model):

    df = df.copy()

    X = df[FEATURES]

    preds = model.predict(X)

    df["predicted_score"] = preds

    df["predicted_score"] = df["predicted_score"].clip(0, 100)

    df["segment"] = df["predicted_score"].apply(classify)

    return df


def classify(score):

    if score >= 70:
        return "Hot"
    elif score >= 40:
        return "Warm"
    else:
        return "Cold"