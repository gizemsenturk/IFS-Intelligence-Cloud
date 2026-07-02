from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "size_score",
    "multi_site",
    "legacy_dependency",
    "erp_hiring_signal"
]


def train_model(df):

    X = df[FEATURES]
    y = df["ifs_fit_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nMODEL PERFORMANCE")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 :", r2_score(y_test, preds))

    return model
