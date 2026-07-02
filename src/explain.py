import shap

FEATURES = [
    "size_score",
    "multi_site",
    "legacy_dependency",
    "erp_hiring_signal"
]

def explain(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer