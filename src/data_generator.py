import numpy as np
import pandas as pd


class ManufacturingDataGenerator:
    def __init__(self, n_samples=2000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)

    def generate(self):
        df = pd.DataFrame()

        df["company"] = [
            f"Company_{i}" for i in range(self.n_samples)
        ] 

        df["company_size"] = np.random.randint(20, 5000, self.n_samples)
        df["multi_site"] = np.random.binomial(1, 0.35, self.n_samples)
        df["legacy_dependency"] = np.random.uniform(0, 1, self.n_samples)
        df["erp_hiring_signal"] = np.random.uniform(0, 1, self.n_samples)

        df["size_score"] = np.log1p(df["company_size"]) / np.log1p(5000)

        true_score = (
            0.3 * np.sqrt(df["size_score"]) +
            0.2 * df["multi_site"] +
            0.25 * df["legacy_dependency"]**2 +
            0.25 * np.sin(df["erp_hiring_signal"] * 3)
        )

        df["ifs_fit_score"] = (
            true_score * 100 + np.random.normal(0, 7, self.n_samples)
        ).clip(0, 100)

        return df
