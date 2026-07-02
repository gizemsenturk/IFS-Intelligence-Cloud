import os
import sys
import joblib

sys.path.append(os.path.abspath("src"))

from data_generator import ManufacturingDataGenerator
from features import create_features
from train import train_model
from predict import predict


MODEL_PATH = "models/regression_model.pkl"
OUTPUT_PATH = "outputs/scored_leads.csv"


def main():

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("Generating data...")
    generator = ManufacturingDataGenerator(n_samples=2000, seed=42)
    df = generator.generate()

    print("Creating features...")
    df = create_features(df)

    print("Training model...")
    model = train_model(df)

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print("Generating predictions...")
    df = predict(df, model)

    print(f"Saving outputs to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n✅ Pipeline completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")