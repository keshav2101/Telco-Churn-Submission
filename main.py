import os
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.data_processor import process_data
from src.models import get_ensemble_model, get_baseline_models
from src.evaluation import save_metrics

def run_pipeline():
    # 1. Ensure output folders exist
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    # 2. Load & Clean
    print("Loading data...")
    df = load_data('data/Telco.csv')
    df_cleaned = clean_data(df)
    
    # Save the processed data for the repository
    df_cleaned.to_csv('processed_data.csv', index=False)
    print("Processed data saved to processed_data.csv")

    # 3. Process & Split
    X_train, X_test, y_train, y_test = process_data(df_cleaned)

    # 4. Train & Evaluate Ensemble
    print("Training Ensemble...")
    ensemble = get_ensemble_model()
    ensemble.fit(X_train, y_train)
    
    probs = ensemble.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.54).astype(int) # Using our optimal threshold
    
    save_metrics(y_test, preds, probs, "voting_ensemble")

    # 5. Train & Evaluate Baselines
    baselines = get_baseline_models()
    for name, model in baselines.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        save_metrics(y_test, model.predict(X_test), p, name)

    print("Pipeline Complete! All files generated in results/ folder.")

if __name__ == "__main__":
    run_pipeline()
