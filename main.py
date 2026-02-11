from src.data_loader import load_data
from src.preprocessing import clean_data
from src.data_processor import process_data
from src.models import get_ensemble_model, get_baseline_models
from src.evaluation import evaluate_model
import os

# 1. Load & Preprocess
df = load_data('data/Telco.csv')
df_clean = clean_data(df)
X_train, X_test, y_train, y_test = process_data(df_clean)

# 2. Train Ensemble
ensemble = get_ensemble_model()
ensemble.fit(X_train, y_train)

# 3. Evaluate Ensemble
print("Evaluating Ensemble Model...")
evaluate_model(ensemble, X_test, y_test, "voting_ensemble", threshold=0.54)

# 4. Evaluate Baselines
baselines = get_baseline_models()
for name, model in baselines.items():
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, name)

print("Pipeline complete. Check results folder.")
