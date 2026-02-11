from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def get_ensemble_model():
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=42)
    lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.01, random_state=42, verbosity=-1)
    lr = LogisticRegression(C=0.1, max_iter=1000)
    
    ensemble = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm), ('lr', lr)], voting='soft')
    return ensemble

def get_baseline_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(random_state=42)
    }
