from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def process_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)
    
    return X_train, X_test, y_train, y_test
