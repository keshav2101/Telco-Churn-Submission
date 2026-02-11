import pandas as pd
import numpy as np

def clean_data(df):
    # TotalCharges cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(df['MonthlyCharges'])
    # Redundancy cleanup
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)
    # Feature Engineering (Interaction Features)
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = (df[services] == 'Yes').sum(axis=1)
    df['Avg_Monthly_Value'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['High_Risk_Group'] = ((df['Contract'] == 'Month-to-month') & (df['InternetService'] == 'Fiber optic')).astype(int)
    # Mapping and Drop
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    return pd.get_dummies(df, drop_first=True)import pandas as pd
import numpy as np

def clean_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(df['MonthlyCharges'])
    df.replace('No internet service', 'No', inplace=True)
    df.replace('No phone service', 'No', inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df_ml = pd.get_dummies(df, drop_first=True)
    return df_ml
