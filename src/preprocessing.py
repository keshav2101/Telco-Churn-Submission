import pandas as pd
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
