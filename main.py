import pandas as pd
from src.preprocessing import clean_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/Telco.csv')
df_clean = clean_data(df)

X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, preds)}")
