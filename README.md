# 📊 Telco Customer Churn Prediction

## 📌 Project Overview
Customer churn refers to customers discontinuing a service. In this project, we built a tabular ML pipeline using XGBoost and Ensemble methods.

## 🔀 Train / Validation Split Method
- **Strategy:** Stratified Train-Test Split (80/20)
- **Total Samples:** ~7,000

## 🏆 Best Result
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **81.19%** |
| **ROC-AUC** | **0.8460** |

## 📁 Project Structure
- `data/`: Raw dataset
- `src/`: Preprocessing scripts
- `results/figures/`: Confusion Matrix and ROC Curves
- `main.py`: Main execution script

## 📊 Visual Results
### Confusion Matrix
![Confusion Matrix](results/figures/confusion_matrix.png)

### ROC Curve
![ROC Curve](results/figures/roc_curve.png)
