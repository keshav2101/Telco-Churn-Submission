import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(model, X_test, y_test, name, threshold=0.54):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # Save Metrics text
    report = classification_report(y_test, preds)
    with open(f"results/metrics/{name}.txt", "w") as f:
        f.write(report)
    
    # Save Confusion Matrix PNG
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
    plt.title(f'CM: {name}')
    plt.savefig(f'results/figures/{name}_confusion_matrix.png')
    plt.close()
    
    # Save ROC Curve PNG
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], 'k--')
    plt.savefig(f'results/figures/{name}_roc_curve.png')
    plt.close()
    
    return report
