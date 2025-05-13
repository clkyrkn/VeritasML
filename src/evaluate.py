from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred))

    acc = accuracy_score(y_val, y_pred)
    acc_percent = acc * 100
    print(f"Accuracy: {acc_percent:.2f}%")

    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.title(f"Confusion Matrix\nAccuracy: {acc_percent:.2f}%", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()