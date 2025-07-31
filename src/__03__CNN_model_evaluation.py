from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from src.__00__paths import figures_dir, docs_dir


def save_report_confusion_matrix(y_test, y_pred):
    # Calculate predictions
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 2. Save classification report to file
    report_path = docs_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # 3. Save confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.show()
