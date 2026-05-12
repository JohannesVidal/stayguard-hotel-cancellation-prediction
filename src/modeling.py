import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def evaluate_classification_model(model, X_test, y_test, model_name):
    """
    Generate predictions and return the main classification metrics for the positive class.
    In this project, the positive class is Canceled = 1.
    """
    y_pred = model.predict(X_test)

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    return results, y_pred


def print_classification_summary(y_test, y_pred):
    """
    Print classification report using the project target labels.
    """
    print(classification_report(y_test, y_pred, target_names=["Not_Canceled", "Canceled"]))


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """
    Plot a confusion matrix for binary cancellation classification.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Not_Canceled", "Predicted Canceled"],
        yticklabels=["Actual Not_Canceled", "Actual Canceled"]
    )

    plt.title(title, fontsize=14, weight="bold", pad=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()


def compare_model_results(model_results):
    """
    Convert a list of model result dictionaries into a sorted comparison table.
    """
    model_comparison = pd.DataFrame(model_results).round(4)
    model_comparison = model_comparison.sort_values("f1_score", ascending=False)

    return model_comparison