from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix
)
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


def evaluate_model(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

# https://www.scribbr.co.uk/stats/confidence-interval-meaning/
# The Z value for 95% confidence is Z=1.96


def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = math.sqrt(
        (AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (round(lower, 2), round(upper, 2))


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, title="Confusion Matrix", save_path=None):

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=labels, yticklabels=labels,
                cmap='Blues', cbar=True, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path) if save_path else None
    plt.show()


def plot_roc(y_test, y_prob, title='ROC Curve', save_path=None):

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure()
    plt.plot(
        fpr, tpr, label=f'LR (AUC = {roc_auc:.4f}), with 95% CI: {roc_auc_ci(y_test, y_prob)}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(save_path) if save_path else None
    plt.show()
