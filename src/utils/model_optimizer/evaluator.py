"""evaluator.py"""
from typing import Any
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


def get_int_predictions(
        y_scores: np.array,
        threshold: float,
        ) -> np.array:
    """Get the binary predictions from probabilities using optimal threshold.

    Args:
        y_scores (np.array): label probabilities
        threshold (float): optimal threshold

    Returns:
        np.array: Binary predictions from probabilities using optimal score.
    """
    return (y_scores >= threshold).astype(int)


def print_confusion_matrix(
        y_real: np.array,
        y_scores: np.array,
        threshold: float,
        ) -> None:
    """Print confusion matrix components."""
    y_pred = get_int_predictions(y_scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")


def get_prediction(
        trained_clf: Any,
        x_data: np.array,
        bool_probabilities: bool = True,
        ) -> np.array:
    """Get predicted labels from a trained model.

    Args:
        trained_clf (Any): Trained ML model
        x_data (np.array): input predictive data
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.

    Returns:
        np.array: Predicted labels calculated with trained model.
    """
    if bool_probabilities:
        y_scores = trained_clf.predict_proba(x_data)[:, 1]
    else:
        y_scores = trained_clf.predict(x_data)
    return y_scores


def get_f1_metric_split(
        trained_clf: Any,
        trained_thr: float,
        x_data: np.array,
        y_data: np.array,
        bool_probabilities: bool = True,
        ) -> float:
    """Calculate F1 score from predictions of specific data.

    Args:
        trained_clf (Any): Trained ML model
        trained_thr (float): optimal threshold
        x_data (np.array): input predictive variables
        y_data (np.array): real label
        bool_probabilities (optional, True): Calculate Probabilities.
            Defaults to True.

    Returns:
        float: F1 score from prediction
    """
    y_scores = get_prediction(trained_clf, x_data, bool_probabilities)
    y_prediction = get_int_predictions(y_scores, trained_thr)
    return f1_score(y_data, y_prediction)


def create_true_pred_auc(
        y_true: list,
        y_pred: list,
        ) -> tuple[np.array, np.array]:
    """
    Obtain real and predicted values of labels
    with representation of all the classes.

    Args:
        y_true (list): real labels
        y_pred (list): predicted labels

    Returns:
        tuple[np.array, np.array]: tuple with real and predited
        labels with representation of all the classes
    """
    if y_true.count(0) == 0:
        y_true.append(0)
        y_pred.append(0)
    elif y_true.count(1) == 0:
        y_true.append(1)
        y_pred.append(1)
    return np.array(y_true), np.array(y_pred)


def confusion_matrix_own(
        y_true: np.array,
        y_predict: np.array,
        ) -> tuple[int, int, int, int]:
    """
    Get confusion matrix.

    Args:
        y_true (np.array): real labels
        y_pred (np.array): predicted labels

    Returns:
        tuple[int, int, int, int]: tuple with counts of
        true positives, true negatives, false positives and true negatives
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i, val in enumerate(y_true):
        if val == 1 and y_predict[i] == 1:
            tp += 1
        elif val == 1 and y_predict[i] == 0:
            fp += 1
        elif val == 0 and y_predict[i] == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def calculate_metrics(
        y_true: np.array,
        y_proba: np.array,
        threshold: float
        ) -> dict:
    """Obtain the evaluation metrics from real labels and predicted labels.

    Args:
        y_true (np.array): real labels
        y_proba (np.array): predicted probabilities
        threshold (float): trained threshold

    Returns:
        dict: Evaluation metrics
    """
    # Apply the threshold to probabilities to obtain binary predictions
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    y_true_auc, y_pred_auc = create_true_pred_auc(list(y_true), list(y_pred))
    auc = roc_auc_score(y_true_auc, y_pred_auc)

    # Calculate additional metrics using the confusion matrix
    tp, fp, tn, fn = confusion_matrix_own(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1
    # Positive Predictive Value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 1
    # Negative Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 1
    # False Discovery Rate
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 1

    # Create a dictionary with all metrics
    metrics = {
        "Accuracy": round(100*accuracy),
        "Recall": round(100*recall),
        "F1_Score": round(100*f1),
        "AUC": round(100*auc),
        "Specificity": round(100*specificity),
        "Precision": round(100*precision),
        "PPV": round(100*ppv),
        "NPV": round(100*npv),
        "FDR": round(100*fdr)
    }
    return metrics
