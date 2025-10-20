"""optimizer.py"""

import statistics
from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from src.utils.model_optimizer.evaluator import get_int_predictions
from src.utils.model_optimizer.evaluator import get_prediction
from src.utils.model_optimizer.evaluator import get_f1_metric_split
from src.utils.model_optimizer.evaluator import calculate_metrics

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

GLOBAL_RANDOM_STATE = 0


def get_best_params_cv(
        clf_algorithm: Any,
        parameters_search: dict,
        x_train: np.array,
        y_train: np.array,
        num_folds: int = 5,
        ) -> dict:
    """Select the best parameters for a model.
    Using the metrics of cross fold validation of a number
    of fold equal to 5 or chosen by the user

    Args:
        clf_algorithm (Any): ML algorithm
        parameters_search (dict): dicstionary with the list of parameters
        x_train (np.array): predictive variables
        y_train (np.array): objective variable
        num_folds (int, optional): number of folds for CV Grid Search

    Returns:
        dict: Best parameters of the chosen model
    """
    skf = StratifiedKFold(n_splits=num_folds,
                          shuffle=True,
                          random_state=GLOBAL_RANDOM_STATE)
    best_clf = GridSearchCV(estimator=clf_algorithm,
                            param_grid=parameters_search,
                            cv=skf,
                            verbose=False,
                            n_jobs=-1)
    best_clf.fit(x_train, y_train)
    return best_clf.best_params_


def get_train_test_loo(
        x_train: np.array,
        y_train: np.array,
        idx_test: int,
        ) -> tuple[np.array, np.array, np.array, np.array]:
    """Create the train and test datasets for Leave-one-out validation.

    Args:
        x_train (np.array): predictive variables
        y_train (np.array): objective variable
        idx_test (int): index of the test dataset

    Returns:
        tuple[np.array, np.array, np.array, np.array]:
            predictive variables of the train split,
            objective variables of the train split,
            predictive variables of the test split,
            objective variables of the test split,
    """
    idx_train = [j for j in range(len(x_train)) if j != idx_test]
    x_train_split = x_train[idx_train, :]
    y_train_split = y_train[idx_train]
    x_test_split = x_train[[idx_test], :]
    y_test_split = y_train[idx_test]
    return x_train_split, y_train_split, x_test_split, y_test_split


def train_optimal_model(
        x_train: np.array,
        y_train: np.array,
        clf_algorithm: Any,
        params_algorithm: dict = None,
        ) -> Any:
    """Get trained model with the chosen parameters

    Args:
        x_train (np.array): predictive variables
        y_train (np.array): objective variable
        clf_algorithm (Any): ML algorithm
        params_algorithm (dict, optional): Dictionary with chosen parameters.
            Defaults to None.

    Returns:
        Any: Trained model
    """
    if params_algorithm:
        clf = clf_algorithm(**params_algorithm)
    else:
        clf = clf_algorithm()
    clf.fit(x_train, y_train)
    return clf


def training_thr(
        trained_clf: Any,
        x_train: np.array,
        y_train: np.array,
        bool_probabilities: bool = True,
        ) -> float:
    """Get the optimal threshold based on maximal TPR and minimal FPR.

    Args:
        trained_clf (Any): Trained model
        x_train (np.array): predictive variables
        y_train (np.array): objective variable
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.

    Returns:
        float: Optimal threshold
    """
    y_scores = get_prediction(trained_clf, x_train, bool_probabilities)
    fpr, tpr, thresholds = roc_curve(np.array(y_train), y_scores)
    thresholds = np.clip(thresholds, 0, 1)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def get_prediction_loo(
        clf_algorithm: Any,
        params_algorithm: dict,
        x_train: np.array,
        y_train: np.array,
        bool_probabilities: bool = True,
        ) -> tuple[np.array, np.array, int]:
    """Obtain the label prediction with the leave-one-out strategy

    Args:
        clf_algorithm (Any): ML algorithm
        params_algorithm (dict): Model parameters
        x_train (np.array): predictive variables of training dataset
        y_train (np.array): objective variable of training dataset
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.

    Returns:
        tuple[np.array, np.array, int]: _description_
    """
    y_pred_loo = []
    y_pred_scores = []
    for idx_test in range(len(x_train)):
        x_train_split, y_train_split, x_test_split, _ = get_train_test_loo(
            x_train,
            y_train,
            idx_test)
        trained_clf = train_optimal_model(
            x_train_split, y_train_split, clf_algorithm, params_algorithm)
        trained_thr = training_thr(
            trained_clf, x_train_split, y_train_split, bool_probabilities)
        test_scores = get_prediction(
            trained_clf, x_test_split, bool_probabilities)
        y_pred_scores.append(test_scores[0])
        y_pred_loo.append(get_int_predictions(test_scores, trained_thr)[0])
    return y_pred_loo, y_pred_scores, 1


def get_f1_score_loo(
        clf_algorithm: Any,
        params_algorithm: dict,
        x_train: np.array,
        y_train: np.array,
        bool_probabilities: bool = True
        ) -> float:
    """Calculate the F1 score with the Leave-one-out strategy

    Args:
        clf_algorithm (Any): ML algorithm
        params_algorithm (dict): Model parameters
        x_train (np.array): predictive variables of training dataset
        y_train (np.array): objective variable of training dataset
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.

    Returns:
        float: F1 score calculated with the leave-one-out strategy
    """
    y_pred_loo, _, _ = get_prediction_loo(
        clf_algorithm,
        params_algorithm,
        x_train,
        y_train,
        bool_probabilities)
    f1_loo = f1_score(y_train, y_pred_loo)
    return f1_loo


def get_f1_score_cfv(
        clf_algorithm: Any,
        params_algorithm: dict,
        x_train: np.array,
        y_train: np.array,
        bool_probabilities: bool = True,
        num_splits: int = 5,
        ) -> float:
    """Calculate the F1 score with the Cross Fold Validation strategy

    Args:
        clf_algorithm (Any): ML algorithm
        params_algorithm (dict): Model parameters
        x_train (np.array): predictive variables of training dataset
        y_train (np.array): objective variable of training dataset
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.
        num_splits (int, optional): number of splits for cross fold validation.
            Defaults to 5.

    Returns:
        float: F1 score calculated with the Cross Fold Validation strategy
    """
    f1_folds = []
    folds = KFold(n_splits=num_splits)
    for train, test in folds.split(x_train):
        x_train_split = x_train[train]
        x_test_split = x_train[test]
        y_train_split = y_train[train]
        y_test_split = y_train[test]
        trained_clf = train_optimal_model(
            x_train_split, y_train_split, clf_algorithm, params_algorithm)
        trained_thr = training_thr(
            trained_clf, x_train_split, y_train_split, bool_probabilities)
        test_scores = get_prediction(
            trained_clf, x_test_split, bool_probabilities)
        test_pred = get_int_predictions(test_scores, trained_thr)
        f1_folds.append(f1_score(y_test_split, test_pred))
    return sum(f1_folds)/len(f1_folds)


def get_f1_score_oss(
        clf_algorithm: Any,
        params_algorithm: dict,
        x_train: np.array,
        y_train: np.array,
        bool_probabilities: bool = True,
        test_fraction: float = 0.2,
        ) -> float:
    """Calculate the F1 score with the one shot split strategy

    Args:
        clf_algorithm (Any): ML algorithm
        params_algorithm (dict): Model parameters
        x_train (np.array): predictive variables of training dataset
        y_train (np.array): objective variable of training dataset
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.
        test_fraction (float, optional): proportion of samples for test split.
            Defaults to 0.2.

    Returns:
        float: F1 score calculated with the one shot split strategy
    """
    (x_train_split, x_test_split,
     y_train_split, y_test_split) = train_test_split(
        x_train,
        y_train,
        test_size=test_fraction,
        random_state=GLOBAL_RANDOM_STATE
    )
    trained_clf = train_optimal_model(
        x_train_split, y_train_split, clf_algorithm, params_algorithm)
    trained_thr = training_thr(
        trained_clf, x_train_split, y_train_split, bool_probabilities)
    f1_test = get_f1_metric_split(
        trained_clf,
        trained_thr,
        x_test_split,
        y_test_split,
        bool_probabilities)
    return f1_test


def get_f1_average_optimal_model(
        clf_algorithm: Any,
        params_algorithm: dict,
        x_train: np.array,
        y_train: np.array,
        bool_probabilities: bool = True,
        test_fraction: float = 0.2,
        ) -> float:
    """Calculate the the F1 average of three evaluation method.

    The F1 score is calculated as the average of leave-one-out,
    cross-fold-validation and one-shot-split.

    Args:
        clf_algorithm (Any): ML algorithm
        params_algorithm (dict): Model parameters
        x_train (np.array): predictive variables of training dataset
        y_train (np.array): objective variable of training dataset
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.
        test_fraction (float, optional): proportion of samples for test split.
            Defaults to 0.2.

    Returns:
        float: FAverage F1 score calculated with three evaluation methods:
            cross-fold-validation, one-shot-split and leave-one-out
    """
    f1_folds = get_f1_score_cfv(
        clf_algorithm=clf_algorithm,
        params_algorithm=params_algorithm,
        x_train=x_train,
        y_train=y_train,
        bool_probabilities=bool_probabilities)
    avrg_f1_folds = sum(f1_folds)/len(f1_folds)
    f1_oss = get_f1_score_oss(
        clf_algorithm=clf_algorithm,
        params_algorithm=params_algorithm,
        x_train=x_train,
        y_train=y_train,
        bool_probabilities=bool_probabilities,
        test_fraction=test_fraction)
    f1_loo = get_f1_score_loo(
        clf_algorithm=clf_algorithm,
        params_algorithm=params_algorithm,
        x_train=x_train,
        y_train=y_train,
        bool_probabilities=bool_probabilities)
    return (avrg_f1_folds+f1_oss+f1_loo)/3


def get_model_opotimal(
        clf_algorithm: Any,
        params_algorithm: dict,
        x_train: np.array,
        y_train: np.array,
        bool_probabilities: bool
        ) -> tuple[Any, float]:
    """Obtain the trained model with hte chosen parameters
    and the optimal threshold

    Args:
        clf_algorithm (Any): ML algorithm
        params_algorithm (dict): Model parameters
        x_train (np.array): predictive variables of training dataset
        y_train (np.array): objective variable of training dataset
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.

    Returns:
        tuple[Any, float]: Trained model and optimal threshold for them
    """
    trained_clf = train_optimal_model(
        x_train, y_train, clf_algorithm, params_algorithm)
    trained_thr = training_thr(
        trained_clf, x_train, y_train, bool_probabilities)
    return trained_clf, trained_thr


def get_f1_data(
        trained_clf: Any,
        trained_thr: float,
        x_data: np.array,
        y_data: np.array,
        bool_probabilities: bool = True,
        ) -> float:
    """Calcualte the F1 score from a trained model in some data

    Args:
        trained_clf (Any): trained ML algorithm
        trained_thr (float): threshold to determine 1 classification
        x_data (np.array): predictive variables
        y_data (np.array): real labels
        bool_probabilities (bool, optional): Calculate Probabilities.
            Defaults to True.

    Returns:
        float: F1 score obtained with the prediction of the input data
    """
    y_scores = get_prediction(trained_clf, x_data, bool_probabilities)
    y_pred = get_int_predictions(y_scores, trained_thr)
    return f1_score(y_data, y_pred)


def get_ic_list(
        list_i: list,
        ) -> tuple[float, float]:
    """Obtain the mean and the standard deviation from a list of numbers

    Args:
        list_i (list): list of numbers

    Returns:
        tuple[float, float]:
            mean and the standard deviation of the list of numbers
    """
    return (round(statistics.mean(list_i), 2),
            round(statistics.stdev(list_i), 2))


def get_metrics_splits_ic(
        y_real: np.array,
        y_predicted: np.array,
        num_splits: int = 5,
        ) -> dict:
    """Obtain the metrics of the leave one out validation as IC.

    IC stands from Interval of Confidence


    Args:
        y_real (np.array): real label values
        y_predicted (np.array): predicted labels
        num_splits (Optional, int): number of splits used. Default 5.

    Returns:
        dict: Mean and Standard deviation of all the metrics
            of the leave one out validation
    """
    metrics_all = {
        "Accuracy": [],
        "Recall": [],
        "F1_Score": [],
        "AUC": [],
        "Specificity": [],
        "Precision": [],
        "PPV": [],
        "NPV": [],
        "FDR": [],
    }
    metrics_stats = {}
    len_splits = int(len(y_real)/num_splits)
    i = 0
    while i < len(y_real):
        y_real_split = np.array(y_real[i:i+len_splits])
        y_pred_split = np.array(y_predicted[i:i+len_splits])
        metrics_split = calculate_metrics(y_real_split, y_pred_split, 1)
        for metric, value in metrics_split.items():
            metrics_all[metric].append(value)
        i += 1
        if i + len_splits >= len(y_real):
            i = len(y_real)
    for metric, value in metrics_all.items():
        mean_metric, std_metric = get_ic_list(metrics_all[metric])
        metrics_stats[metric] = [round(mean_metric), std_metric]
    return metrics_stats


class CellScaler(BaseEstimator, TransformerMixin):
    """Custom scaler that applies MinMax scaling per sample (row-wise)."""

    def __init__(self):
        self.scalers = {}

    def fit(self, X, y=None):
        """Fit method (no fitting required for this transformer)."""
        return self

    def transform(self, X):
        """Transform data by applying row-wise MinMax scaling."""
        X_normalized = []
        for index, row in X.iterrows():
            row_data = row.values.reshape(-1, 1)
            scaler = MinMaxScaler()
            normalized_row_data = scaler.fit_transform(row_data)
            normalized_df = pd.DataFrame(
                normalized_row_data.reshape(1, -1),
                columns=row.index,
                index=[index]
            )
            X_normalized.append(normalized_df)
        return pd.concat(X_normalized)

    def fit_transform(self, X, y=None):
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)
