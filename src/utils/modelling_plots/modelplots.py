""" modelplots.py """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.utils.model_optimizer.evaluator import create_true_pred_auc


def model_evaluation_plots(
        alg_f1_metrics: dict,
        dataset_name: str,
        ) -> None:
    """Plot the algorithm performance evaluation with different methods. 

    Args:
        alg_f1_metrics (dict): F1 score of different methods used:
            leave-one-out, cross-fold-validation, one-shot-split and validation
        dataset_name (str): dataset name
    """
    models = list(alg_f1_metrics.keys())
    f1_loo = [alg_f1_metrics[model][0] for model in models]
    f1_cfv = [alg_f1_metrics[model][1] for model in models]
    f1_oss = [alg_f1_metrics[model][2] for model in models]
    f1_val = [alg_f1_metrics[model][3] for model in models]

    x = np.arange(len(models))  # X positions
    width = 0.1  # Bar withd

    # Plot
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x-0.15, f1_loo, width, label="F1 LOO", color='coral')
    bars2 = plt.bar(x-0.05, f1_cfv, width, label="F1 CFV", color='gold')
    bars3 = plt.bar(x+0.05, f1_oss, width, label="F1 OSS", color='bisque')
    bars4 = plt.bar(x+0.15, f1_val, width, label="F1 VAL", color='palegreen')

    # Lables and legends
    plt.xlabel("Models")
    plt.ylabel("Percentage F1 (%)")
    plt.title(f"Model Comparison - F1-SCORE TEST & VAL - {dataset_name}")
    plt.xticks(x + width / 4, models)
    plt.legend()

    for bar_num in [bars1, bars2, bars3, bars4]:
        for bar in bar_num:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/4,
                     yval+1,
                     f'{yval:.0f}%',
                     ha='center', va='bottom', fontsize="6")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.8)
    plt.show()


def auc_plot_test_val(
        y_test: np.array,
        y_test_scores: np.array,
        y_val: np.array,
        y_val_scores: np.array,
        dataset_name: str,
        auc_test: float,
        auc_val: float) -> None:
    """Plot AUC-ROC for discovery and validation dataset

    Args:
        y_test (np.array): real labels of training dataset
        y_test_scores (np.array): probabilities predicted for discovery dataset
        y_val (np.array): real labels of validation dataset
        y_val_scores (np.array): probabilities predicted for validation dataset
        dataset_name (str): dataset name
        auc_test (float): AUC metric of discovery dataset
        auc_val (float): AUC metric of validation dataset
    """
    train_fpr, train_tpr, _ = roc_curve(y_test, y_test_scores)
    val_fpr, val_tpr, _ = roc_curve(y_val, y_val_scores)
    plt.plot(
        train_fpr,
        train_tpr,
        label=" AUC DISCOVERY = "+str(auc_test)+"%", color="darkgoldenrod"
        )
    plt.plot(val_fpr,
             val_tpr,
             ls='-.',
             label=" AUC VALIDATION = "+str(auc_val)+"%", color="cadetblue")
    plt.legend()
    plt.xlabel("Falsos Positives Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"AREA UNDER THE CURVE - {dataset_name}")
    plt.show()


def get_fpr_tpr_loo(
        y_real: np.array,
        y_predicted: np.array,
        num_splits: int = 5,
        ) -> tuple[list, list]:
    """_summary_

    Args:
        y_labels (np.array): _description_
        y_scores (np.array): _description_
        num_splits (int, optional): _description_. Defaults to 5.

    Returns:
        tuple[list, list]: _description_
    """
    len_splits = int(len(y_real)/num_splits)
    splits_fpr = []
    splits_tpr = []
    i = 0
    while i < len(y_real):
        y_real_split = np.array(y_real[i:i+len_splits])
        y_pred_split = np.array(y_predicted[i:i+len_splits])
        y_true_auc, y_pred_auc = create_true_pred_auc(list(y_real_split),
                                                      list(y_pred_split))
        train_fpr, train_tpr, _ = roc_curve(y_true_auc, y_pred_auc)
        if len(splits_fpr) > 0:
            if len(train_fpr) == len(splits_fpr[-1]):
                splits_fpr.append(train_fpr)
                splits_tpr.append(train_tpr)
        else:
            splits_fpr.append(train_fpr)
            splits_tpr.append(train_tpr)
        i += 1
        if i + len_splits >= len(y_real):
            i = len(y_real)
    splits_fpr = np.array(splits_fpr)
    splits_tpr = np.array(splits_tpr)
    return (np.average(splits_fpr, axis=0),
            np.average(splits_tpr, axis=0))


def auc_plot_loo_val(
        y_test: np.array,
        y_test_scores: np.array,
        y_val: np.array,
        y_val_scores: np.array,
        dataset_name: str,
        auc_test: float,
        auc_val: float,
        num_splits: int = 5,
        ) -> None:
    """Plot AUC-ROC for leave-one-out and  one-shot-split validation dataset

    Args:
        y_test (np.array): real labels of training dataset
        y_test_scores (np.array): probabilities predicted for discovery dataset
        y_val (np.array): real labels of validation dataset
        y_val_scores (np.array): probabilities predicted for validation dataset
        dataset_name (str): dataset name
        auc_test (float): AUC metric of discovery dataset
        auc_val (float): AUC metric of validation dataset
        num_splits (Optional, int): number of splits used. Default 5.
    """
    mean_fpr, mean_tpr = get_fpr_tpr_loo(y_test, y_test_scores, num_splits)
    val_fpr, val_tpr, _ = roc_curve(y_val, y_val_scores)
    plt.plot(
        mean_fpr,
        mean_tpr,
        label=" AUC DISCOVERY = "+str(auc_test)+"%", color="darkgoldenrod"
        )
    plt.plot(val_fpr,
             val_tpr,
             ls='-.',
             label=" AUC VALIDATION = "+str(auc_val)+"%", color="cadetblue")
    plt.legend()
    plt.xlabel("Falsos Positives Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"AREA UNDER THE CURVE - {dataset_name}")
    plt.show()
