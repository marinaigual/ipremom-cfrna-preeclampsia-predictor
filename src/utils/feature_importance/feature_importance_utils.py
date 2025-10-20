"""feature_importance_utils.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
from scipy.stats import pearsonr

from typing import Any
from typing import Union
from typing import Optional
from typing import List

from scipy.stats import ttest_1samp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
import seaborn as sns


def feature_importance(
        model,
        X,
        y,
        subsample_size=None,
        num_shuffle_sets=10,
        confidence_level=0.99,
        scoring=None
) -> pd.DataFrame:
    """
    Calculates feature importance using permutation, with p-values
    and confidence intervals.

    Parameters:
    - model: Trained model.
    - X: Feature set of the data.
    - y: True labels of the data.
    - subsample_size: Sample size to use.
    - num_shuffle_sets: Number of permutations to calculate standard
      deviation and p-values.
    - confidence_level: Confidence level for confidence intervals.

    Returns:
    - DataFrame with feature importance, p-values, and confidence intervals.
    """
    # Subsample the data if necessary
    if subsample_size:
        print(f"Subsampling to {subsample_size} samples...")
        X = X.sample(n=subsample_size, random_state=42)
        y = y.loc[X.index]
    # Calculate feature importance by permutation
    result = permutation_importance(model, X, y, n_repeats=num_shuffle_sets,
                                    random_state=42, scoring=scoring)
    # Get feature names
    feature_names = (X.columns if isinstance(X, pd.DataFrame) else
                     [f"Feature {i}" for i in range(X.shape[1])])
    # Calculate p-values based on one-sample t-test
    p_values = np.array([
        ttest_1samp(result.importances[i], 0, alternative='greater').pvalue
        for i in range(len(feature_names))
    ])
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_bound = np.percentile(result.importances, 100 * (alpha / 2),
                                axis=1)
    upper_bound = np.percentile(result.importances, 100 * (1 - alpha / 2),
                                axis=1)

    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance Mean': result.importances_mean,
        'Importance StdDev': result.importances_std,
        'p_value': p_values,
        f'{int(confidence_level * 100)}% CI Lower': lower_bound,
        f'{int(confidence_level * 100)}% CI Upper': upper_bound,
        'n_repeats': num_shuffle_sets
    })
    # Sort by mean importance
    return importance_df


def plot_feature_importance(importance_df):
    """
    Creates a bar plot with features on the x-axis
    and mean importance on the y-axis.
    Also plots confidence intervals (99% CI) and shows p-value above
    each bar.

    Parameters:
    - importance_df: DataFrame with columns 'Feature', 'Importance Mean',
    'p-value', '99% CI Lower', '99% CI Upper'.
    """
    # Sort data by mean importance
    importance_df = importance_df.sort_values(by='Importance Mean',
                                              ascending=False)

    # Data for plot
    features = importance_df['Feature']
    importance_mean = importance_df['Importance Mean']
    lower_bound = importance_df['99% CI Lower']
    upper_bound = importance_df['99% CI Upper']
    p_values = importance_df['p_value']

    # Create bar plot
    plt.figure(figsize=(12, 8))
    plt.bar(features, importance_mean,
            yerr=[importance_mean - lower_bound,
                  upper_bound - importance_mean],
            capsize=5, color='skyblue', label='Importance Mean')

    # Add p-values above bars
    for i, (upper, p_val) in enumerate(zip(upper_bound, p_values)):
        plt.text(i, upper, f'p-val: {p_val:.3f}', ha='center', va='bottom',
                 fontsize=7, color='black', rotation=70)

    # Labels and title
    plt.xlabel('Features')
    plt.ylabel('Importance Mean')
    title = 'Feature Importance with 99% Confidence Intervals and P-Values'
    plt.title(title)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()


def load_data_and_predict(file_path, model_path):
    """Load data and model for prediction."""
    # Load dataset
    cm = pd.read_csv(file_path, index_col=0)
    # meta_columns = ["class", "group"]
    train = cm[cm.group == "train"].drop(["group"], axis=1)
    val = cm[cm.group == "validation"].drop(["group"], axis=1)

    X_train = train.drop("class", axis=1)
    y_train = train["class"]
    X_val = val.drop("class", axis=1)
    y_val = val["class"]

    # Load the pre-trained model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model, X_train, y_train, X_val, y_val


def loo_adjusted_probabilities(model: Any,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               thr_final: float) -> np.ndarray:
    """
    Perform Leave-One-Out Cross-Validation (LOO) and adjust probabilities
    based on a final threshold.

    This function applies LOO cross-validation on a given model,
    calculating an optimal threshold for each iteration (using only the
    training split in that iteration). For each hold-out sample,
    the probability is adjusted based on the threshold calculated in the
    training split and a final threshold (`thr_final`). The adjusted
    probabilities are then returned.

    Args:
        model (Any): A trained classifier model with `fit` and
            `predict_proba` methods.
        X_train (np.ndarray): Feature matrix for the training set.
        y_train (np.ndarray): Class labels for the training set.
        thr_final (float): The final threshold used for scaling the
            probabilities.

    Returns:
        np.ndarray: Array of adjusted probabilities for each sample in the
            training set.
    """
    loo = LeaveOneOut()
    adjusted_train_probs = []

    for train_index, holdout_index in loo.split(X_train):
        X_train_split = X_train[train_index]
        X_holdout = X_train[holdout_index]
        y_train_split = y_train[train_index]

        # Train the model on the training split
        model.fit(X_train_split, y_train_split)

        # Get probabilities on the training split and calculate the
        # optimal threshold
        train_probs_split = model.predict_proba(X_train_split)[:, 1]
        best_thr_split = find_optimal_threshold(train_probs_split,
                                                y_train_split)

        # Get the probability of the holdout sample
        init_prob = model.predict_proba(X_holdout)[:, 1][0]

        # Adjust probability based on best_thr_split and thr_final
        adjusted_prob = (init_prob * thr_final) / best_thr_split

        adjusted_train_probs.append(adjusted_prob)

    return np.array(adjusted_train_probs)


def find_optimal_threshold(y_scores: Union[np.ndarray, list],
                           y_test: Union[np.ndarray, list]) -> float:
    """
    Find the optimal threshold for classification based on the maximum
    F1-score.

    Args:
        y_scores (Union[np.ndarray, list]): Predicted probability scores for
            the positive class.
        y_test (Union[np.ndarray, list]): True binary class labels.

    Returns:
        float: Optimal threshold that maximizes the F1-score.
    """
    # Calculate ROC curve metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores)

    # Clip thresholds to ensure they are between 0 and 1
    thresholds = np.clip(thresholds, 0, 1)

    # Calculate F1-score for each threshold and find the one with the
    # highest score
    f1_scores = (2 * (tpr * (1 - fpr)) /
                 (tpr + (1 - fpr) + 1e-6))  # Avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    optimal_thr = thresholds[optimal_idx]

    return optimal_thr


def plot_violin(adjusted_train_probs: np.ndarray,
                y_train: np.ndarray,
                val_probs: np.ndarray,
                y_val: np.ndarray,
                threshold: float,
                title: str,
                color_filled: str,
                color_border: str) -> None:
    """
    Plot violin plots for adjusted probabilities, with separate subplots for
    discovery and validation sets.

    Args:
        adjusted_train_probs (np.ndarray): Adjusted probabilities for the
            training set.
        y_train (np.ndarray): Class labels for the training set.
        val_probs (np.ndarray): Probabilities for the validation set.
        y_val (np.ndarray): Class labels for the validation set.
        threshold (float): Threshold line for probability classification.
        title (str): Title of the plot.
        color_filled (str): Color for filled points in the plot.
        color_border (str): Color for border points in the plot.
    """
    # Map class labels to descriptive strings for plotting
    train_classes = np.where(y_train == 0, "Control", "PE")
    val_classes = np.where(y_val == 0, "Control", "PE")

    # Prepare DataFrames for Discovery and Validation sets
    train_df = pd.DataFrame({
        "Probability": adjusted_train_probs,
        "Class": train_classes,
        "Set": "Discovery"
    })
    val_df = pd.DataFrame({
        "Probability": val_probs,
        "Class": val_classes,
        "Set": "Validation"
    })

    # Set up subplots for Discovery and Validation sets
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    # Discovery Set Plot
    sns.violinplot(x="Class", y="Probability", data=train_df, ax=axes[0],
                   inner=None, alpha=0.3,
                   palette={"Control": "lightblue", "PE": color_filled},
                   order=["PE", "Control"], width=0.3)
    sns.stripplot(x="Class", y="Probability", data=train_df, ax=axes[0],
                  jitter=True, marker="o", size=8, dodge=False,
                  alpha=1,
                  palette={"Control": "lightblue", "PE": color_border},
                  order=["PE", "Control"])
    axes[0].axhline(y=threshold, color='k', linestyle='--',
                    label=f"Threshold: {threshold}")
    axes[0].set_title(f"{title} - Discovery Set")
    axes[0].set_ylim([-0.1, 1.1])

    # Validation Set Plot
    sns.violinplot(x="Class", y="Probability", data=val_df, ax=axes[1],
                   inner=None, alpha=0.3,
                   palette={"Control": "lightblue", "PE": color_filled},
                   order=["PE", "Control"], width=0.3)
    sns.stripplot(x="Class", y="Probability", data=val_df, ax=axes[1],
                  jitter=True, marker="o", size=8, dodge=False,
                  alpha=1,
                  palette={"Control": "lightblue", "PE": color_border},
                  order=["PE", "Control"])
    axes[1].axhline(y=threshold, color='k', linestyle='--',
                    label=f"Threshold: {threshold}")
    axes[1].set_title(f"{title} - Validation Set")
    axes[1].set_ylim([-0.1, 1.1])

    # Set labels and common title
    for ax in axes:
        ax.set_ylabel("Probability")
        ax.set_xlabel("Class")
    plt.suptitle(f"{title} - Violin Plot with Adjusted Probabilities",
                 y=1.02)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def plot_heatmap(data: pd.DataFrame,
                 y_labels: Union[pd.Series, pd.DataFrame],
                 title: str) -> None:
    """
    Create a heatmap with color-coded classes and sorted columns.

    This function plots a heatmap of the input data, with columns sorted by
    class labels and a separation line between the class label bar and the
    main heatmap data. It adds a legend outside the plot to differentiate
    between PE and Control classes.

    Args:
        data (pd.DataFrame): DataFrame containing the data to be visualized.
        y_labels (Union[pd.Series, pd.DataFrame]): Series or DataFrame
            containing class labels for each sample.
        title (str): Title of the plot.
    """
    # Sort columns by class label (PE and Control) to group them
    sorted_indices = y_labels.sort_values().index
    data_sorted = data[sorted_indices]

    # Insert a separation row with NaNs between the class label bar and
    # heatmap data
    separation_row = pd.DataFrame(np.nan, index=["Separation"],
                                  columns=data_sorted.columns)
    data_with_sep = pd.concat([data_sorted.iloc[:1], separation_row,
                               data_sorted.iloc[1:]], axis=0)

    # Plot the heatmap with specified color map and formatting options
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_with_sep, cmap="coolwarm", cbar_kws={'label': 'Value'},
                xticklabels=False, yticklabels=True, linecolor="white",
                linewidths=0.5)

    # Define color palette for PE and Control classes
    unique_labels = ["PE", "Control"]
    color_palette = sns.color_palette("Set1", len(unique_labels))
    color_map = dict(zip(unique_labels, color_palette))

    # Add a custom legend for PE and Control classes outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color_map[label], markersize=10)
               for label in unique_labels]
    plt.legend(handles, unique_labels, title="Class",
               bbox_to_anchor=(-0.25, 1), loc='upper left',
               borderaxespad=0)

    # Add title to the plot
    plt.title(title)

    # Adjust y-axis labels to include the class label row and hide the
    # separation row label
    ax = plt.gca()
    y_labels_combined = (["Class"] + [""] +
                         list(data_with_sep.index[2:]))
    ax.set_yticklabels(y_labels_combined, rotation=0, fontsize=5)

    plt.tight_layout()
    plt.show()


def plot_pca_2d(data: pd.DataFrame,
                color: str,
                title: str = "PCA of Data") -> None:
    """
    Perform PCA on the input data and plot the first two principal
    components with custom colors.

    Args:
        data (pd.DataFrame): DataFrame containing the features and class
            labels. Assumes the class labels are stored in a column named
            'class'.
        color (str): Color to use for the 'PE' class in the plot.
        title (str, optional): Title of the plot. Defaults to "PCA of Data".
    """
    # Separate features and class labels
    features = data.drop(columns="class")
    labels = data["class"]
    print(f"Labels: {labels.unique()}")

    # Perform PCA to reduce dimensionality to 2 components
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(features)

    # Create a DataFrame with the PCA components and class labels
    pca_df = pd.DataFrame(data=pca_components, columns=["PC1", "PC2"])
    pca_df["class"] = labels.values

    # Plot the PCA results with the specified color for the 'PE' class and
    # steelblue for 'Control'
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class",
                    palette={"PE": color, "Control": "steelblue"},
                    s=60, alpha=0.7)

    # Customize plot aesthetics
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class", loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_probabilities(train_probs: Union[np.ndarray, list],
                       y_train: Union[np.ndarray, list],
                       val_probs: Union[np.ndarray, list],
                       y_val: Union[np.ndarray, list],
                       threshold: float,
                       title: str,
                       color_filled: str,
                       color_border: str) -> None:
    """
    Create a probability plot with custom colors for PE classes (EOPE, LOPE)
    and Control.

    This function creates a two-panel plot for the train and validation sets,
    each showing the probability of each sample being a PE or Control class,
    with distinct colors indicating whether each prediction is good or bad.

    Args:
        train_probs (Union[np.ndarray, list]): Probabilities for the training
            set.
        y_train (Union[np.ndarray, list]): Class labels for the training set.
        val_probs (Union[np.ndarray, list]): Probabilities for the validation
            set.
        y_val (Union[np.ndarray, list]): Class labels for the validation set.
        threshold (float): Probability threshold for determining good/bad
            predictions.
        title (str): Title for the plot.
        color_filled (str): Color for correctly predicted PE samples.
        color_border (str): Border color for PE samples.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    def plot_subplot(probs: Union[np.ndarray, list],
                     y_true: Union[np.ndarray, list],
                     ax: plt.Axes,
                     set_name: str) -> None:
        """
        Helper function to plot a probability subplot for a given dataset
        (train or validation).
        """
        sorted_indices = np.argsort(probs)
        sorted_probs = np.array(probs)[sorted_indices]
        sorted_labels = np.array(y_true)[sorted_indices]

        # Plot each sample with custom color and marker based on probability
        # and threshold
        for i, (prob, label) in enumerate(zip(sorted_probs, sorted_labels)):
            if label == 1:  # PE class (EOPE or LOPE)
                color = color_filled if prob >= threshold else 'none'
                edge_color = color_border
                marker = '*'  # Star for PE class
                pe_label = ('PE (good predicted)' if color == color_filled
                            else 'PE (bad predicted)')
                ax.scatter(i, prob, facecolors=color, edgecolors=edge_color,
                           marker=marker, s=80, label=pe_label)
            else:  # Control class
                color = 'lightsteelblue' if prob < threshold else 'none'
                edge_color = 'lightsteelblue'
                marker = 'o'  # Circle for Control class
                ctrl_label = ('Control (good predicted)' if color ==
                              'lightsteelblue' else 'Control (bad predicted)')
                ax.scatter(i, prob, facecolors=color, edgecolors=edge_color,
                           marker=marker, s=80, label=ctrl_label)

        # Add threshold line and labels
        ax.axhline(y=threshold, color='gray', linestyle='--',
                   label=f'Threshold = {threshold}')
        ax.set_title(f"{title} - {set_name}")
        ax.set_xlabel("Patients (ranked by probability)")
        ax.set_ylabel("Probability")

        # Filter duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    # Plot training and validation subplots
    plot_subplot(train_probs, y_train, axes[0], "Train")
    plot_subplot(val_probs, y_val, axes[1], "Validation")

    plt.tight_layout()
    plt.show()


class ThresholdedModel(BaseEstimator, ClassifierMixin):
    """
    A wrapper classifier that applies a custom threshold to a base model's
    predicted probabilities.

    This class allows using any classifier model that outputs probabilities
    and adjusts the decision threshold for classification, making it more
    flexible in cases where you want to tune the sensitivity/specificity
    trade-off.

    Attributes:
        model (Any): The base classifier model with `predict_proba` method.
        threshold (float): Custom threshold for converting probabilities to
            binary predictions.
    """

    def __init__(self, model: Any, threshold: float = 0.5):
        """
        Initialize the ThresholdedModel with a base model and a custom
        threshold.

        Args:
            model (Any): A classifier model that implements `predict_proba`.
            threshold (float): Probability threshold for predicting class 1.
                Defaults to 0.5.
        """
        self.model = model
        self.threshold = threshold
        self.classes_ = self.model.classes_

    def fit(self, X, y):
        """
        Fit the base model to the training data.

        Args:
            X (array-like): Feature matrix for training.
            y (array-like): Target labels for training.

        Returns:
            self: Fitted instance of ThresholdedModel.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        """
        Predict class labels based on the custom threshold.

        Args:
            X (array-like): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted class labels based on the custom threshold.
        """
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Predict class probabilities using the base model.

        Args:
            X (array-like): Feature matrix for probability prediction.

        Returns:
            np.ndarray: Predicted probabilities from the base model.
        """
        return self.model.predict_proba(X)


def regression_coefs(file_path: str, title: str) -> None:
    """
    Load data, train regression models, and plot feature coefficients without
    scaling.

    This function loads the dataset, splits it into training and validation
    sets, and trains three models (Linear Regression, Lasso, and Elastic Net)
    without scaling the data. It then plots the coefficients of each model
    for comparison.

    Args:
        file_path (str): Path to the CSV file containing the data.
        title (str): Title for the plots, typically representing the dataset
            or analysis type.
    """
    # Load dataset
    cm = pd.read_csv(file_path, index_col=0)

    # Split data into train and validation sets
    meta_columns = ["class", "group"]
    cols = [col for col in cm.columns if col not in meta_columns]
    train = cm[cm.group == "train"].drop(["group"], axis=1)

    X_train = train.drop("class", axis=1)
    y_train = train["class"]

    # Train models without scaling
    lin_reg = LinearRegression().fit(X_train, y_train)
    lasso = Lasso(alpha=0.1).fit(X_train, y_train)
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)

    # Retrieve coefficients for each model
    coef_lin_reg = lin_reg.coef_
    coef_lasso = lasso.coef_
    coef_elastic_net = elastic_net.coef_

    # Plot coefficients
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    model_names = ['Linear Regression', 'Lasso', 'Elastic Net']
    coefficients = [coef_lin_reg, coef_lasso, coef_elastic_net]

    for ax, coef, model in zip(axes, coefficients, model_names):
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        sorted_features = np.array(cols)[sorted_idx]
        ax.bar(range(len(coef)), coef[sorted_idx])
        ax.set_xticks(range(len(coef)))
        ax.set_xticklabels(sorted_features, rotation=90, fontsize=6)
        ax.set_title(f"{title} - {model}")
        ax.set_ylabel("Coefficient Value")
        ax.grid(True)

    axes[-1].set_xlabel("Features")
    plt.tight_layout()
    plt.show()


def plot_pdp_ice(model: Any,
                 X_val: Any,
                 feature_names: List[str],
                 title: str) -> None:
    """
    Plots Partial Dependence Plots (PDP) and Individual Conditional
    Expectation (ICE) for a model.

    This function splits features into rows of up to 5 features per row, and
    plots PDP and ICE for each feature. It adapts the layout dynamically
    based on the number of features.

    Args:
        model (Any): A fitted model with a `predict_proba` or `predict`
            method for PDP and ICE.
        X_val (Any): Validation feature data (should match model input
            format).
        feature_names (List[str]): List of feature names to plot.
        title (str): Title for the overall plot.
    """
    # Split features into chunks of 5 for rows
    feature_chunks = [feature_names[i:i + 5]
                      for i in range(0, len(feature_names), 5)]
    num_rows = len(feature_chunks)

    # Set up the figure
    fig, axes = plt.subplots(num_rows, 5, figsize=(20, 5 * num_rows),
                             sharey=True)
    axes = axes.flatten()

    # Plot PDP and ICE for each feature in the layout
    for row, features in enumerate(feature_chunks):
        for i, feature in enumerate(features):
            ax_idx = row * 5 + i
            PartialDependenceDisplay.from_estimator(
                model, X_val, [feature], kind="both", ax=axes[ax_idx],
                grid_resolution=50
            )
            axes[ax_idx].set_title(f"{title} - {feature}")
            axes[ax_idx].set_xlabel("Feature Value")
            axes[ax_idx].set_ylabel("Partial Dependence / ICE")

    # Hide any extra axes if features are not a multiple of 5
    for j in range(len(feature_chunks) * 5, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"PDP and ICE on Validation Set - {title}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def get_gene_length(gene_id: str) -> Optional[int]:
    """
    Fetches the genomic length of a gene from the Ensembl REST API.

    Args:
        gene_id (str): Ensembl gene ID.

    Returns:
        int: The length of the gene in base pairs.
        None: If the request fails or the gene is not found.

    Raises:
        HTTPError: If the API request encounters an error.
    """
    url = f"https://rest.ensembl.org/lookup/id/{gene_id}?expand=1"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        # Calculate the gene length
        gene_length = data['end'] - data['start'] + 1
        return gene_length
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {gene_id}: {e}")
        return None


def get_gene_length_cod(gene_id: str) -> Optional[int]:
    """
    Fetches the total coding exon length of a gene from the Ensembl REST API.

    Args:
        gene_id (str): Ensembl gene ID.

    Returns:
        int: The total coding exon length in base pairs.
        None: If the request fails or the gene is not found.

    Raises:
        HTTPError: If the API request encounters an error.
    """
    url = f"https://rest.ensembl.org/lookup/id/{gene_id}?expand=1"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        coding_length = 0
        # Sum lengths of all coding exons in the transcripts
        for transcript in data.get("Transcript", []):
            for exon in transcript.get("Exon", []):
                if "start" in exon and "end" in exon:
                    coding_length += exon["end"] - exon["start"] + 1

        return coding_length
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {gene_id}: {e}")
        return None


def plot_expression_vs_exon_length(df: pd.DataFrame,
                                   exon_lengths: dict,
                                   timepoint: str,
                                   point_size: int = 10,
                                   color: str = 'blue') -> None:
    """
    Plots gene expression levels against exon lengths for a given timepoint.

    Args:
        df (pd.DataFrame): DataFrame containing gene expression data with
            genes as columns and samples as rows. The index should include
            timepoint information for filtering.
        exon_lengths (dict): Dictionary mapping gene names to exon lengths
            (in bases).
        timepoint (str): Timepoint to filter the DataFrame (e.g., 'T1', 'T2').
        point_size (int, optional): Size of the scatterplot points.
            Defaults to 10.
        color (str, optional): Color of the points in the scatterplot.
            Defaults to 'blue'.

    Returns:
        None: Displays the scatterplot with regression line and prints
            Pearson correlation coefficient.
    """
    # Filter rows based on the given timepoint
    df_filtered = df[df.index.str.endswith(timepoint)]

    # Extract exon lengths for genes in the DataFrame columns
    exon_length_list = [
        exon_lengths.get(gene, None) for gene in df_filtered.columns
    ]

    # Filter out genes with missing exon lengths
    valid_indices = [i for i, length in enumerate(exon_length_list)
                     if length is not None]
    exon_length_list = [exon_length_list[i] for i in valid_indices]
    expression_values = df_filtered.iloc[:, valid_indices].values.flatten()

    # Repeat exon lengths for each patient (row in the DataFrame)
    exon_lengths_repeated = np.tile(exon_length_list, df_filtered.shape[0])

    # Calculate Pearson correlation coefficient
    r, _ = pearsonr(exon_lengths_repeated, expression_values)
    print(f'Pearson Correlation Coefficient: {r:.2f}')

    # Fit a linear regression line
    slope, intercept = np.polyfit(exon_lengths_repeated, expression_values,
                                  1)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(exon_lengths_repeated, expression_values, alpha=0.7,
                color=color, s=point_size)
    # Plot the regression line
    x = np.linspace(min(exon_lengths_repeated), max(exon_lengths_repeated),
                    500)
    plt.plot(x, slope * x + intercept, color='red', linewidth=2,
             label=f'Linear Regression\nSlope: {slope:.2f}')

    # Add plot labels and title
    plt.title(f'Gene Expression vs Exon Length ({timepoint})')
    plt.xlabel('Exon Length (bases)')
    plt.ylabel('Expression Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
