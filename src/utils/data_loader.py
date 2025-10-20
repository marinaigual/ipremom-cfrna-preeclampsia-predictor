""" data_loader.py """
import yaml
import pickle
import numpy as np
import pandas as pd
from datetime import date

from typing import Any


# Load functions
def get_train_validation_input(
        path_file_input: str,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Obtain the discovery and validation datasets from input file path.

    Args:
        path_file_input (str): Input file path

    Returns:
        tuple[np.array, np.array, np.array, np.array]:
            discovery predictive variables,
            discovery labels,
            validation predictive variables,
            validation labels,

    """
    model_input_df = pd.read_csv(path_file_input, index_col=0)
    training_data = (
        model_input_df[model_input_df.group == "train"].drop(["group"],
                                                             axis=1))
    validation_data = (
        model_input_df[model_input_df.group == "validation"].drop(["group"],
                                                                  axis=1))

    x_train = np.array(training_data.drop("class", axis=1))
    y_train = np.array(training_data["class"])

    x_val = np.array(validation_data.drop("class", axis=1))
    y_val = np.array(validation_data["class"])
    return x_train, y_train, x_val, y_val


def get_train_validation_input_pandas(
        path_file_input: str,
) -> tuple[np.array, np.array, np.array, np.array]:
    """Obtain the discovery and validation datasets from input file path.

    Args:
        path_file_input (str): Input file path

    Returns:
        tuple[np.array, np.array, np.array, np.array]:
            discovery predictive variables,
            discovery labels,
            validation predictive variables,
            validation labels,

    """
    model_input_df = pd.read_csv(path_file_input, index_col=0)
    training_data = (
        model_input_df[model_input_df.group == "train"].drop(["group"],
                                                             axis=1))
    validation_data = (
        model_input_df[model_input_df.group == "validation"].drop(["group"],
                                                                  axis=1))

    x_train = training_data.drop("class", axis=1)
    y_train = training_data["class"]

    x_val = validation_data.drop("class", axis=1)
    y_val = validation_data["class"]
    return x_train, y_train, x_val, y_val


def load_numeric_data(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and filter for numeric columns only.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
            - A DataFrame containing only numeric features.
            - A Series containing the target class labels.
    """
    # Load the dataset
    data = pd.read_csv(file_path, index_col=0)

    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[float, int])

    # Extract target class labels
    class_labels = data["class"]

    return numeric_data, class_labels


def load_predictor(path_model: str) -> Any:
    """Load a trained model from a specified path using pickle.

    Args:
        path_model (str): Path to the saved model file.

    Returns:
        Any: Loaded trained model.
    """
    with open(path_model, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model


def load_configuration_yaml(
        path_configuration_yaml: str,
        ) -> dict:
    """Obtain dictionary with configuration parameters from .yaml file

    Args:∫∫
        path_configuration_yaml (str): .yaml file with configuration parameters

    Returns:
        dict: dictionary with configuration parameters from .yaml file
    """
    with open(path_configuration_yaml, 'r', encoding="utf-8") as file_in:
        return yaml.load(file_in, Loader=yaml.SafeLoader)


# Save functions
def save_trained_model(
        trained_clf: Any,
        path_file_name: str,
        ) -> None:
    """Create a .sav file with the trained model

    Args:
        trained_clf (Any): Trained ML algorithm
        path_file_name (str): path to save the model (locally)
    """
    pickle.dump(trained_clf, open(path_file_name, 'wb'))


def save_trained_thr(
        trained_thr: float,
        path_file_name_thr: str,
        ) -> None:
    """Create a .txt file with the trained thr of the model

    Args:
        trained_thr (Any): Trained threshold of the ML algorithm
        path_file_name_thr (str): path to save the threshold (locally)
    """
    with open(path_file_name_thr, "w") as file_out:
        file_out.write(str(trained_thr))


def save_predictor(
        trained_clf: Any,
        trained_thr: float,
        path_folder_save: str,
        num_variables: int = None,
        dataset_name: str = "dataset",
        ) -> None:
    """Save trained model and threshold locally

    Args:
        trained_clf (Any): Trained ML algorithm
        trained_thr (float): Trained threshold of the ML algorithm
        path_folder_save (str): path to save the model and threshold
        num_variables (int, optional): Number of variables used in the model.
            Defaults to None.
        dataset_name (str, optional): Dataset name.
            Defaults to "dataset".
    """
    model_name = type(trained_clf).__name__
    today = date.today()
    date_now = today.strftime("%m%d%y")
    if not num_variables:
        num_variables = "X"
    else:
        num_variables = str(num_variables)
    filename_model = (dataset_name + "_" + num_variables
                      + "_" + model_name + "_optimal_"
                      + date_now + ".sav")
    filename_thr = (dataset_name + "_" + num_variables
                    + "_opt_thr_" + model_name + "_"
                    + date_now + ".txt")
    path_file_name_model = path_folder_save + "/" + filename_model
    path_file_name_thr = path_folder_save + "/" + filename_thr
    save_trained_model(
        trained_clf, path_file_name_model)
    save_trained_thr(
        trained_thr, path_file_name_thr)
    
