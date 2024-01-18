import sys
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes   # Import dataset loader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def load_dataset(config):
    """
    Loads the dataset specified in the config.
    This can either be a local file e.g. Titancic
    or an inbuilt sklearn dataset e.g. Iris or California
    housing.

    Also does the test train split.

    :param config: Configuration dict
    :return: The loaded dataset, split into test and train
    """
    try:
        # Load the specified dataset
        dataset_type = config.get("dataset", {}).get("type")
        dataset_params = config.get("dataset", {}).get("params", {})

        if dataset_type == "file":
            dataset_path = config.get("dataset", {}).get("dataset_path")
            try:
                dataset = pd.read_csv(dataset_path)
            except FileNotFoundError as fnf:
                print(f"File not found at path: {dataset_path}")
                raise fnf

            target_column = config.get("dataset", {}).get("target_column")
            feat_columns = config.get("dataset", {}).get("features")

            features = dataset[feat_columns]
            target = dataset[target_column]
        else:
            dataset_name = config.get("dataset", {}).get("name")

            if dataset_name.lower() == "iris":
                dataset = load_iris()
            elif dataset_name.lower() == "california_housing":
                dataset = fetch_california_housing()
            elif dataset_name.lower() == "diabetes":
                dataset = load_diabetes()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            features = dataset.data
            target = dataset.target

        test_size = config.get("dataset", {}).get("params", {}).get("test_size", 0.2)
        if not 0 <= test_size <= 1:
            raise ValueError(f"Invalid test size: {test_size}. It should be a float between 0 and 1.")

        # Split the dataset into features (X) and labels (y)
        x_train, x_test, y_train, y_test = train_test_split(
           features, target, test_size=test_size, random_state=dataset_params.get("random_seed")
        )

        return x_train, x_test, y_train, y_test
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        raise e


def create_model(config):
    """
    Currently the pipeline supports Linear Regression,
    Logistic Regression and RandomForestClassifiers

    Parameters:
        config (dict): Configuration dictionary containing model details.

    Returns:
        object: The instantiated  model.
    """
    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})

    if model_name == "LogisticRegression":
        model = LogisticRegression(**model_params)
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**model_params)
    elif model_name == "LinearRegression":
        model = LinearRegression(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def build_pipeline(config):
    """
    Build a scikit-learn pipeline based on the provided configuration.
    It does both pre-processing and model creation

    Pre-processing supports Imputation, Scaling and OneHot Encoding

    :param config: Configuration dictionary containing details for
    preprocessing and modeling.
    :return: The constructed scikit-learn pipeline.
    """
    try:
        steps = []

        for step_config in config["preprocessing_steps"]:
            step_name = step_config["name"]
            step_params = step_config.get("params", {})

            # Handle missing values
            if step_name == "imputation":
                step = (step_name, SimpleImputer(**step_params))
            elif step_name == "scaling":
                step = (step_name, StandardScaler(**step_params))
            elif step_name == "onehot":
                categorical_features = step_params.get("categorical_features", [])
                step = ("onehot", ColumnTransformer(transformers=
                                                    [("onehot", OneHotEncoder(), categorical_features)],
                                                    remainder='passthrough'))

            else:
                raise ValueError(f"Unknown preprocessing step: {step_name}")

            steps.append(step)

        model = create_model(config)
        steps.append(("model", model))

        return Pipeline(steps)
    except Exception as e:
        print(f"An error occurred while constructing the pipeline: {e}")
        raise


def evaluate_model(y_test, predictions, config):
    """
    Evaluate the performance of a machine learning model based on the
    provided predictions and configuration.

    Parameters:
        y_test (array-like): True labels or values of the target variable.
        predictions (array-like): Predicted labels or values from the model.
        config (dict): Configuration dictionary containing model details.

    Returns:
        None

    Raises:
        ValueError: If the model type in the configuration is neither
        'Classifier' nor 'Regressor'.
    """

    model_type = config["model"]["type"]

    if model_type == "Classifier":
        # Print the classification report
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        # Print the overall accuracy
        """
        Accuracy = (True Positives + True Negatives)/ Total Number of Predictions
        """
        print("Accuracy Score:")
        print(accuracy_score(y_test, predictions))

    elif model_type == "Regressor":
        # Regression evaluation
        print("Mean Squared Error:")
        print(mean_squared_error(y_test, predictions))

        print("R-squared Score:")
        print(r2_score(y_test, predictions))

    else:
        raise ValueError("Invalid model type for evaluation. Must be 'Classifier' or 'Regressor'.")


def save_pipeline(pipeline, output_path):
    """
    Saves the model/pipeline to an external path
    using the joblib utility

    :param pipeline:
    :param output_path:
    :return:
    """
    joblib.dump(pipeline, output_path)


def load_config_from_file(config_file_path):
    """
    Load configuration from a JSON file.

    Parameters:
        config_file_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration loaded from the file.
    """
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    return config


def pipeline_wrapper():
    """
    The main method

    1. Loads the configuration
    2. Loads the dataset from the configuration
    3. Builds the sklearn pipeline from the configuration
    4. Run training using the pipeline
    5. Save the pipeline to an external file
    6. Make predictions using the pipeline
    7. Print evaluation metrics on the model

    :return:
    """

    print(sys.argv)
    if len(sys.argv) < 2:
        print("Please provided a config file: python config_based_ml_pipeline.py config.json")
        sys.exit(1)

    config_file_path = sys.argv[1]
    config = load_config_from_file(config_file_path)

    # Load the specified dataset
    x_train, x_test, y_train, y_test = load_dataset(config)

    pipeline = build_pipeline(config)

    # train the model on the training data
    pipeline.fit(x_train, y_train)

    save_pipeline(pipeline, config["model_output_path"])

    # This runs the pipeline's preprocessing steps and prediction of the test data
    predictions = pipeline.predict(x_test)

    evaluate_model(y_test, predictions, config)


if __name__ == "__main__":
    pipeline_wrapper()
