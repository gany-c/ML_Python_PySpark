import sys
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes   # Import dataset loader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score


def load_dataset(name, random_seed=None):
    # Load dataset based on name

    if name.lower() == "iris":
        dataset = load_iris()
    elif name.lower() == "california_housing":
        dataset = fetch_california_housing()
    elif name.lower() == "diabetes":
        dataset = load_diabetes()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print("dataset.target = ", dataset.target)
    # Split the dataset into features (X) and labels (y)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=random_seed
    )

    return x_train, x_test, y_train, y_test


def create_model(config):
    """
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
    steps = []

    # TODO: What other pre-processing steps can be used besides
    # Scaling and imputation?
    for step_config in config["preprocessing_steps"]:
        step_name = step_config["name"]
        step_params = step_config.get("params", {})

        # Handle missing values
        if step_name == "imputation":
            step = (step_name, SimpleImputer(**step_params))
        elif step_name == "scaling":
            step = (step_name, StandardScaler(**step_params))
        else:
            raise ValueError(f"Unknown preprocessing step: {step_name}")

        steps.append(step)

    model = create_model(config)
    steps.append(("model", model))

    return Pipeline(steps)


def evaluate_model(y_test, predictions, config):
    model_type = config["model"]["type"]

    if model_type == "Classifier":
        # Print the classification report
        print("Classification Report:")
        print(classification_report(y_test, predictions))

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        # Print the overall accuracy
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
    joblib.dump(pipeline, output_path)


def pipeline_wrapper():

    print(sys.argv)
    if len(sys.argv) < 2:
        print("Please provided a config file: python config_based_ml_pipeline.py config.json")
        sys.exit(1)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load the specified dataset
    dataset_name = config.get("dataset", {}).get("name")
    dataset_params = config.get("dataset", {}).get("params", {})
    x_train, x_test, y_train, y_test = load_dataset(dataset_name, **dataset_params)

    pipeline = build_pipeline(config)

    # train the model on the training data
    pipeline.fit(x_train, y_train)

    save_pipeline(pipeline, config["model_output_path"])

    predictions = pipeline.predict(x_test)

    evaluate_model(y_test, predictions, config)


if __name__ == "__main__":
    pipeline_wrapper()
