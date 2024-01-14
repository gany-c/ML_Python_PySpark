import json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris  # Import dataset loader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_dataset(name, random_seed=None):
    # Load dataset based on name
    if name.lower() == "iris":
        dataset = load_iris()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split the dataset into features (X) and labels (y)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=random_seed
    )

    return X_train, X_test, y_train, y_test


def build_pipeline(config):
    steps = []

    # Load the specified dataset
    dataset_name = config.get("dataset", {}).get("name", "iris")
    dataset_params = config.get("dataset", {}).get("params", {})
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, **dataset_params)

    for step_config in config["preprocessing_steps"]:
        step_name = step_config["name"]
        step_params = step_config.get("params", {})

        if step_name == "imputation":
            step = (step_name, SimpleImputer(**step_params))
        elif step_name == "scaling":
            step = (step_name, StandardScaler(**step_params))
        else:
            raise ValueError(f"Unknown preprocessing step: {step_name}")

        steps.append(step)

    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})

    if model_name == "LogisticRegression":
        model = LogisticRegression(**model_params)
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    steps.append(("model", model))

    return Pipeline(steps), X_train, X_test, y_train, y_test


def pipeline_wrapper():
    with open('config.json', 'r') as f:
        config = json.load(f)

    pipeline, x_train, x_test, y_train, y_test = build_pipeline(config)

    # Now you can train the model on the full training data and proceed with the inference step
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Print the overall accuracy
    print("Accuracy Score:")
    print(accuracy_score(y_test, predictions))


if __name__ == "__main__":
    pipeline_wrapper()
