from dataclasses import dataclass
from typing import Dict, Any

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelResult:
    name: str
    accuracy: float
    report: str


def load_data():
    """Load the Iris dataset from scikit-learn."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names
    return X, y, target_names, feature_names


def build_models() -> Dict[str, Any]:
    """Create a dictionary of candidate models wrapped in Pipelines."""
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf")),
            ]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    return models


def evaluate_models(X, y) -> Dict[str, ModelResult]:
    """Train and evaluate multiple models, returning their performance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results: Dict[str, ModelResult] = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        results[name] = ModelResult(name=name, accuracy=acc, report=report)

    return results


def main():
    X, y, target_names, feature_names = load_data()
    print("Features:", feature_names)
    print("Target classes:", target_names)

    results = evaluate_models(X, y)

    print("\n\n=== MODEL COMPARISON ===")
    for name, res in results.items():
        print(f"\n{name}")
        print(f"Accuracy: {res.accuracy:.3f}")
        print("Classification Report:")
        print(res.report)


if __name__ == "__main__":
    main()
