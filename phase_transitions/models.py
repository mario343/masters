"""Canonical supervised model factory."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import Birch, KMeans, MeanShift
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_classifier(
    name: str,
    *,
    calibrated: bool = False,
    standardize: bool = False,
    calibration_cv: int = 5,
):
    """Build one classifier with explicit reproducibility parameters.

    Calibrated models expose probabilities.  The uncalibrated SVM path is
    retained only for historical comparison; the canonical scientific
    pipeline should use ``calibrated=True`` for SVM and MLP.
    """

    normalized = name.lower().replace(" ", "_")
    factories = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=0),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=0, n_jobs=-1
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
        "knn": lambda: KNeighborsClassifier(n_jobs=-1),
        "svm_rbf": lambda: SVC(kernel="rbf"),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(50, 50), max_iter=500, random_state=0
        ),
    }
    if normalized not in factories:
        raise KeyError(f"unknown classifier: {name}")
    if calibration_cv < 2:
        raise ValueError("calibration_cv must be at least 2")

    estimator = factories[normalized]()
    steps = []
    if standardize:
        from sklearn.preprocessing import StandardScaler

        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    pipeline = Pipeline(steps)

    if normalized in {"svm_rbf", "mlp"} and calibrated:
        return CalibratedClassifierCV(
            estimator=pipeline,
            method="sigmoid",
            cv=calibration_cv,
        )
    if calibrated and normalized not in {"svm_rbf", "mlp"}:
        return CalibratedClassifierCV(
            estimator=pipeline,
            method="sigmoid",
            cv=calibration_cv,
        )
    return pipeline


def build_clusterer(name: str):
    """Build an unsupervised model with explicit deterministic parameters."""

    normalized = name.lower().replace(" ", "_")
    factories = {
        "kmeans": lambda: KMeans(
            n_clusters=2,
            random_state=0,
            n_init="auto",
        ),
        "gaussian_mixture": lambda: GaussianMixture(
            n_components=2,
            random_state=0,
            max_iter=1000,
        ),
        "bayesian_gmm": lambda: BayesianGaussianMixture(
            n_components=2,
            random_state=0,
            max_iter=3000,
        ),
        "meanshift": lambda: MeanShift(n_jobs=-1),
        "birch": lambda: Birch(n_clusters=2),
    }
    if normalized not in factories:
        raise KeyError(f"unknown clusterer: {name}")
    return factories[normalized]()
