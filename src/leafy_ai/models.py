"""Trainable spectral baselines and fusion components from review section 6.10."""
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBClassifier
from .validation import grouped_folds


class PLSDA(ClassifierMixin, BaseEstimator):
    """PLS on one-hot labels. Softmax scores are not calibrated probabilities."""
    def __init__(self, n_components=3):
        self.n_components = n_components

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Classification requires at least two classes")
        self.n_features_in_ = X.shape[1]
        n = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.model_ = PLSRegression(n_components=n).fit(X, np.eye(len(self.classes_))[encoded])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return softmax(self.model_.predict(check_array(X)), axis=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def spectral_model(name, seed=42):
    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        "pls_da": lambda: PLSDA(),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=2),
        # No built-in SVC probability CV: that CV does not respect plant groups.
        "svm": lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=False, random_state=seed)),
        # No random validation split for early stopping: tune using grouped folds.
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, early_stopping=False, random_state=seed)),
        "xgboost": lambda: XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, tree_method="hist", n_jobs=2, random_state=seed),
    }
    if name not in models:
        raise ValueError(f"Unknown spectral model: {name}")
    return models[name]()


def chlorophyll_model(n_components=3):
    return PLSRegression(n_components=n_components)


def intermediate_fusion(embedding_dim, total_dim, pca_components=32, seed=42):
    """F2: [image embedding | spectra | optional positional features].

    PCA is fitted only when this pipeline is fit on a training fold. Reduce
    pca_components explicitly when the smallest training fold has fewer rows.
    Embeddings must come from a frozen ImageNet encoder or a fold-local encoder.
    """
    if not 0 < embedding_dim < total_dim:
        raise ValueError("Require both image and spectral columns")
    transform = ColumnTransformer([
        ("image", make_pipeline(StandardScaler(), PCA(n_components=pca_components, random_state=seed)), list(range(embedding_dim))),
        ("spectral", "passthrough", list(range(embedding_dim, total_dim))),
    ])
    return make_pipeline(transform, spectral_model("xgboost", seed))


class GroupedLateFusion:
    """F1: cross-fitted probabilities, then logistic stacking.

    Inputs are separate numeric views. Supply estimators with predict_proba.
    Evaluate this whole object in an outer plant/cycle split, never its fit data.
    Any supervised image feature extraction must also be nested per fold.
    """
    def __init__(self, estimators, n_splits=3, seed=42):
        self.estimators, self.n_splits, self.seed = estimators, n_splits, seed

    def fit(self, views, y, plants):
        y = np.asarray(y)
        views = [check_array(x) for x in views]
        if len(views) != len(self.estimators) or not views or any(len(x) != len(y) for x in views):
            raise ValueError("One aligned view per estimator required")
        self.classes_ = np.unique(y)
        c = len(self.classes_)
        oof = np.zeros((len(y), c * len(views)))
        splits = list(grouped_folds(y, plants, self.n_splits, self.seed))
        self.models_ = []
        for i, (estimator, x) in enumerate(zip(self.estimators, views)):
            for train, test in splits:
                fitted = clone(estimator).fit(x[train], y[train])
                if not np.array_equal(fitted.classes_, self.classes_):
                    raise ValueError("All classes must be present in each training fold")
                oof[test, i*c:(i+1)*c] = fitted.predict_proba(x[test])
            self.models_.append(clone(estimator).fit(x, y))
        self.meta_ = LogisticRegression(max_iter=2000, class_weight="balanced").fit(oof, y)
        return self

    def predict_proba(self, views):
        check_is_fitted(self, "meta_")
        if len(views) != len(self.models_):
            raise ValueError("Wrong number of views")
        return self.meta_.predict_proba(np.column_stack([m.predict_proba(x) for m, x in zip(self.models_, views)]))

    def predict(self, views):
        return self.classes_[self.predict_proba(views).argmax(axis=1)]
