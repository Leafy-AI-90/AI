import numpy as np
import pytest

from leafy_ai.features import absorbance, positional_features, reflectance, spectral_features
from leafy_ai.models import intermediate_fusion, spectral_model
from leafy_ai.validation import cycle_folds, grouped_folds


def test_repeated_observations_never_cross_folds():
    plants = np.repeat(np.arange(50), 3)
    y = np.repeat(np.tile(np.arange(5), 10), 3)
    seen = []
    for train, test in grouped_folds(y, plants):
        assert not set(plants[train]) & set(plants[test])
        assert set(y[test]) == set(range(5))
        seen.extend(test)
    assert sorted(seen) == list(range(len(y)))


def test_bad_group_labels_and_insufficient_plants_rejected():
    with pytest.raises(ValueError):
        list(grouped_folds([0, 1], ["same", "same"], 2))
    with pytest.raises(ValueError):
        list(grouped_folds([0, 1], ["a", "b"], 2))


def test_cycles_isolated_and_reused_ids_rejected():
    for train, test in cycle_folds([1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3]):
        assert not set(train) & set(test)
        assert len(test) == 2
    with pytest.raises(ValueError):
        list(cycle_folds([1, 1], [1, 2]))


def test_calibration_and_positional_signs():
    assert reflectance(60, 10, 110) == 0.5
    assert absorbance(10, 100) == 1
    with pytest.raises(ValueError):
        reflectance(10, 10, 10)
    with pytest.raises(ValueError):
        absorbance(0, 1)
    f = positional_features([[1, 2, 0, 3]])
    np.testing.assert_allclose(f[0, :2], [-1, 3])
    assert spectral_features(np.full((2, 18), 0.5), np.ones((2, 3))).shape == (2, 45)


def test_normalization_is_fit_on_train_only():
    train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    model = spectral_model("logistic").fit(train, [0, 0, 1, 1])
    original = model[0].mean_.copy()
    model.predict([[999.0, 999.0]])
    np.testing.assert_array_equal(model[0].mean_, original)
    np.testing.assert_allclose(original, train.mean(axis=0))


def test_fusion_pca_sees_only_training_rows():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(30, 12))
    model = intermediate_fusion(8, 12, pca_components=3).fit(train, np.tile([0, 1, 2], 10))
    transformer = model[0].named_transformers_["image"]
    mean = transformer[0].mean_.copy()
    model.predict(np.full((2, 12), 1000.0))
    np.testing.assert_allclose(mean, train[:, :8].mean(0))
    np.testing.assert_array_equal(transformer[0].mean_, mean)
