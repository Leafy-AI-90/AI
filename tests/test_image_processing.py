import numpy as np
import pytest
from PIL import Image

from leafy_ai.colour_correction import (
    apply_ccm,
    correct_image,
    fit_ccm,
    linear_to_srgb,
    reference_linear_rgb,
    srgb_to_linear,
)
from leafy_ai.segmentation import leaf_mask, otsu_threshold, remove_background, segment_image


def checker_image(colour_cast=None):
    ref = reference_linear_rgb()
    if colour_cast is not None:
        ref = ref @ colour_cast
    srgb = np.round(linear_to_srgb(ref) * 255).astype(np.uint8)
    img = np.full((600, 900, 3), 30, np.uint8)
    for i in range(24):
        r, c = divmod(i, 6)
        img[120 + r * 100 : 200 + r * 100, 150 + c * 100 : 230 + c * 100] = srgb[i]
    return img


def test_srgb_round_trip_and_reference_values():
    x = np.linspace(0, 1, 50)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-12)
    srgb = np.round(linear_to_srgb(reference_linear_rgb()) * 255)
    np.testing.assert_allclose(srgb[0], [115, 82, 68], atol=3)  # dark skin
    assert srgb[18].min() > 230 and srgb[23].max() < 60  # white and black patches


def test_fit_recovers_known_matrix_and_rotation():
    ref = reference_linear_rgb()
    cast = np.array([[0.9, 0.05, 0.0], [0.1, 0.8, 0.05], [0.0, 0.1, 1.1]])
    measured = ref @ np.linalg.inv(cast)
    fit = fit_ccm(measured, ref)
    np.testing.assert_allclose(fit.matrix, cast, atol=1e-9)
    assert fit.rmse < 1e-9 and not fit.reversed_order
    assert fit_ccm(measured[::-1], ref).reversed_order
    with pytest.raises(ValueError):
        fit_ccm(measured[:10], ref)


def test_detect_and_correct_colour_cast(tmp_path):
    cast = np.diag([1.15, 1.0, 0.8])
    path = tmp_path / "cast.png"
    Image.fromarray(checker_image(cast)).save(path)
    corrected, fit = correct_image(path, tmp_path / "out")
    assert corrected.exists() and fit.rmse < 0.02
    np.testing.assert_allclose(np.diag(fit.matrix), 1 / np.diag(cast), atol=0.1)
    blank = tmp_path / "blank.png"
    Image.fromarray(np.full((300, 300, 3), 128, np.uint8)).save(blank)
    assert correct_image(blank, tmp_path / "out") == (None, None)


def test_leaf_mask_on_green_disc(tmp_path):
    yy, xx = np.mgrid[:200, :200]
    disc = (yy - 100) ** 2 + (xx - 100) ** 2 < 60**2
    img = np.zeros((200, 200, 3), np.uint8)
    img[:] = (150, 120, 100)  # soil-like background
    img[disc] = (60, 140, 50)
    mask = leaf_mask(img)
    assert (mask == disc).mean() > 0.99
    assert (remove_background(img, mask)[~mask] == 0).all()
    path = tmp_path / "leaf.png"
    Image.fromarray(img).save(path)
    mask_path, nobg_path, fraction = segment_image(path, tmp_path / "seg")
    assert mask_path.exists() and nobg_path.exists()
    assert fraction == pytest.approx(disc.mean(), abs=0.01)


def test_otsu_two_levels_and_constant_rejected():
    values = np.r_[np.zeros(100), np.ones(100)]
    assert 0 < otsu_threshold(values) < 1
    with pytest.raises(ValueError):
        otsu_threshold(np.ones(10))


def test_apply_identity_ccm_preserves_image():
    img = checker_image()
    np.testing.assert_array_equal(apply_ccm(img, np.eye(3)), img)
