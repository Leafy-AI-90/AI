"""Hand-calculated checks for every feature equation."""

import math

import numpy as np
import pandas as pd
import pytest

from leafy_ai import BANDS
from leafy_ai.features import (
    SPECTRAL_NAMES,
    baseline_change,
    cwsi,
    delta_t,
    positional_contrasts,
    positional_features,
    recent_slope,
    saturation_vapour_pressure,
    spectral_features,
    temperature_spread,
    vpd,
    water_index,
)


def test_spectral_indices_by_hand():
    r = np.array([[0.05 + 0.01 * i for i in range(18)]])
    band = dict(zip(BANDS, r[0], strict=True))
    a = np.array([[0.9, 0.6, 0.2]])
    f = dict(zip(SPECTRAL_NAMES, spectral_features(r, a)[0], strict=True))
    assert f["ndvi"] == pytest.approx((band[810] - band[680]) / (band[810] + band[680]))
    assert f["ndre"] == pytest.approx((band[760] - band[705]) / (band[760] + band[705]))
    assert f["ci_red_edge"] == pytest.approx(band[760] / band[705] - 1)
    assert f["gndvi"] == pytest.approx((band[810] - band[560]) / (band[810] + band[560]))
    assert f["pri_proxy"] == pytest.approx((band[535] - band[560]) / (band[535] + band[560]))
    assert f["dr_410_435"] == pytest.approx(0.01 / 25)
    assert f["m660"] == pytest.approx(0.7) and f["m710"] == pytest.approx(0.4)


def test_water_index_by_hand():
    assert water_index([0.35], [0.20])[0] == pytest.approx(0.15)


def test_thermal_features_by_hand():
    assert delta_t(24.5, 27.0) == pytest.approx(-2.5)
    assert cwsi(25.0, 22.0, 30.0) == pytest.approx(3 / 8)
    assert temperature_spread(26.0, 23.5) == pytest.approx(2.5)
    assert math.isnan(cwsi(25.0, np.nan, 30.0))  # missing reference stays missing
    with pytest.raises(ValueError):
        cwsi(25.0, 30.0, 22.0)
    with pytest.raises(ValueError):
        temperature_spread(20.0, 21.0)


def test_vpd_by_hand():
    # FAO-56: e_s(20 C) = 2.338 kPa; at RH 60 % VPD = 0.935 kPa
    assert saturation_vapour_pressure(20.0) == pytest.approx(2.338, abs=1e-3)
    assert vpd(20.0, 60.0) == pytest.approx(2.338 * 0.4, abs=1e-3)
    assert vpd(30.0, 100.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        vpd(20.0, 120.0)


def test_positional_contrasts_for_three_and_four_positions():
    four = positional_contrasts(
        [[1.0, 2.0, 0.0, 3.0]], ["oldest", "youngest", "lesion", "off_lesion"]
    )
    assert four["age"][0] == -1 and four["lesion"][0] == 3 and "middle" not in four
    np.testing.assert_allclose(positional_features([[1, 2, 0, 3]])[0], [-1, 3, four["variance"][0]])
    three = positional_contrasts([[0.4, 0.1, 0.3]], ["oldest", "middle", "youngest"])
    assert three["age"][0] == pytest.approx(0.1)
    assert three["middle"][0] == pytest.approx(0.1 - 0.35)
    assert three["variance"][0] == pytest.approx(np.var([0.4, 0.1, 0.3]))
    with pytest.raises(ValueError):
        positional_contrasts([[1.0, 2.0]], ["oldest", "youngest"])


def plant_days():
    return pd.DataFrame(
        dict(
            plant_id=["a"] * 5 + ["b"] * 3,
            day=[-1, 0, 2, 4, 6, 0, 2, 4],
            x=[1.0, 2.0, 4.0, 5.0, 9.0, 1.0, 2.0, np.nan],
        )
    )


def test_baseline_change_by_hand():
    df = plant_days()
    delta = baseline_change(df, ["x"])["x_delta"]
    np.testing.assert_allclose(delta[:5], [0, 1, 3, 4, 8])
    assert delta[5:].isna().all()  # plant b has no day -1 baseline


def test_recent_slope_by_hand_and_causal():
    df = plant_days()
    slope = recent_slope(df, ["x"])["x_slope"]
    assert slope[:2].isna().all()
    assert slope[2] == pytest.approx(np.polyfit([-1, 0, 2], [1, 2, 4], 1)[0])
    assert slope[4] == pytest.approx(np.polyfit([2, 4, 6], [4, 5, 9], 1)[0])
    assert slope[7] != slope[7]  # NaN inside window stays NaN
    future = df.copy()
    future.loc[4, "x"] = 1000.0  # change last day only
    np.testing.assert_allclose(recent_slope(future, ["x"])["x_slope"][:4], slope[:4])


def test_shuffled_rows_give_same_temporal_features():
    df = plant_days()
    shuffled = df.sample(frac=1, random_state=1)
    a = recent_slope(df, ["x"]).sort_index()
    b = recent_slope(shuffled, ["x"]).sort_index()
    pd.testing.assert_frame_equal(a, b)
