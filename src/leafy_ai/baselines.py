import numpy as np


def low_chlorophyll(m660, threshold):
    """B0 binary comparator. Select threshold using calibration/training data only.

    True denotes below-threshold chlorophyll proxy, never a nutrient/disease label.
    """
    values = np.asarray(m660, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Finite M660 values and threshold required")
    return values < threshold
