"""Small shared assertions for numerically replayed tabular artifacts."""
import pandas as pd


def assert_same_table(observed, expected):
    """Compare values while normalizing Arrow/pandas null representations."""
    pd.testing.assert_frame_equal(observed.isna(), expected.isna(), check_dtype=False)
    left = observed.astype(object).where(observed.notna(), None)
    right = expected.astype(object).where(expected.notna(), None)
    pd.testing.assert_frame_equal(
        left, right, check_dtype=False, check_exact=False, rtol=0, atol=1e-12
    )


__all__ = ["assert_same_table"]
