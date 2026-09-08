import numpy as np
import pandas as pd

from fucci_pca import compute_color_z_pca, pca_output_tables


def test_pca_centers_without_scaling_and_orients_sign() -> None:
    matrix = pd.DataFrame({"track_uid": ["a", "b", "c"], "s0": [1.0, 2.0, 3.0], "s1": [2.0, 4.0, 6.0]})
    result = compute_color_z_pca(matrix, ["s0", "s1"])
    assert np.allclose(result.centered_matrix.mean(axis=0), 0)
    assert result.loadings[0, np.argmax(np.abs(result.loadings[0]))] > 0
    tables = pca_output_tables(result, pd.DataFrame({"session_id": ["s0", "s1"], "elapsed_days": [0, 2]}))
    assert np.isclose(tables["explained_variance"]["explained_variance_ratio"].sum(), 1)
    reconstructed = result.scores @ result.loadings + result.column_means
    assert np.allclose(reconstructed, matrix[["s0", "s1"]])
