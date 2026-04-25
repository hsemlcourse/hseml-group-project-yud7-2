import numpy as np

from src.preprocessing import label_from_npz_name, vectorize_timeseries


def test_label_from_npz_name() -> None:
    assert label_from_npz_name("PT11_12345_3301010000.npz") == "3301010000"


def test_vectorize_timeseries_pads_and_flattens() -> None:
    array = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    result = vectorize_timeseries(array, max_timesteps=3)

    assert result.tolist() == [1.0, 0.0, 3.0, 4.0, 0.0, 0.0]
