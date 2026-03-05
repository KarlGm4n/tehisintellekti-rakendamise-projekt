import pytest
import pandas as pd
from app5_ready import apply_filters

def test_apply_filters_categorical():
    df = pd.DataFrame({
        'semester': ['kevad', 'sügis', 'kevad'],
        'keel': ['ET', 'EN', 'ET'],
        'eap': [3, 6, 9],
    })
    filters = {'semester': 'kevad', 'keel': 'ET'}
    filtered = apply_filters(df, filters)
    assert len(filtered) == 2
    assert all(filtered['semester'] == 'kevad')
    assert all(filtered['keel'] == 'ET')
