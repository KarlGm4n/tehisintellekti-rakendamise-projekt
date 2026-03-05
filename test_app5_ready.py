import pytest
import pandas as pd
from app5_ready import apply_filters, merge_filters, get_allowed_values

def test_apply_filters_numeric():
    df = pd.DataFrame({
        'eap': [3, 6, 9],
        'semester': ['kevad', 'sügis', 'kevad'],
        'keel': ['ET', 'EN', 'ET'],
    })
    filters = {'eap': 6, 'semester': 'sügis', 'keel': 'EN'}
    filtered = apply_filters(df, filters)
    assert len(filtered) == 1
    assert filtered.iloc[0]['eap'] == 6

def test_merge_filters():
    current = {'semester': 'kevad', 'eap': 3}
    new = {'eap': 6, 'keel': 'EN'}
    merged = merge_filters(current, new)
    assert merged == {'semester': 'kevad', 'eap': 6, 'keel': 'EN'}

def test_get_allowed_values():
    df = pd.DataFrame({
        'semester': ['kevad', 'sügis', 'kevad'],
        'keel': ['ET', 'EN', 'ET'],
        'eap': [3, 6, 9],
    })
    allowed = get_allowed_values(df)
    assert 'semester' in allowed
    assert 'keel' in allowed
    assert allowed['semester'] == ['kevad', 'sügis'] or allowed['semester'] == ['sügis', 'kevad']
    assert allowed['keel'] == ['EN', 'ET'] or allowed['keel'] == ['ET', 'EN']
