import pandas as pd
from typing import Any, Dict, List

FILTER_FIELDS = {
    "semester": {"column": "semester", "type": "categorical"},
    "eap": {"column": "eap", "type": "numeric"},
    "keel": {"column": "keel", "type": "categorical"},
    "linn": {"column": "linn", "type": "categorical"},
    "oppeaste": {"column": "oppeaste", "type": "categorical"},
    "veebiope": {"column": "veebiope", "type": "categorical"},
}

def get_allowed_values(source_df: pd.DataFrame) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    for key, meta in FILTER_FIELDS.items():
        if meta["type"] == "categorical":
            col = meta["column"]
            uniques = (
                source_df[col]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            values[key] = uniques
    return values

def apply_filters(source_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=source_df.index)
    for key, value in filters.items():
        if key not in FILTER_FIELDS:
            continue
        if value in (None, "", "any", "*"):
            continue
        meta = FILTER_FIELDS[key]
        col = meta["column"]
        if meta["type"] == "numeric":
            if isinstance(value, list):
                nums = [float(v) for v in value if v not in (None, "")]
                if nums:
                    mask &= source_df[col].isin(nums)
            else:
                mask &= source_df[col] == float(value)
        else:
            mask &= source_df[col].astype(str).str.lower() == str(value).lower()
    return source_df[mask].copy()
