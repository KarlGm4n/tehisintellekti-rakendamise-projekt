import csv
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from filters_utils import get_allowed_values, apply_filters

BENCHMARK_CASES_PATH = Path("benchmark_data/testjuhtumid.csv")

class BenchmarkCase:
    def __init__(self, query: str, expected_ids: List[str], expects_empty: bool):
        self.query = query
        self.expected_ids = expected_ids
        self.expects_empty = expects_empty

class BenchmarkResult:
    def __init__(self, case: BenchmarkCase, found_ids: List[str]):
        self.case = case
        self.found_ids = found_ids
        self.passed = self._check_pass()

    def _check_pass(self):
        if self.case.expects_empty:
            return len(self.found_ids) == 0
        return all(cid in self.found_ids for cid in self.case.expected_ids)

def load_benchmark_cases(path=BENCHMARK_CASES_PATH):
    cases = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            query = row[0].strip()
            expected_raw = row[1].strip()
            if expected_raw == "-":
                expected_ids = []
                expects_empty = True
            else:
                expected_ids = [x.strip() for x in expected_raw.split(",") if x.strip()]
                expects_empty = False
            cases.append(BenchmarkCase(query, expected_ids, expects_empty))
    return cases

def run_benchmark(embedder, df, embeddings_df, n_cases=0):
    cases = load_benchmark_cases()
    if n_cases > 0:
        cases = cases[:n_cases]
    merged_df = pd.merge(df, embeddings_df, on="unique_ID")
    results = []
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    for case in cases:
        filtered_df = merged_df.copy()  # No filters for benchmark
        query_vec = embedder.encode([case.query])[0]
        embeddings = np.stack(filtered_df["embedding"])
        filtered_df["score"] = cosine_similarity([query_vec], embeddings)[0]
        top_ids = filtered_df.sort_values("score", ascending=False)["unique_ID"].head(5).tolist()
        result = BenchmarkResult(case, top_ids)
        results.append(result)
    return results
