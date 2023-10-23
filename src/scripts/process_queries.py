import json
import sys
import random
import gc

sys.path.append("../")

from models.vector_model import vector_model
from tqdm import tqdm
from preprocessors.preprocessor import Preprocessor
from preprocessors.synonym_expander import Expander

import pandas as pd
import numpy as np

import IR_utils

random.seed(0)
np.random.seed(0)

preprocessor = Preprocessor(expander=Expander())

test_queries_t1 = IR_utils.load_test_queries_t1(
    "../../data/dataset/queries.jsonl", "../../data/task1_test.tsv"
)[0]

test_queries_t2 = IR_utils.load_test_queries_t2(
    "../../data/dataset/queries.jsonl", "../../data/task2_test.tsv"
)[0]

print("Number of queries (t1):", len(test_queries_t1))
print("Number of queries (t2):", len(test_queries_t2))

for query_data in tqdm(
    test_queries_t1, desc="Query Preprocessing and Expansion", unit=" queries"
):
    query_text = query_data["text"]
    query_data["tokens"] = preprocessor.preprocess_query(query_text, expand=False)

preprocessor.save_queries(test_queries_t1, "../../data/dataset/test_queries_t1.jsonl")

for query_data in tqdm(
    test_queries_t2, desc="Query Preprocessing and Expansion", unit=" queries"
):
    query_text = query_data["text"]
    query_data["tokens"] = preprocessor.preprocess_query(query_text, expand=True)

preprocessor.save_queries(test_queries_t1, "../../data/dataset/test_queries_t2.jsonl")
