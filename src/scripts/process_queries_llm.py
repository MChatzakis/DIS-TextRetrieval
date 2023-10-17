import json
import sys
import random
import gc

sys.path.append("../")

from models.vector_model import vector_model
from tqdm import tqdm
from preprocessors.preprocessor import Preprocessor
from preprocessors.llm_expander import LLMExpander

import pandas as pd
import numpy as np

import IR_utils

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

random.seed(0)
np.random.seed(0)

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

llmExpander = LLMExpander(model, tokenizer, max_new_tokens=1000)

test_queries_t1 = IR_utils.load_test_queries_t1(
    "../../data/dataset/queries.jsonl", "../../data/task1_test.tsv"
)[0]

test_queries_t2 = IR_utils.load_test_queries_t2(
    "../../data/dataset/queries.jsonl", "../../data/task2_test.tsv"
)[0]

print("Number of queries (t1):", len(test_queries_t1))
print("Number of queries (t2):", len(test_queries_t2))

preprocessor = Preprocessor()

for query_data in tqdm(
    test_queries_t1, desc="Query Preprocessing and Expansion", unit=" queries"
):
    query_text = query_data["text"]
    expanded = llmExpander.expand(query_text)
    query_data["tokens"] = preprocessor.preprocess_query(expanded, expand=False)


preprocessor.save_queries(
    test_queries_t1, "../../data/dataset/test_queries_t1_expanded.jsonl"
)

for query_data in tqdm(
    test_queries_t2, desc="Query Preprocessing and Expansion", unit=" queries"
):
    query_text = query_data["text"]
    expanded = llmExpander.expand(query_text)
    query_data["tokens"] = preprocessor.preprocess_query(expanded, expand=True)

preprocessor.save_queries(
    test_queries_t1, "../../data/dataset/test_queries_t2_expanded.jsonl"
)
