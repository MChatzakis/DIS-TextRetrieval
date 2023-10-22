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


QUERIES_EXPANDED = 0
EXPANSION_TH = 2
MODEL_TYPE = "small"
MAX_NEW_TOKENS = 20

model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{MODEL_TYPE}")
tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{MODEL_TYPE}")

llmExpander = LLMExpander(model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)

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
    query_terms = preprocessor.preprocess_query(query_text)
    if (len(query_terms) <= EXPANSION_TH):
        old_query_terms = query_terms
        query_terms = preprocessor.preprocess_query(llmExpander.expand(query_text))
        QUERIES_EXPANDED += 1
    
    query_data["tokens"] = query_terms


preprocessor.save_queries(
    test_queries_t1, f"../../data/dataset/test_queries_t1_expanded_{MAX_NEW_TOKENS}_{MODEL_TYPE}.jsonl"
)

for query_data in tqdm(
    test_queries_t2, desc="Query Preprocessing and Expansion", unit=" queries"
):
    query_text = query_data["text"]
    query_terms = preprocessor.preprocess_query(query_text)
    if (len(query_terms) <= EXPANSION_TH):
        old_query_terms = query_terms
        query_terms = preprocessor.preprocess_query(llmExpander.expand(query_text))
        QUERIES_EXPANDED += 1
        
    query_data["tokens"] = query_terms

preprocessor.save_queries(
    test_queries_t2, f"../../data/dataset/test_queries_t2_expanded_{MAX_NEW_TOKENS}_{MODEL_TYPE}.jsonl"
)

print("Queries expanded:", QUERIES_EXPANDED)
