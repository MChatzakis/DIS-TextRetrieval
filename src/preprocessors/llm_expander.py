import pandas as pd
import numpy as np

import json
import random
import string
import Levenshtein

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from tqdm import tqdm

class LLMExpander():
    """
    The expand method of LLMExpander should load a query before processsing and tokenizing.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def expand(self, query_text: str):
        form_query = self.formulate(query_text)
        inputs = self.tokenizer(form_query, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        
        expanded_query = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return expanded_query
    
    def formulate(self, query_text: str):
        form_query = "Answer the following question:\n"
        form_query += query_text + "\n"
        form_query += "Give the rationale before answering."
        return form_query