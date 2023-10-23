import pandas as pd
import numpy as np
import json
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from tqdm import tqdm

nltk.download("stopwords")
nltk.download("punkt")


class Preprocessor:
    def __init__(self, expander=None):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)
        self.expander = expander

    def preprocess(self, documents):
        tokenized_docs = []
        if isinstance(documents, list):
            tokenized_docs = self.preprocess_document_list(documents)
        elif isinstance(documents, dict):
            tokenized_docs = self.preprocess_document_dict(documents)
        else:
            raise TypeError("Documents must be either a list or a dictionary")

        return tokenized_docs

    def preprocess_query(self, query, expand=False):
        query = self.tolowercase(query)
        query = self.remove_punctuation(query)

        query_tokens = self.tokenize(query)
        query_tokens = self.remove_stopwords(query_tokens)

        if expand:
            query_tokens = self.expand(query_tokens)

        query_tokens = self.stem(query_tokens)

        return query_tokens

    def preprocess_document_list(self, document_list):
        tokenized_docs = []
        for i in tqdm(range(len(document_list))):
            tokenized_docs.append(self.preprocess_doc(document_list[i]))
        return tokenized_docs

    def preprocess_document_dict(self, document_dict):
        tokenized_docs = {}
        for doc_id in tqdm(document_dict.keys()):
            document = document_dict[doc_id]
            tokenized_docs[doc_id] = self.preprocess_doc(document)
        return tokenized_docs

    def preprocess_doc(self, document):
        document = self.tolowercase(document)
        document = self.remove_punctuation(document)

        document_tokens = self.tokenize(document)
        document_tokens = self.remove_stopwords(document_tokens)
        document_tokens = self.stem(document_tokens)

        return document_tokens

    def tolowercase(self, document):
        return document.lower()

    def remove_punctuation(self, document):
        return "".join([char for char in document if char not in self.punctuation])

    def tokenize(self, document):
        return word_tokenize(document)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]

    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def save_docs(self, docs, path):
        with open(path, "w") as jsonl_file:
            for docID in docs:
                doc_data = {"_id": str(docID), "tokens": docs[docID]}
                json_line = json.dumps(doc_data)
                jsonl_file.write(json_line + "\n")

    def save_queries(self, queries, path):
        with open(path, "w") as jsonl_file:
            for query in queries:
                doc_data = {
                    "query_index": int(query["id"]),
                    "tokens": list(query["tokens"]),
                    "query_id": int(query["query_id"]),
                }
                # print(doc_data)

                json_line = json.dumps(doc_data)
                jsonl_file.write(json_line + "\n")

    def load_docs(self, path):
        raw_queries = {}
        with open(path, "r") as file:
            for line in file:
                data = json.loads(line)
                raw_queries[data["_id"]] = data["tokens"]

        return raw_queries

    def load_queries(self, path):
        queries = []

        with open(path, "r") as file:
            for line in file:
                data = json.loads(line)
                queries.append(
                    {
                        "query_index": data["query_index"],
                        "query_id": data["query_id"],
                        "tokens": data["tokens"],
                    }
                )

        return queries

    def expand(self, terms):
        return self.expander.expand(terms)
