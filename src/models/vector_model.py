import numpy as np
import pandas as pd

import math
import random
import json
import gc
import gzip
import pickle

import resource

from tqdm import tqdm
from collections import Counter


class vector_model:
    def __init__(
        self,
        min_df,
        docs,
        index_to_docID,
        docID_to_index,
        term_to_index,
        idf,
        vectors,
        vector_norms,
    ):
        self.min_df = min_df
        self.docs = docs
        self.index_to_docID = index_to_docID
        self.docID_to_index = docID_to_index
        self.term_to_index = term_to_index
        self.idf = idf
        self.vectors = vectors
        self.vector_norms = vector_norms

    @classmethod
    def create_model(cls, documents, min_df=1):
        docs = []
        index_to_docID = {}
        docID_to_index = {}
        for index, docID in enumerate(documents.keys()):
            index_to_docID[index] = docID
            docID_to_index[docID] = index
            docs.append(documents[docID])

        vocab = cls.build_vocab(cls, documents, min_df)

        term_to_index = {}
        for index, term in enumerate(vocab):
            term_to_index[term] = index

        idf = cls.idf_values(cls, docs)

        vectors = np.zeros((len(docs), len(term_to_index)))
        vector_norms = None

        return cls(
            min_df,
            docs,
            index_to_docID,
            docID_to_index,
            term_to_index,
            idf,
            vectors,
            vector_norms,
        )

    @classmethod
    def from_pretrained(cls, path):
        with gzip.GzipFile(
            path + ".TDiDF.matrix.npy.gz", "rb"
        ) as f:
            vectors = np.load(f)

        with gzip.GzipFile(
            path + ".TDiDF.vectors.npy.gz", "rb"
        ) as f:
            vector_norms = np.load(f)

        with gzip.GzipFile(
            path + ".TDiDF.metadata.pkl.gz", "rb"
        ) as f:
            metadata = pickle.load(f)

        min_df = metadata["min_df"]
        index_to_docID = metadata["index_to_docID"]
        docID_to_index = metadata["docID_to_index"]
        term_to_index = metadata["term_to_index"]
        idf = metadata["idf"]
        
        docs = None
        
        return cls(
            min_df,
            docs,
            index_to_docID,
            docID_to_index,
            term_to_index,
            idf,
            vectors,
            vector_norms,
        )

    def fit(self):
        for i, doc in enumerate(self.docs):#tqdm(
            #enumerate(self.docs), desc="Building TF-iDF Matrix", unit=" docs"
        #):
            self.vectors[i] = self.vectorize(doc)
        self.vector_norms = np.linalg.norm(self.vectors, axis=1)

    def build_vocab(self, documents, min_df):
        vocabulary = set()
        term_counter = {}
        for docID in documents.keys():
            text = documents[docID]
            for term in text:
                if term in term_counter.keys():
                    term_counter[term] += 1
                else:
                    term_counter[term] = 1

        for term in term_counter.keys():
            if term_counter[term] >= min_df:
                vocabulary.add(term)

        vocabulary = list(vocabulary)
        vocabulary.sort()

        return vocabulary

    def idf_values(self, docs):
        idf = {}
        term_document_counts = {}

        #for i, doc in tqdm(
        #    enumerate(docs),
        #    desc="Calculating iDF values (per-doc-computation)",
        #    unit=" docs",
        #):
        for i, doc in enumerate(docs):
            doc_terms = set()
            for term in doc:
                if term not in doc_terms:
                    if term in term_document_counts.keys():
                        term_document_counts[term] += 1
                    else:
                        term_document_counts[term] = 1
                    doc_terms.add(term)

        for term in term_document_counts.keys():
            idf[term] = math.log(len(docs) / term_document_counts[term], math.e)

        return idf

    def vectorize(self, doc_terms):
        query_vector = np.zeros((len(self.term_to_index),))
        counts = Counter(doc_terms)
        max_count = counts.most_common(1)[0][1]

        for index, term in enumerate(doc_terms):
            if term not in self.term_to_index.keys():
                continue
            term_index = self.term_to_index[term]
            query_vector[term_index] = self.idf[term] * counts[term] / max_count

        return query_vector

    def cosine_similarity(self, query_vector):
        dot_products = np.dot(self.vectors, query_vector)

        norm_target = np.linalg.norm(query_vector)
        norm_vectors = self.vector_norms
        
        #if norm_target == 0 or norm_vectors == 0:
        #    return np.zeros((len(self.vectors),))

        # Calculate the cosine similarity between the target vector and all vectors in the array
        return dot_products / (norm_target * norm_vectors)

    def find_similar(self, query_terms, topn=1):
        # Calculate the vector of the query document
        query_vector = self.vectorize(query_terms)

        if np.linalg.norm(query_vector) == 0:
            return []
        
        # Calculate cosine similarity between the input doc_vector and all doc_vectors in the model
        similarities = self.cosine_similarity(query_vector)
        
        # Get the indices of the top 'topn' most similar documents
        top_indices = np.argpartition(similarities, -topn)[-topn:]

        # Sort the top indices by similarity score in descending order
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Get the similarity scores and document indices of the top documents
        most_similar_docs = [
            (self.index_to_docID[index], similarities[index]) for index in top_indices
        ]
        
        return most_similar_docs

    def describe(self):
        desc = f"Vector Model (tf-idf)\n"
        desc += f"Number of documents: {len(self.docs)} \n"
        desc += f"Number of terms: {len(self.term_to_index)}\n"
        desc += f"Minimum document frequency: {self.min_df}\n"
        desc += f"Tf-idf shape: {self.vectors.shape}\n"

        return desc

    def save(self, path):
        metaparameters = {
            "min_df": self.min_df,
            "index_to_docID": self.index_to_docID,
            "term_to_index": self.term_to_index,
            "docID_to_index": self.docID_to_index,
            "idf": self.idf,
        }

        with gzip.GzipFile(path + f"mdf{self.min_df}.TDiDF.matrix.npy.gz", "wb") as f:
            np.save(f, self.vectors)

        with gzip.GzipFile(path + f"mdf{self.min_df}.TDiDF.vectors.npy.gz", "wb") as f:
            np.save(f, self.vector_norms)

        with gzip.GzipFile(path + f"mdf{self.min_df}.TDiDF.metadata.pkl.gz", "wb") as f:
            pickle.dump(metaparameters, f)

    def get_document_scores(self, document_ids, query_terms):
        document_vectors = [self.vectors[self.docID_to_index[docID]] for docID in document_ids]
        query_vector = self.model.infer_vector(query_terms)

        query_vector_norm = np.linalg.norm(query_vector)
        document_vectors_norm = np.linalg.norm(document_vectors, axis=1)

        dot_products = np.dot(document_vectors, query_vector)
        document_scores = dot_products / (query_vector_norm * document_vectors_norm)

        return document_scores