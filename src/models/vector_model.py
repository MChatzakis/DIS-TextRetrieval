import numpy as np
import pandas as pd

import math
import random

from tqdm import tqdm
from collections import Counter


class vector_model():
    def __init__(self, documents, min_df=1):
        self.min_df = min_df

        self.index_to_docID = {}
        self.docID_to_index = {}
        self.docs = []

        for index, docID in enumerate(documents.keys()):
            self.docID_to_index[docID] = index
            self.index_to_docID[index] = docID
            self.docs.append(documents[docID])

        self.vocab = self.build_vocab(documents)

        self.index_to_term = {}
        self.term_to_index = {}
        for index, term in enumerate(self.vocab):
            self.term_to_index[term] = index
            self.index_to_term[index] = term

        self.idf = self.idf_values()

        self.vectors = np.zeros((len(self.docs), len(self.vocab)))
        self.vector_norms = np.zeros((len(self.docs),))

    def fit(self):
        for i, doc in tqdm(enumerate(self.docs)):
            self.vectors[i] = self.vectorize(doc)
        self.vector_norms = np.linalg.norm(self.vectors, axis=1)

    def build_vocab(self, documents):
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
            if term_counter[term] >= self.min_df:
                vocabulary.add(term)

        vocabulary = list(vocabulary)
        vocabulary.sort()

        return vocabulary

    def idf_values(self):
        idf = {}
        term_document_counts = {}

        for i, doc in enumerate(self.docs):
            doc_terms = set()
            for term in doc:
                if term not in doc_terms:
                    if term in term_document_counts.keys():
                        term_document_counts[term] += 1
                    else:
                        term_document_counts[term] = 1
                    doc_terms.add(term)

        for term in term_document_counts.keys():
            idf[term] = math.log(len(self.docs) / term_document_counts[term], math.e)

        return idf

    def vectorize(self, doc_terms):
        query_vector = np.zeros((len(self.vocab),))
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

        # Calculate the cosine similarity between the target vector and all vectors in the array
        return dot_products / (norm_target * norm_vectors)

    def find_similar(self, query_terms, topn=5):
        # Calculate the vector of the query document
        query_vector = self.vectorize(query_terms)

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

    def generate_submission_file(self):
        raise NotImplementedError
    
    def describe(self):
        desc = f"Vector Model (tf-idf)\n"
        desc += f"Number of documents: {len(self.docs)}\n"
        desc += f"Number of terms: {len(self.vocab)}\n"
        desc += f"Minimum document frequency: {self.min_df}\n"
        desc += f"Tf-idf shape: {self.vectors.shape}\n"
        
        return desc
        