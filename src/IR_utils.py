import numpy as np
import pandas as pd

import json

def recall_K(retrieved_docs, relevant_docs, K=10):
    """
    Calculates recall@X

    Args:
        retrieved_docs (list): List of documents retrieved by the model
        relevant_docs (list): List of relevant documents
        K (int, optional): Number of documents to consider. Defaults to 10. (Top K

    Returns:
        number: recall@X value
    """
    if len(relevant_docs) == 0:
        return 0
    
    #retrieved_docs = set(retrieved_docs)
    #relevant_docs = set(relevant_docs)

    top_X_retrieved_docs = retrieved_docs
    if len(retrieved_docs) >= K:
        top_X_retrieved_docs = retrieved_docs[:K]

    top_X_retrieved_docs = set(top_X_retrieved_docs)
    relevant_docs = set(relevant_docs)
    
    relevant_retrieved_docs = top_X_retrieved_docs.intersection(relevant_docs)

    return len(relevant_retrieved_docs) / len(relevant_docs)


def precision_K(retrieved_docs, relevant_docs, K=10):
    """
    Calculates precision@X

    Args:
        retrieved_docs (list): List of documents retrieved by the model
        relevant_docs (list): List of relevant documents
        K (int, optional): Number of documents to consider. Defaults to 10. (Top K

    Returns:
        number: precision@X value
    """
    if len(relevant_docs) == 0:
        return 0

    correct_predict = set(retrieved_docs[:K]).intersection(set(relevant_docs))
    return len(correct_predict) / K


def cosine_similarity(vectors, query):
    """
    Calculates cosine similarity between two vectors

    Args:
        X : Matrix X (np.array of vectors)
        y : Vector y (np.array of query vector)

    Returns:
        number: cosine similarity between X and y
    """
    dot_products = np.dot(vectors, query)
    
    norm_target = np.linalg.norm(query)
    norm_vectors = np.linalg.norm(vectors, axis=1)

    # Calculate the cosine similarity between the target vector and all vectors in the array
    return dot_products / (norm_target * norm_vectors)


def load_document_corpus(data_path, max_docs = -1):
    docs = {}
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            docs[data["_id"]] = data["text"]

            if max_docs > 0 and len(docs) == max_docs:
                break

    return docs