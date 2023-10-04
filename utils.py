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
    retrieved_docs = set(retrieved_docs)
    relevant_docs = set(relevant_docs)

    if len(relevant_docs) == 0:
        return 0

    top_X_retrieved_docs = retrieved_docs
    if len(retrieved_docs) >= K:
        top_X_retrieved_docs = retrieved_docs[:K]

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
