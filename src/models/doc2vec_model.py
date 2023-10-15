from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import numpy as np
import pandas as pd

# from IR_model import IR_model


class Doc2VecModel:
    """
    A wrapper class for Doc2Vec model.
    """

    def __init__(self, tagged_documents, model, normalized_norms):
        self.tagged_documents = tagged_documents
        self.model = model
        self.normalized_norms = normalized_norms  # np.linalg.norm(self.model.dv.get_normed_vectors(), axis=1)

    @classmethod
    def create_model(
        cls,
        documents,
        vector_size=20,
        window=2,
        min_count=1,
        workers=4,
        epochs=100,
    ):
        # Documents must be already tokenized.
        tagged_documents = [
            TaggedDocument(documents[docID], [str(docID)]) for docID in documents.keys()
        ]

        model = Doc2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
        )

        return cls(tagged_documents, model, [])

    @classmethod
    def from_pretrained(cls, path):
        model = Doc2Vec.load(path)
        tagged_documents = []
        for tag in model.dv.index_to_key:
            vector = model.dv[tag]
            tagged_documents.append((tag, vector))

        normalized_norms = np.linalg.norm(model.dv.get_normed_vectors(), axis=1)

        return cls(tagged_documents, model, normalized_norms)

    def fit(self, progress_bar=False):
        self.model.build_vocab(self.tagged_documents)

        if progress_bar:
            with tqdm(total=self.model.epochs, desc="Training") as pbar:
                for epoch in range(self.model.epochs):
                    self.model.train(
                        self.tagged_documents,
                        total_examples=self.model.corpus_count,
                        epochs=1,
                    )
                pbar.update(1)
        else:
            self.model.train(
                self.tagged_documents,
                total_examples=self.model.corpus_count,
                epochs=self.model.epochs,
            )

        if len(self.normalized_norms) == 0:
            self.normalized_norms = np.linalg.norm(
                self.model.dv.get_normed_vectors(), axis=1
            )

    def find_similar(self, query_terms, topk=5, use_buildin=False):
        vector = self.model.infer_vector(query_terms)

        if use_buildin:
            similar_documents = self.model.dv.most_similar(positive=[vector], topn=topk)
        else:
            similar_documents = self.most_similar(vector, topn=topk)

        similar_documents = [
            (int(docID), similarity) for docID, similarity in similar_documents
        ]

        return similar_documents

    def save(self, path):
        self.model.save(path)

    def most_similar(self, query_vector, topn=5):
        # Calculate cosine similarity between the input doc_vector and all doc_vectors in the model
        similarities = self.cosine_similarity(query_vector)

        # Get the indices of the top 'topn' most similar documents
        top_indices = np.argpartition(similarities, -topn)[-topn:]

        # Sort the top indices by similarity score in descending order
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Get the similarity scores and document indices of the top documents
        most_similar_docs = [
            (self.model.dv.index_to_key[index], similarities[index])
            for index in top_indices
        ]

        return most_similar_docs

    def cosine_similarity(self, query_vector):
        dot_products = np.dot(self.model.dv.get_normed_vectors(), query_vector)

        norm_target = np.linalg.norm(query_vector)
        norm_vectors = self.normalized_norms
        # norm_vectors = self.model.dv.norms # Does not work properly..

        # Calculate the cosine similarity between the target vector and all vectors in the array
        return dot_products / (norm_target * norm_vectors)

    def get_document_scores(self, document_ids, query_terms):
        # model.dv[tag]
        document_vectors = [self.model.dv[str(docID)] for docID in document_ids]
        query_vector = self.model.infer_vector(query_terms)

        query_vector_norm = np.linalg.norm(query_vector)
        document_vectors_norm = np.linalg.norm(document_vectors, axis=1)

        dot_products = np.dot(document_vectors, query_vector)
        document_scores = dot_products / (query_vector_norm * document_vectors_norm)

        return document_scores


def SimpleExample():
    docs = {
        29: "This is the first document",
        14: "This is the second document",
        45: "And the third one",
        12: "Is this the first document?",
        90: "The last document?",
        25: "Nope, this is the last document",
    }

    d2v = Doc2VecModel.create_model(
        docs, vector_size=20, window=20, min_count=1, workers=16, epochs=3
    )
    print("Processed Documents:", d2v.tagged_documents)

    d2v.fit()
    print("Model Fitted. Number of documents:", len(d2v.model.dv))

    similar_documents = d2v.find_similar(
        query="This is the first document", topk=len(d2v.model.dv)
    )
    print("Similar Documents (id, similarity):", similar_documents)

    return


def CompareBuildinAndCustomMostSimilar(
    query: str, d2v: Doc2VecModel, print_enable=False
):
    print(f"Query: {query}")

    build_in_similar_documents = d2v.find_similar(
        query=query, topk=len(d2v.model.dv), use_buildin=True
    )

    custom_similar_documents = d2v.find_similar(
        query=query,
        topk=len(d2v.model.dv),
        use_buildin=False,
    )

    custom_ids = [docId for docId, _ in custom_similar_documents]
    buildin_ids = [docId for docId, _ in build_in_similar_documents]

    custom_scores = [round(score, 3) for _, score in custom_similar_documents]
    buildin_scores = [round(score, 3) for _, score in build_in_similar_documents]

    if print_enable:
        print(" Custom DocIdIs: ", custom_ids)
        print(" Buildin DocIdIs:", buildin_ids)

    assert (
        custom_ids == buildin_ids
    ), f"Custom and Buildin DocIds are not equal for query: {query}: {custom_ids} != {buildin_ids}🛑"

    if print_enable:
        print(" Custom Similarities:", custom_scores)
        print(" Buildin Similarities:", buildin_scores)


def CheckCustomMostSimilar():
    docs = {
        29: "This is the first document",
        14: "This is the second document",
        45: "And the third one",
        12: "Is this the first document?",
        90: "The last document?",
        25: "Nope, this is the last document",
        11: "Nah, this is the last document",
        90: "Hello, this is the last document",
    }

    queries = [
        "This is query 1",
        "This is query 2",
        "This is query 3",
        "This is query 4",
        "This is query 5",
        "This is query 6",
        "This is query 7",
        "This is query 8",
        "This is query 9",
        "This is query 10",
    ]

    d2v = Doc2VecModel.create_model(
        docs, vector_size=20, window=20, min_count=1, workers=16, epochs=3
    )

    d2v.fit()
    print("Model Fitted. Number of documents:", len(d2v.model.dv))

    for query in queries:
        CompareBuildinAndCustomMostSimilar(query, d2v, print_enable=True)
        print()

    print("All Custom and Buildin DocIds are equal!✅")

    return


if __name__ == "__main__":
    CheckCustomMostSimilar()
