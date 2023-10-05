from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import numpy as np
import pandas as pd


class Doc2VecModel:
    """
    A wrapper class for Doc2Vec model.
    """

    def __init__(
        self, documents, vector_size=20, window=2, min_count=1, workers=4, epochs=100
    ):
        self.tagged_documents = [
            TaggedDocument(documents[docID].split(), [str(docID)])
            for docID in documents.keys()
        ]

        self.model = Doc2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
        )

    def __init__(self, path: str):
        self.load(path)
        
        self.tagged_documents = []
        for tag in self.model.dv.index_to_key:
            vector = self.model.dv[tag]
            self.tagged_documents.append((tag, vector))
        

    def fit(self):
        self.model.build_vocab(self.tagged_documents)

        # self.model.train(
        #    self.tagged_documents,
        #    total_examples=self.model.corpus_count,
        #    epochs=self.model.epochs,
        # )

        with tqdm(total=self.model.epochs, desc="Training") as pbar:
            for epoch in range(self.model.epochs):
                self.model.train(
                    self.tagged_documents,
                    total_examples=self.model.corpus_count,
                    epochs=1,
                )
                pbar.update(1)

    def find_similar(self, query, topk=5):
        vector = self.model.infer_vector(query.split())
        similar_documents = self.model.docvecs.most_similar(
            positive=[vector], topn=topk
        )
        
        similar_documents = [(int(docID), similarity) for docID, similarity in similar_documents]
        
        return similar_documents

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = Doc2Vec.load(path)

    def most_similar_custom(self, query_vector, topn=10):
        # Calculate cosine similarity between the input doc_vector and all doc_vectors in the model
        similarities = np.dot(self.model.dv.vectors_norm, query_vector)

        # Sort the documents by similarity score in descending order
        most_similar_docs = sorted(enumerate(similarities), key=lambda item: -item[1])

        return most_similar_docs[:topn]


def SimpleExample():
    docs = {
        29: "This is the first document",
        14: "This is the second document",
        45: "And the third one",
    }

    d2v = Doc2VecModel(
        docs, vector_size=20, window=20, min_count=1, workers=4, epochs=100
    )
    print("Processed Documents:", d2v.tagged_documents)

    d2v.fit()
    print("Model Fitted. Number of documents:", len(d2v.model.dv))

    similar_documents = d2v.find_similar(
        query="This is the first document", topk=len(d2v.model.dv)
    )
    print("Similar Documents (id, similarity):", similar_documents)

    return


def CheckCustomMostSimilar():
    pass
