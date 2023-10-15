import json
import sys
import random

sys.path.append("../")

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from models.doc2vec_model import Doc2VecModel, CompareBuildinAndCustomMostSimilar
from tqdm import tqdm
from preprocessors.preprocessor import Preprocessor

import pandas as pd
import numpy as np

import IR_utils

random.seed(0)
np.random.seed(0)

docs = Preprocessor().load_docs("../../data/dataset/tokenized_corpus.jsonl")
print("Number of documents in corpus: {}".format(len(docs)))

vector_size = 50
window = 30
min_count = 5
workers = 16
epochs = 200

d2v = Doc2VecModel.create_model(
    documents=docs,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=workers,
    epochs=epochs,
)

d2v.fit(progress_bar=False)

d2v.save(
    f"../../models/d2v/doc2vec.docs{len(d2v.model.dv)}.vs{vector_size}.win{window}.min{min_count}.ep{epochs}.model"
)
