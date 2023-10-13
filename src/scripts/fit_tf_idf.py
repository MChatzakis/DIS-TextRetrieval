import sys
import gc
sys.path.append("../")

import random
import resource

import pandas as pd
import numpy as np

from models.vector_model import vector_model
from preprocessors.preprocessor import Preprocessor

random.seed(0)
np.random.seed(0)

docs = Preprocessor().load_docs("../../data/dataset/tokenized_corpus.jsonl")
print("Number of documents in corpus: {}".format(len(docs)))

vm = vector_model.create_model(documents=docs, min_df=4000)
vm.fit()
vm.save("../../models/vm/")

