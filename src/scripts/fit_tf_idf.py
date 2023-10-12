import json
import sys
import random
import gc

sys.path.append("../")

from models.vector_model import vector_model
from tqdm import tqdm
from preprocessors.preprocessor import Preprocessor

import pandas as pd
import numpy as np

import IR_utils

random.seed(0)
np.random.seed(0)

data_path = "../../data/dataset/corpus.jsonl"
docs = IR_utils.load_document_corpus(data_path=data_path)

print("Number of documents in corpus: {}".format(len(docs)))

preprocessor = Preprocessor()
tokenized_docs = preprocessor.preprocess(docs)

vm = vector_model.create_model(documents=tokenized_docs, min_df=50)

vm.fit()
print(vm.describe())

vm.save("../../models/vm/vm")

print("Model saved")
