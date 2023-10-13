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

def getUsedGBs():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e+9

docs = Preprocessor().load_docs("../../data/dataset/tokenized_corpus.jsonl")
print("Number of documents in corpus: {}".format(len(docs)))

vm = vector_model.create_model(documents=docs, min_df=1700)
vm.fit()
print(vm.describe())
#print(getUsedGBs(), "GBs")

#vm.save("../../models/vm/vm")
#print("Model saved")
