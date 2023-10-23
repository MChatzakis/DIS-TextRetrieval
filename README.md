# A Text Retrieval Approach Using BM25+ and BERT Reranking
Manos Chatzakis, Hind El-Bouchrifi and Lluka Stojollari
{emmanouil.chatzakis, hind.elbouchrifi, lluka.stojollari}@epfl.ch
Distributed Information Systems, Text Retrieval Project, EPFL

# Context
This repository 

# Contents
    -src/
        - models/
            - Code for all implemented information retrieval models (TF-iDF, Doc2vec, DSSM, BM25)
        - preprocessors/
            - Code for all preprocessing and query expansion techniques
        - scripts/
            - Pipelines for model training, text preprocessing, query expansion
        - testing/
            - Examples of model usage
        *IR_Utils.ipynb: Utility functions for IR
        *kaggle.ipynb: Final competition notebook, implementing our final best scoring approach
    *report/
        *report.pdf: A brief report describing our approach
    *data/
        *Test and training query splits

# Data Availability
Test and training query splits are present under data/ directory.