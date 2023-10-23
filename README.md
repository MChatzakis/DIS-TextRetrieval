# 🎉 A Text Retrieval Approach Using BM25+ and BERT Reranking 🎉
Manos Chatzakis, Hind El-Bouchrifi and Lluka Stojollari

{emmanouil.chatzakis, hind.elbouchrifi, lluka.stojollari}@epfl.ch

Distributed Information Systems, Text Retrieval Project, EPFL

## Context
This repository contains a solution to a [Kaggle](https://www.kaggle.com/competitions/dis-project-1-text-retrieval/) competition of Distributed Information Systems course of EPFL. The competition is about Text Retrieval, where models are automatically evaluated by their document retrieval and reranking power.

## Approach
We describe our approach in our report (report/Report.pdf), as well us our final submission notebook (src/kaggle.ipynb). Briefly, we utilize BM25+ to retrieve a big set of relevant documents and rerank them using BERT Sentence embeddings.

## Contents
- src/
    - models/
        - Code for all implemented information retrieval models (TF-iDF, Doc2vec, DSSM, BM25)
    - preprocessors/
        - Code for all preprocessing and query expansion techniques
    - scripts/
        - Pipelines for model training, text preprocessing, query expansion
    - testing/
            - Examples of model usage
    - IR_Utils.ipynb: Utility functions for IR
    - kaggle.ipynb: Final competition notebook, implementing our final best scoring approach

- report/
    - report.pdf: A brief report describing our approach

- data/
    - Test and training query splits

## Data Availability
Test and training query splits are present under data/ directory. Every other needed file is available at [Google Drive](https://drive.google.com/drive/folders/1Vw6yYoB8Akq_kde3RIS4y9HQdMXjih07)