import pandas as pd
import numpy as np

import json
import random
import string
import Levenshtein

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from tqdm import tqdm

nltk.download("wordnet")

random.seed(0)


class Expander:
    def __init__(self, add_synonym_prob=0.5, levenshtein_th = 2):
        self.add_synonym_prob = add_synonym_prob
        self.levenshtein_th = levenshtein_th

    def expand(self, word_list):
        expanded_word_list = []
        for word in word_list:
            synonym = self.get_synonym(word)
            synonym_lower = synonym.lower()
            if (
                random.random() < self.add_synonym_prob
                and Levenshtein.distance(synonym_lower, word) > self.levenshtein_th
            ):
                expanded_word_list.append(synonym_lower)
                #print("Synonym added: {} -> {}".format(word, synonym))

        return word_list + expanded_word_list

    def get_synonym(self, word):
        synonyms = set()

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name().isalpha():
                    synonyms.add(lemma.name())
                    break

        synonyms = list(synonyms)
        return synonyms[0] if len(synonyms) > 0 else word
