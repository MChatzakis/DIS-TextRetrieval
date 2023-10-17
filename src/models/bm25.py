import math
from six import iteritems
from six.moves import range
import numpy as np
import heapq
from collections.abc import Iterable
from collections import defaultdict, Counter



class bm25(object):

    def __init__(self, corpus_ids, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.avg_doc_length = 0
        self.doc_frequencies = []
        self.idf = {}
        self.doc_lengths = []
        self.corpus = corpus
        self.corpus_ids = corpus_ids

    def fit(self):
        term_to_freq = defaultdict(int)  
        total_length = 0

        for document in self.corpus:
            self.corpus_size += 1
            doc_length = len(document)
            total_length += doc_length
            self.doc_lengths.append(doc_length)

            frequencies = Counter(document)
            self.doc_frequencies.append(frequencies)

            for term, freq in frequencies.items():
                term_to_freq[term] += 1

        self.avg_doc_length = float(total_length) / self.corpus_size
        self.nd = term_to_freq

        idf_sum = 0
        idf_len = 0
        negative_idfs = []

        for word, freq in term_to_freq.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_len += 1
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        self.average_idf = idf_sum / idf_len
        eps = self.epsilon * self.average_idf
        self.idf.update({word: eps for word in negative_idfs})

        document_score = {}
        for i, document in enumerate(self.corpus):
            doc_freqs = self.doc_frequencies[i]
            for word in document:
                if word not in doc_freqs:
                    continue
                score = (self.idf[word] * doc_freqs[word] * (self.k1 + 1)
                          / (doc_freqs[word] + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avg_doc_length)))
                if word not in document_score:
                    document_score[word] = {i: round(score, 2)}
                else:
                    document_score[word].update({i: round(score, 2)})
        self.document_score = document_score


    def compute_similarity(self, query, doc):
        score = 0
        doc_freqs = Counter(query)
        freq = 1
        default_idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
        for word in doc:
            if word not in doc_freqs:
                continue
            score += (self.idf.get(word,default_idf) * doc_freqs[word] * (self.k1 + 1)
                      / (doc_freqs[word] + self.k1 * (1 - self.b + self.b * len(query) / self.avg_doc_length)))
        return score

        
    def get_top_k_documents(self,document,k=1):
        score_overall = {}
        for word in document:
            if word not in self.document_score:
                continue
            for key, value in self.document_score[word].items():
                score_overall[key] = score_overall.get(key, 0) + value

        k_keys_sorted = heapq.nlargest(k, score_overall,key=score_overall.get)
        return [(score_overall.get(item,None), self.corpus_ids[item], self.corpus[item]) for item in k_keys_sorted]