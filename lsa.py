# -*- coding: utf-8 -*-

import numpy as np
from math import *


def build_M(docs):
    """
    take a list of string docs, extracts vector of words and build term-doc matrix
    """

    n_docs = len(docs)
    terms = list(set([item for s in docs for item in s.split(" ")]))
    terms.sort()

    print "term vector is :\n", terms

    n_terms = len(terms)
    M = np.zeros((n_terms, n_docs))

    for i, k in enumerate(terms):
        for j, d in enumerate(docs):
            M[i][j] = 1 if k in d else 0

    return M


def tfidf(M):
    """
    take matrix term-doc with frequencies and return tf-idf instead
    """

    n_terms = M.shape[0]
    n_docs = M.shape[1]

    Mtfidf = np.zeros((n_terms, n_docs))

    for doc in range(n_docs):
        for term in range(n_terms):
            tf=M[term, doc]/M[:,doc].sum()
            idf=log(n_docs/M[term,:].sum())
            Mtfidf[term, doc] = tf*idf

    return Mtfidf


def main():
    docs = [
            "chat poursuit souris",
            "chat souris sont animaux",
            "joue souris clavier",
            "clavier permet écrire ordinateur"
          ]
    M = build_M(docs)
    print M

    MM = tfidf(M)

    print MM

if __name__ == '__main__':
    main()