# -*- coding: utf-8 -*-

import numpy as np
from math import *


def build_M(docs):
    """take a list of string docs, extracts vector of words and build term-doc matrix"""
    n_docs = len(docs)
    terms = list(set([item for s in docs for item in s.split(" ")]))
    terms.sort()

    print "term vector is :\n", terms, "\n"

    n_terms = len(terms)
    M = np.zeros((n_terms, n_docs))

    for i, k in enumerate(terms):
        for j, d in enumerate(docs):
            M[i][j] = 1 if k in d else 0

    return M


def tfidf(M):
    """take matrix term-doc with frequencies and normalize with tf-idf instead"""
    n_terms = M.shape[0]
    n_docs = M.shape[1]
    Mtfidf = np.zeros((n_terms, n_docs))

    for doc in range(n_docs):
        for term in range(n_terms):
            tf = M[term, doc]/M[:, doc].sum()
            idf = log(n_docs/M[term, :].sum(), 2)
            Mtfidf[term, doc] = tf*idf
    return Mtfidf


def multiply(*args):
    """takes numpy arrays and multiplies them"""
    i = 0
    res = 1
    while i < len(args):
        M = args[i]
        i += 1
        res = np.dot(res, M)
    return res


# def test_multiply():
#     a = np.array([[1, 0], [0, 1]])
#     b = np.array([[0, 2], [2, 0]])
#     c = np.array([[0, 1], [1, 0]])
#     d = np.array([[3, 0], [0, 3]])
#     result = np.dot(np.dot(np.dot(a, b), c), d)
#     test = multiply(a, b, c, d)
#     # print "test = ", test
#     # print "result = ", result
#     print np.allclose(test, result)


def main():
    docs = [
        "chat poursuit souris",
        "chat souris sont animaux",
        "joue souris clavier",
        "clavier permet écrire ordinateur"
    ]

    MM = tfidf(build_M(docs))
    U, s, V = np.linalg.svd(MM)
    S = np.zeros(MM.shape)

    # S[:s.size, :s.size] = np.diag(s)
    # if np.allclose(MM, np.dot(U, np.dot(S, V))):
    #     print "SVD OK"

    S[:s.size, :s.size] = np.diag([k if i < 2 else 0 for (i, k) in enumerate(s)])

    print np.dot(U, np.dot(S, V))
    print MM

if __name__ == '__main__':
    # main()
    test_multiply()
