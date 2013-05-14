# -*- coding: utf-8 -*-

import numpy as np
from math import *


def multiply(*args):
    """takes custom number of numpy arrays and multiplies them"""
    i = 0
    res = 1
    while i < len(args):
        M = args[i]
        i += 1
        res = np.dot(res, M)
    return res


def cosine(x, y):
    """returns cosine of angle between x et y"""
    return np.dot(x, y)/sqrt(np.dot(x, x))/sqrt(np.dot(y, y))


def build_terms(docs, stopwords):
    """build dictionary of words from docs, ignoring list of stopwords """
    terms = list(set([item.lower() for s in docs for item in s.split(" ")
                  if item.lower() not in stopwords]))
    terms.sort()
    terms = dict((key, value) for (value, key) in enumerate(terms))
    return terms

def build_M(terms, docs):
    """take a list of string docs, and a dict for terms
       extracts vector of words and build term-doc matrix"""
    docs_split = [doc_list.lower().split(" ") for doc_list in docs]
    n_docs = len(docs)
    n_terms = len(terms)

    M = np.zeros((n_terms, n_docs))

    for i, doc in enumerate(docs_split):
        for term in doc:
            if term in terms : M[terms.get(term), i] += 1
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




# def main():

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]

stopwords = set(['for', 'a', 'of', 'the', 'and', 'to', 'in'])

# terms = build_terms(documents, stopwords)

# using custom terms
terms =  ["human",
        "interface",
        "computer",
        "user",
        "system",
        "response",
        "time",
        "EPS",
        "survey",
        "trees",
        "graph",
        "minors"]

terms = dict((key,value) for (value,key) in enumerate(terms))


M = build_M(terms, documents)
print M
# MM = tfidf(M)

# U, s, V = np.linalg.svd(M)
# S = np.zeros(MM.shape)

# S[:s.size, :s.size] = np.diag(s)
# if np.allclose(MM, np.dot(U, np.dot(S, V))):
#     print "SVD OK"

# S[:s.size, :s.size] = np.diag([k if i < 2 else 0 for (i, k) in enumerate(s)])

# print np.dot(U, np.dot(S, V))
# print MM

#print topics
# print "topic 1"
# print [(i, j) for (i, j) in zip(np.dot(U, S)[:, 0], terms) if abs(i)>0.01]  # if abs(i) > 0.1]
# print ""
# print "topic 2"
# print [(i, j) for (i, j) in zip(np.dot(U, S)[:, 1], terms) if abs(i)>0.01]  #if abs(i) > 0.1]


# if __name__ == '__main__':
#     main()
